import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math

from train_teq import OpticalDataset, SinusoidalPE, count_parameters

# ================= KAN-Transformer 配置参数 =================
CONFIG = {
    'window_size': 21,
    'sps': 2,
    'd_model': 16,
    'nhead': 2,
    'dim_feedforward': 8,     # KAN FFN 隐藏维度 (因 KAN 自身参数更多, 用更小的隐藏层)
    'grid_size': 3,           # B-spline 网格区间数
    'spline_order': 3,        # B-spline 阶数 (3=三次样条)
    'num_passes': 2,          # 权重共享循环次数
    'use_weight_sharing': True,
    'use_center_token': True,
    'use_sinusoidal_pe': True,
    'batch_size': 256,
    'epochs': 150,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ================= KAN 线性层 (B-spline 可学习激活) =================
class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network 线性层。
    与标准 Linear 的区别：每条"边"上有一个可学习的 B-spline 激活函数，
    输出 = W_base · SiLU(x)  +  Σ c_i · B_i(x)
    参考: KAN (ICLR 2025), KAT (ICLR 2025)
    """

    def __init__(self, in_features, out_features,
                 grid_size=3, spline_order=3, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1,
                             dtype=torch.float) * h + grid_range[0])
        self.register_buffer('grid', grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        num_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, num_basis) * 0.1
        )

        self.base_activation = nn.SiLU()

    def compute_bspline_basis(self, x):
        """Cox-de Boor 递推计算 B-spline 基函数。
        x: [batch, in_features] -> [batch, in_features, num_basis]
        """
        grid = self.grid
        x = x.unsqueeze(-1)

        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            left = ((x - grid[:-(k + 1)])
                    / (grid[k:-1] - grid[:-(k + 1)] + 1e-8)
                    * bases[..., :-1])
            right = ((grid[k + 1:] - x)
                     / (grid[k + 1:] - grid[1:-k] + 1e-8)
                     * bases[..., 1:])
            bases = left + right

        return bases

    def forward(self, x):
        shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        base_out = F.linear(self.base_activation(x_2d), self.base_weight)
        spline_basis = self.compute_bspline_basis(x_2d)
        spline_out = torch.einsum('bik,oik->bo', spline_basis, self.spline_weight)

        return (base_out + spline_out).reshape(*shape[:-1], self.out_features)


# ================= KAN Encoder Layer (替换标准 FFN) =================
class KANEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer，其中 FFN 被替换为 KAN 层。
    结构: Self-Attention → Add&Norm → KAN FFN → Add&Norm
    这是 KAT (ICLR 2025) 的核心思想。
    """

    def __init__(self, d_model, nhead, dim_feedforward,
                 grid_size=3, spline_order=3, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=dropout
        )
        self.kan1 = KANLinear(d_model, dim_feedforward, grid_size, spline_order)
        self.kan2 = KANLinear(dim_feedforward, d_model, grid_size, spline_order)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout(src2))
        src2 = self.kan2(self.kan1(src))
        src = self.norm2(src + self.dropout(src2))
        return src


# ================= KAN-Transformer 均衡器 =================
class KANTransformerEQ(nn.Module):
    """
    KAN-Transformer 均衡器：将 Transformer 中的 MLP/FFN 替换为 KAN 层。
    理论动机：KAN 的可学习 B-spline 激活函数能更好地拟合光纤信道中的
    非线性损伤（色散 + 平方律检测 + 带宽限制产生的非线性ISI）。
    """

    def __init__(self, input_dim, d_model, nhead, dim_feedforward,
                 grid_size=3, spline_order=3, num_passes=2,
                 use_weight_sharing=True, use_center_token=True,
                 use_sinusoidal_pe=True):
        super().__init__()
        self.use_center_token = use_center_token
        self.num_passes = num_passes
        self.center_idx = input_dim // 2

        self.input_proj = nn.Linear(1, d_model)

        if use_sinusoidal_pe:
            self.pos_encoder = SinusoidalPE(d_model, max_len=input_dim)
        else:
            self.pos_encoder = nn.Parameter(
                torch.randn(1, input_dim, d_model) * 0.02
            )

        self.encoder_layer = KANEncoderLayer(
            d_model, nhead, dim_feedforward, grid_size, spline_order
        )

        if use_center_token:
            self.head = nn.Linear(d_model, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(d_model * input_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1)
            )

    def forward(self, src):
        x = self.input_proj(src)

        if isinstance(self.pos_encoder, SinusoidalPE):
            x = self.pos_encoder(x)
        else:
            x = x + self.pos_encoder[:, :x.size(1), :]

        for _ in range(self.num_passes):
            x = self.encoder_layer(x)

        if self.use_center_token:
            x = x[:, self.center_idx, :]
        else:
            x = x.reshape(x.size(0), -1)

        return self.head(x)


# ================= 训练流程 =================
def train():
    print(f"Running on {CONFIG['device']}")
    print("模型: KAN-Transformer Equalizer (参考 KAT, ICLR 2025)")

    data = scipy.io.loadmat('dataset_for_python.mat')
    rx_train = data['rx_train_export'].flatten()
    if np.iscomplexobj(rx_train):
        print("检测到复数信号，取模用于PAM4 IM/DD处理")
        rx_train = np.abs(rx_train)

    symb_train = data['symb_train_export'].flatten()

    rx_mean = np.mean(rx_train)
    rx_std = np.std(rx_train)
    print(f"全局统计: rx_mean={rx_mean:.4f}, rx_std={rx_std:.4f}")

    train_dataset = OpticalDataset(
        rx_train, symb_train, CONFIG['window_size'], CONFIG['sps'],
        rx_mean=rx_mean, rx_std=rx_std
    )
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0
    )

    model = KANTransformerEQ(
        input_dim=CONFIG['window_size'],
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        dim_feedforward=CONFIG['dim_feedforward'],
        grid_size=CONFIG['grid_size'],
        spline_order=CONFIG['spline_order'],
        num_passes=CONFIG['num_passes'],
        use_weight_sharing=CONFIG['use_weight_sharing'],
        use_center_token=CONFIG['use_center_token'],
        use_sinusoidal_pe=CONFIG['use_sinusoidal_pe'],
    ).to(CONFIG['device'])

    print("\n===== KAN-Transformer 参数量统计 =====")
    count_parameters(model)
    print("======================================\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=1e-5
    )

    loss_history = []
    model.train()
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.unsqueeze(-1).to(CONFIG['device'])
            targets = targets.to(CONFIG['device'])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'KAN',
        'rx_mean': rx_mean,
        'rx_std': rx_std,
        'config': CONFIG,
    }, 'kan_model.pth')
    print("KAN-Transformer 模型已保存至 kan_model.pth")

    plt.plot(loss_history)
    plt.title('KAN-Transformer Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()


if __name__ == '__main__':
    train()
