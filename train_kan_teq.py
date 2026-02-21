import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math

# ================= 配置参数 =================
CONFIG = {
    'window_size': 21,
    'sps': 2,
    'd_model': 16,
    'nhead': 2,
    'dim_feedforward': 16,  # 因为KAN非线性强，中间维度可进一步从32降至16
    'num_layers': 1,
    'num_passes': 2,
    'use_weight_sharing': True,
    'use_center_token': True,
    'use_sinusoidal_pe': True,
    'kan_grid_size': 3,  # KAN 样条网格/多项式阶数 (创新点：控制非线性复杂度)
    'batch_size': 256,
    'epochs': 50,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


# ================= 数据集与PE (保持不变) =================
class OpticalDataset(Dataset):
    def __init__(self, rx_signal, labels, window_size, sps, rx_mean=None, rx_std=None, label_scale=3.0):
        self.rx = rx_signal
        self.labels = labels
        self.w = window_size
        self.sps = sps
        self.label_scale = label_scale
        self.n_samples = len(labels) - (window_size // sps) - 1
        self.rx_mean = rx_mean if rx_mean is not None else np.mean(rx_signal)
        self.rx_std = rx_std if rx_std is not None else np.std(rx_signal)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start_sample = idx * self.sps
        end_sample = start_sample + self.w
        x_seq = self.rx[start_sample:end_sample]
        x_seq = (x_seq - self.rx_mean) / (self.rx_std + 1e-8)
        label_idx = idx + (self.w // self.sps) // 2
        y = self.labels[label_idx] / self.label_scale
        return torch.FloatTensor(x_seq.real), torch.FloatTensor([y])


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ================= 创新模块：轻量化 KAN 线性层 =================
class FastKANLinear(nn.Module):
    """
    轻量化 KAN 核心：使用基础线性映射 + SiLU激活的可学习多项式基
    这比原版B-Spline实现更适合DSP部署和低延迟光通信
    """

    def __init__(self, in_features, out_features, grid_size=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        # 基础线性部分 (类似残差)
        self.base_linear = nn.Linear(in_features, out_features)
        self.base_activation = nn.SiLU()

        # 可学习的非线性基 (Spline/Polynomial 近似)
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size) / (in_features * grid_size))

    def forward(self, x):
        # x: (batch_size, seq_len, in_features)
        base_output = self.base_linear(self.base_activation(x))

        # 生成非线性基
        basis = []
        for i in range(1, self.grid_size + 1):
            basis.append(torch.sin(i * x))  # 使用傅里叶基近似非线性特征映射，对周期性光器件响应极佳
        basis = torch.stack(basis, dim=-1)  # (batch, seq, in, grid)

        # 爱因斯坦求和计算张量乘法，替代传统MLP的权重矩阵
        spline_output = torch.einsum('bsig,oig->bso', basis, self.spline_weight)

        return base_output + spline_output


# ================= 创新模块：KAN-Transformer 编码器层 =================
class KANTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1, grid_size=3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 创新点1：KAN-FFN 取代 Linear-ReLU-Linear
        self.kan1 = FastKANLinear(d_model, dim_feedforward, grid_size)
        self.kan2 = FastKANLinear(dim_feedforward, d_model, grid_size)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Attention Block
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # KAN-FFN Block (直接输出，无需额外ReLU，因为KAN已包含非线性基)
        src2 = self.kan2(self.kan1(src))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# ================= 模型本体：KAN-Former =================
class KANFormerEQ(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward,
                 num_passes=1, use_weight_sharing=False, use_center_token=True,
                 use_sinusoidal_pe=True, grid_size=3):
        super().__init__()
        self.use_center_token = use_center_token
        self.use_weight_sharing = use_weight_sharing
        self.num_passes = num_passes if use_weight_sharing else 1
        self.center_idx = input_dim // 2

        # 创新点2：Spline-based KAN Input Projection
        self.input_proj = FastKANLinear(1, d_model, grid_size)

        if use_sinusoidal_pe:
            self.pos_encoder = SinusoidalPE(d_model, max_len=input_dim)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)

        # 挂载 KAN-Encoder Layer
        kan_layer = KANTransformerEncoderLayer(d_model, nhead, dim_feedforward, grid_size=grid_size)

        if use_weight_sharing:
            self.shared_layer = kan_layer
        else:
            self.layers = nn.ModuleList(
                [KANTransformerEncoderLayer(d_model, nhead, dim_feedforward, grid_size=grid_size) for _ in
                 range(num_layers)])

        # 预测头
        self.head = nn.Linear(d_model, 1)

    def forward(self, src):
        x = self.input_proj(src)

        if isinstance(self.pos_encoder, SinusoidalPE):
            x = self.pos_encoder(x)
        else:
            x = x + self.pos_encoder[:, :x.size(1), :]

        if self.use_weight_sharing:
            for _ in range(self.num_passes):
                x = self.shared_layer(x)
        else:
            for layer in self.layers:
                x = layer(x)

        if self.use_center_token:
            x = x[:, self.center_idx, :]
        else:
            x = x.reshape(x.size(0), -1)

        return self.head(x)


# 参数量统计工具 (保持不变)
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total:,}")
    print(f"  可训练参数量: {trainable:,}")
    return total


def train():
    print(f"Running KAN-Former on {CONFIG['device']}")
    # ... 数据加载部分同原版一致 ...
    data = scipy.io.loadmat('dataset_for_python.mat')
    rx_train = data['rx_train_export'].flatten()
    if np.iscomplexobj(rx_train):
        rx_train = np.abs(rx_train)
    symb_train = data['symb_train_export'].flatten()

    rx_mean = np.mean(rx_train)
    rx_std = np.std(rx_train)

    train_dataset = OpticalDataset(rx_train, symb_train, CONFIG['window_size'], CONFIG['sps'], rx_mean=rx_mean,
                                   rx_std=rx_std)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

    # 实例化 KAN-Former
    model = KANFormerEQ(
        input_dim=CONFIG['window_size'],
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dim_feedforward=CONFIG['dim_feedforward'],
        num_passes=CONFIG['num_passes'],
        use_weight_sharing=CONFIG['use_weight_sharing'],
        use_center_token=CONFIG['use_center_token'],
        use_sinusoidal_pe=CONFIG['use_sinusoidal_pe'],
        grid_size=CONFIG['kan_grid_size']
    ).to(CONFIG['device'])

    print("\n===== KAN-Former 参数量统计 =====")
    count_parameters(model)
    print("==========================\n")

    criterion = nn.MSELoss()
    # KAN模型初期由于基函数剧烈变化，建议使用稍低的Weight Decay防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-5)

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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{CONFIG['epochs']}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'rx_mean': rx_mean,
        'rx_std': rx_std,
        'config': CONFIG,
    }, 'kan_teq_model.pth')
    print("KAN-Former 模型已保存。")


if __name__ == '__main__':
    train()