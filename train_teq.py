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
    'window_size': 21,       # 输入滑窗大小 (采样点数)
    'sps': 2,                # 每个符号的采样点数 (需与Matlab一致)
    'd_model': 16,           # Transformer特征维度 (大幅降低: 64->16)
    'nhead': 2,              # 注意力头数 (4->2, 确保 d_model % nhead == 0)
    'dim_feedforward': 32,   # FFN 中间维度 (PyTorch默认2048, 这里显式设小)
    'num_layers': 1,         # Encoder层数 (2->1, 配合权重共享可等效多层)
    'num_passes': 2,         # 权重共享: 单层循环执行的次数 (等效深度)
    'use_weight_sharing': True,
    'use_center_token': True,   # True: 只取中心token; False: flatten全部token
    'use_sinusoidal_pe': True,  # True: 固定正弦编码(0参数); False: 可学习编码
    'batch_size': 256,
    'epochs': 150,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ================= 数据集定义 =================
class OpticalDataset(Dataset):
    def __init__(self, rx_signal, labels, window_size, sps,
                 rx_mean=None, rx_std=None, label_scale=3.0):
        self.rx = rx_signal
        self.labels = labels
        self.w = window_size
        self.sps = sps
        self.label_scale = label_scale
        self.n_samples = len(labels) - (window_size // sps) - 1

        # 全局归一化统计量 (用整个信号计算，而非逐窗口)
        self.rx_mean = rx_mean if rx_mean is not None else np.mean(rx_signal)
        self.rx_std = rx_std if rx_std is not None else np.std(rx_signal)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start_sample = idx * self.sps
        end_sample = start_sample + self.w
        x_seq = self.rx[start_sample:end_sample]

        # 全局归一化 (稳定，不受窗口长度影响)
        x_seq = (x_seq - self.rx_mean) / (self.rx_std + 1e-8)

        label_idx = idx + (self.w // self.sps) // 2
        y = self.labels[label_idx]

        # 标签归一化: {-3,-1,1,3} -> {-1,-1/3,1/3,1}
        y = y / self.label_scale

        return torch.FloatTensor(x_seq.real), torch.FloatTensor([y])

# ================= 固定正弦位置编码 (零参数) =================
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

# ================= 轻量化 Transformer 均衡器 =================
class LightweightTransformerEQ(nn.Module):
    """
    针对 IM/DD PAM4 均衡场景的轻量化 Transformer，核心改动：
    1) 显式设置极小的 dim_feedforward (默认2048→32)
    2) 只取中心 token 做预测，避免 flatten 全序列
    3) 支持层间权重共享 (单层多次循环)
    4) 固定正弦位置编码，省去可学习参数
    5) 极简输出头 (d_model→1)
    """
    def __init__(self, input_dim, d_model, nhead, num_layers,
                 dim_feedforward=32, num_passes=1, use_weight_sharing=False,
                 use_center_token=True, use_sinusoidal_pe=True):
        super().__init__()
        self.use_center_token = use_center_token
        self.use_weight_sharing = use_weight_sharing
        self.num_passes = num_passes if use_weight_sharing else 1
        self.center_idx = input_dim // 2

        self.input_proj = nn.Linear(1, d_model)

        if use_sinusoidal_pe:
            self.pos_encoder = SinusoidalPE(d_model, max_len=input_dim)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1
        )
        if use_weight_sharing:
            self.shared_layer = encoder_layer
        else:
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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

        if self.use_weight_sharing:
            for _ in range(self.num_passes):
                x = self.shared_layer(x)
        else:
            x = self.transformer_encoder(x)

        if self.use_center_token:
            x = x[:, self.center_idx, :]
        else:
            x = x.reshape(x.size(0), -1)

        return self.head(x)

# ================= 参数量统计工具 =================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total:,}")
    print(f"  可训练参数量: {trainable:,}")
    print("  各层参数明细:")
    for name, p in model.named_parameters():
        print(f"    {name}: {p.numel():,}  {list(p.shape)}")
    return total

# ================= 训练流程 =================
def train():
    print(f"Running on {CONFIG['device']}")

    data = scipy.io.loadmat('dataset_for_python.mat')
    rx_train = data['rx_train_export'].flatten()
    if np.iscomplexobj(rx_train):
        print("检测到复数信号，取实部用于PAM4 IM/DD处理")
        rx_train = np.abs(rx_train)

    symb_train = data['symb_train_export'].flatten()

    # 计算全局归一化参数 (训练集统计量，测试时也要复用)
    rx_mean = np.mean(rx_train)
    rx_std = np.std(rx_train)
    print(f"全局统计: rx_mean={rx_mean:.4f}, rx_std={rx_std:.4f}")

    train_dataset = OpticalDataset(rx_train, symb_train, CONFIG['window_size'], CONFIG['sps'],
                                   rx_mean=rx_mean, rx_std=rx_std)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

    model = LightweightTransformerEQ(
        input_dim=CONFIG['window_size'],
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dim_feedforward=CONFIG['dim_feedforward'],
        num_passes=CONFIG['num_passes'],
        use_weight_sharing=CONFIG['use_weight_sharing'],
        use_center_token=CONFIG['use_center_token'],
        use_sinusoidal_pe=CONFIG['use_sinusoidal_pe'],
    ).to(CONFIG['device'])

    print("\n===== 模型参数量统计 =====")
    count_parameters(model)
    print("==========================\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
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
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'TEQ',
        'rx_mean': rx_mean,
        'rx_std': rx_std,
        'config': CONFIG,
    }, 'teq_model.pth')
    print("模型及归一化参数已保存。")

    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()

if __name__ == '__main__':
    train()