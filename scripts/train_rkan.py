"""
RKAN (Recurrent Kolmogorov-Arnold Network) 均衡器 — 训练脚本

核心思想:
  KAN 将可学习的 B-样条激活函数放在"边"（权重）上，节点仅做求和。
  RKAN 在此基础上引入递归结构，使隐藏状态通过样条函数反馈：
      h_{t,j} = tanh( Σ_i φ^{in}_{j,i}(x_{t,i})
                     + Σ_k φ^{rec}_{j,k}(h_{t-1,k}) )
  其中 φ(x) = w_b · SiLU(x)  +  Σ_l c_l · B_l^k(x)
       B_l^k 为 k 阶 B-样条基函数，c_l 为可学习样条系数。

  整体流程：
    1. 将接收窗 (window_size 个采样) 逐步送入 RKANCell；
    2. 取末时刻隐状态 h_T 经 KANLinear 输出层映射为 PAM4 符号估计。
    3. 通过 BPTT 训练所有样条系数和基权重。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from pathlib import Path
from datetime import datetime

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / 'models'
IMAGES_DIR = ROOT / 'images'
LOGS_DIR   = ROOT / 'logs'
MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def get_device():
    """自动识别：NVIDIA CUDA > Apple MPS > CPU"""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


CONFIG = {
    'window_size': 21,
    'sps': 2,
    'hidden_size': 16,
    'grid_size': 5,           # B-样条网格区间数
    'spline_order': 3,        # 样条阶数 (3 = 三次样条)
    'grid_range': (-2.0, 2.0),
    'grad_clip': 1.0,         # 梯度裁剪 (BPTT 稳定性)
    'batch_size': 256,
    'epochs': 50,
    'lr': 0.001,
    'label_scale': 3.0,
    'eval_interval': 1,
    'device': get_device()
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
        self.rx_mean = rx_mean if rx_mean is not None else np.mean(rx_signal)
        self.rx_std  = rx_std  if rx_std  is not None else np.std(rx_signal)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start_sample = idx * self.sps
        end_sample   = start_sample + self.w
        x_seq = self.rx[start_sample:end_sample]
        x_seq = (x_seq - self.rx_mean) / (self.rx_std + 1e-8)
        label_idx = idx + (self.w // self.sps) // 2
        y = self.labels[label_idx] / self.label_scale
        return torch.FloatTensor(x_seq.real), torch.FloatTensor([y])


# ================= KAN 线性层 =================
class KANLinear(nn.Module):
    """
    KAN (Kolmogorov-Arnold Network) 线性层。

    每条边 (i→j) 上放置一个可学习单变量函数：
        φ_{j,i}(x_i) = w_b_{j,i} · SiLU(x_i) + Σ_l c_{j,i,l} · B_l^k(x_i)

    参数:
        in_features   – 输入维度
        out_features  – 输出维度
        grid_size     – B-样条均匀网格区间数 G
        spline_order  – B-样条阶数 k (3 = 三次)
        grid_range    – 网格覆盖范围 [a, b]

    内部张量:
        grid          – (in, G+2k+1) 含两端扩展的节点向量
        spline_weight – (out, in, G+k) 样条系数
        base_weight   – (out, in) SiLU 基权重
    """

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 grid_range=(-2.0, 2.0)):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self.num_bases     = grid_size + spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - h * spline_order,
            grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
        )
        # 每个输入特征共享同一组节点位置 → (in_features, n_knots)
        self.register_buffer('grid', grid.unsqueeze(0).expand(in_features, -1).contiguous())

        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, self.num_bases) * 0.1
        )
        self.base_weight = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(out_features, in_features), a=5**0.5)
        )

    def _bspline_bases(self, x):
        """
        向量化 de Boor 递推，计算 k 阶 B-样条基。

        x:       (..., in_features)
        Returns: (..., in_features, num_bases)

        递推公式 (p = 1 … k):
          B_{i,p+1}(x) = [(x-t_i)/(t_{i+p}-t_i)] B_{i,p}(x)
                       + [(t_{i+p+1}-x)/(t_{i+p+1}-t_{i+1})] B_{i+1,p}(x)
        其中 t 为节点向量 (self.grid)。

        实现上用张量切片一次性计算所有基函数，避免 Python 循环开销。
        """
        x_expand = x.unsqueeze(-1)  # (..., in, 1)

        # 0 阶基: 指示函数
        bases = ((x_expand >= self.grid[..., :-1]) &
                 (x_expand <  self.grid[..., 1:])).to(x.dtype)
        # bases: (..., in, G+2k)

        for p in range(1, self.spline_order + 1):
            # 左递推
            left_num  = x_expand - self.grid[..., :-(p + 1)]
            left_den  = (self.grid[..., p:-1] - self.grid[..., :-(p + 1)]).clamp(min=1e-8)
            # 右递推
            right_num = self.grid[..., (p + 1):] - x_expand
            right_den = (self.grid[..., (p + 1):] - self.grid[..., 1:(-p)]).clamp(min=1e-8)

            bases = (left_num / left_den) * bases[..., :-1] \
                  + (right_num / right_den) * bases[..., 1:]

        return bases  # (..., in, G+k)

    def forward(self, x):
        """
        x: (..., in_features) → (..., out_features)
        """
        # SiLU 基分量
        base_out = F.linear(F.silu(x), self.base_weight)

        # B-样条分量
        bases = self._bspline_bases(x)  # (..., in, num_bases)
        batch_shape = x.shape[:-1]
        flat_bases  = bases.reshape(-1, self.in_features * self.num_bases)
        flat_weight = self.spline_weight.reshape(self.out_features, -1)
        spline_out  = F.linear(flat_bases, flat_weight).reshape(*batch_shape, self.out_features)

        return base_out + spline_out


# ================= RKAN 递归单元 =================
class RKANCell(nn.Module):
    """
    RKAN 递归单元 (Recurrent KAN Cell)。

    隐藏状态更新 (带 tanh 有界化以保证 BPTT 稳定):
        h_t = tanh( KAN_in(x_t) + KAN_rec(h_{t-1}) )

    KAN_in  : 输入→隐藏 的 KAN 映射 (φ^{in})
    KAN_rec : 隐藏→隐藏 的 KAN 映射 (φ^{rec})

    样条参数和基权重通过 BPTT 联合训练。
    """

    def __init__(self, input_size, hidden_size,
                 grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)):
        super().__init__()
        self.hidden_size = hidden_size

        self.kan_in = KANLinear(
            input_size, hidden_size,
            grid_size=grid_size, spline_order=spline_order,
            grid_range=grid_range,
        )
        # 递归映射 φ^{rec}: 输入为 tanh 输出，有界于 (-1, 1)
        self.kan_rec = KANLinear(
            hidden_size, hidden_size,
            grid_size=grid_size, spline_order=spline_order,
            grid_range=(-1.5, 1.5),
        )

    def forward(self, x_t, h_prev):
        """
        x_t:    (batch, input_size)
        h_prev: (batch, hidden_size)
        Returns: h_t (batch, hidden_size)
        """
        return torch.tanh(self.kan_in(x_t) + self.kan_rec(h_prev))


# ================= RKAN 均衡器 =================
class RKANEqualizer(nn.Module):
    """
    递归 KAN (RKAN) 均衡器。

    结构:
      1) 将长度为 T 的接收窗 (window_size) 内每个采样逐步送入 RKANCell，
         隐藏状态在时间步间通过 B-样条参数化的 KAN 映射递归更新；
      2) 取末时刻 h_T 经 KAN 输出层映射到 PAM4 符号估计值。

    参数量约 hidden² × (G+k+1) + 2×hidden × (G+k+1)，
    与相同 hidden_size 的 BiLSTM 量级相当。
    """

    def __init__(self, input_size=1, hidden_size=16,
                 grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)):
        super().__init__()
        self.hidden_size = hidden_size

        self.rkan_cell = RKANCell(
            input_size, hidden_size,
            grid_size=grid_size, spline_order=spline_order,
            grid_range=grid_range,
        )
        # 输出映射 (tanh 的输出在 (-1,1))
        self.output_kan = KANLinear(
            hidden_size, 1,
            grid_size=grid_size, spline_order=spline_order,
            grid_range=(-1.5, 1.5),
        )

    def forward(self, src):
        """
        src: (batch, seq_len, 1)
        Returns: (batch, 1)
        """
        batch_size, seq_len, _ = src.shape
        h = torch.zeros(batch_size, self.hidden_size, device=src.device)

        for t in range(seq_len):
            x_t = src[:, t, :]            # (batch, 1)
            h = self.rkan_cell(x_t, h)    # BPTT: 梯度沿时间步回传

        return self.output_kan(h)


# ================= 构建函数 (供对比脚本调用) =================
def build_rkan(config, device):
    return RKANEqualizer(
        input_size    = 1,
        hidden_size   = config['hidden_size'],
        grid_size     = config['grid_size'],
        spline_order  = config['spline_order'],
        grid_range    = config.get('grid_range', (-2.0, 2.0)),
    ).to(device)


# ================= 参数量统计工具 =================
def count_parameters(model, log):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  总参数量:     {total:,}")
    log.info(f"  可训练参数量: {trainable:,}")
    log.info("  各层参数明细:")
    for name, p in model.named_parameters():
        log.info(f"    {name}: {p.numel():,}  {list(p.shape)}")
    return total


# ================= 验证集评估（MSE + BER） =================
def evaluate(model, loader, device, criterion, label_scale):
    model.eval()
    total_loss = 0.0
    preds_list, targets_list = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.unsqueeze(-1).to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    model.train()
    avg_loss = total_loss / len(loader)

    preds   = np.concatenate(preds_list).flatten()   * label_scale
    targets = np.concatenate(targets_list).flatten() * label_scale

    thresholds = [-2, 0, 2]
    pred_labels = np.select(
        [preds < thresholds[0], preds < thresholds[1], preds < thresholds[2]],
        [-3, -1, 1], default=3
    )
    true_labels = np.select(
        [targets < thresholds[0], targets < thresholds[1], targets < thresholds[2]],
        [-3, -1, 1], default=3
    )
    ber = float(np.mean(pred_labels != true_labels))
    return avg_loss, ber


# ================= 训练流程 =================
def train():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = LOGS_DIR / f'rkan_train_{timestamp}.log'
    log = logging.getLogger('rkan_trainer')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(logging.FileHandler(log_path, mode='w', encoding='utf-8'))
    log.addHandler(logging.StreamHandler())
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for h in log.handlers:
        h.setFormatter(formatter)

    log.info("=" * 60)
    log.info("  RKAN Equalizer — 训练开始")
    log.info("=" * 60)
    dev = CONFIG['device']
    log.info(f"运行设备: {dev} (NVIDIA CUDA)" if dev == 'cuda' else
             f"运行设备: {dev} (Apple MPS)" if dev == 'mps' else f"运行设备: {dev} (CPU)")
    log.info(f"配置参数: {CONFIG}")

    # ---------- 数据加载 ----------
    data = scipy.io.loadmat(str(ROOT / 'dataset_for_python.mat'))

    rx_train = data['rx_train_export'].flatten()
    if np.iscomplexobj(rx_train):
        log.info("检测到复数信号，取绝对值作为 IM/DD 包络")
        rx_train = np.abs(rx_train)
    symb_train = data['symb_train_export'].flatten()

    rx_test = data['rx_test_export'].flatten()
    if np.iscomplexobj(rx_test):
        rx_test = np.abs(rx_test)
    symb_test = data['symb_test_export'].flatten()

    rx_mean = float(np.mean(rx_train))
    rx_std  = float(np.std(rx_train))
    log.info(f"归一化统计 (训练集): mean={rx_mean:.4f}, std={rx_std:.4f}")
    log.info(f"训练信号点数: {len(rx_train):,}  |  训练符号数: {len(symb_train):,}")
    log.info(f"测试信号点数: {len(rx_test):,}   |  测试符号数: {len(symb_test):,}")

    train_dataset = OpticalDataset(
        rx_train, symb_train, CONFIG['window_size'], CONFIG['sps'],
        rx_mean=rx_mean, rx_std=rx_std, label_scale=CONFIG['label_scale']
    )
    test_dataset = OpticalDataset(
        rx_test, symb_test, CONFIG['window_size'], CONFIG['sps'],
        rx_mean=rx_mean, rx_std=rx_std, label_scale=CONFIG['label_scale']
    )
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=0)

    # ---------- 模型初始化 ----------
    model = build_rkan(CONFIG, CONFIG['device'])

    log.info("\n===== RKAN 参数量统计 =====")
    count_parameters(model, log)
    log.info("============================\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=1e-5
    )

    # ---------- 训练循环 ----------
    train_loss_history = []
    val_loss_history   = []
    val_ber_history    = []

    best_val_loss = float('inf')
    best_val_ber  = float('inf')
    best_epoch    = 0

    model.train()
    for epoch in range(1, CONFIG['epochs'] + 1):
        epoch_start = time.perf_counter()
        epoch_loss  = 0.0

        for inputs, targets in train_loader:
            inputs  = inputs.unsqueeze(-1).to(CONFIG['device'])
            targets = targets.to(CONFIG['device'])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            # BPTT 梯度裁剪：防止递归展开时梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_time     = time.perf_counter() - epoch_start
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % CONFIG['eval_interval'] == 0:
            val_loss, val_ber = evaluate(
                model, test_loader, CONFIG['device'], criterion, CONFIG['label_scale']
            )
            val_loss_history.append(val_loss)
            val_ber_history.append(val_ber)

            log.info(
                f"Epoch [{epoch:3d}/{CONFIG['epochs']}] "
                f"Time: {epoch_time:5.1f}s | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val BER: {val_ber:.4e} | "
                f"LR: {current_lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ber  = val_ber
                best_epoch    = epoch
                save_path = MODELS_DIR / 'rkan_model.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_type': 'RKAN',
                    'rx_mean': rx_mean,
                    'rx_std':  rx_std,
                    'config':  CONFIG,
                    'epoch':   epoch,
                    'best_val_loss': best_val_loss,
                    'best_val_ber':  best_val_ber,
                }, str(save_path))
                log.info(f"  ★ 最优模型已更新 → Epoch {epoch}, Val Loss={val_loss:.6f}, Val BER={val_ber:.4e}")
        else:
            log.info(
                f"Epoch [{epoch:3d}/{CONFIG['epochs']}] "
                f"Time: {epoch_time:5.1f}s | "
                f"Train Loss: {avg_train_loss:.6f} | LR: {current_lr:.2e}"
            )

    log.info("\n" + "=" * 60)
    log.info(f"训练完成！最优 Epoch: {best_epoch}")
    log.info(f"最优验证损失: {best_val_loss:.6f}")
    log.info(f"最优验证 BER:  {best_val_ber:.4e}")
    log.info(f"模型已保存至:  {MODELS_DIR / 'rkan_model.pth'}")
    log.info("=" * 60)

    # ---------- 绘制训练/验证双曲线 ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_loss_history, label='Train Loss')
    eval_epochs = list(range(CONFIG['eval_interval'], CONFIG['epochs'] + 1, CONFIG['eval_interval']))
    axes[0].plot(eval_epochs, val_loss_history, label='Val Loss')
    axes[0].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    axes[0].set_title('MSE Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].semilogy(eval_epochs, val_ber_history, 'o-', color='C2', label='Val BER')
    axes[1].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    axes[1].set_title('Validation BER (PAM4 硬判决)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('BER')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.4)

    plt.suptitle(f'RKAN — Training Summary (Best BER={best_val_ber:.4e})', fontsize=12)
    plt.tight_layout()
    fig_path = IMAGES_DIR / 'rkan_training_loss.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"训练曲线已保存: {fig_path}")


if __name__ == '__main__':
    train()
