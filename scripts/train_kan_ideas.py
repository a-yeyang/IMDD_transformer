"""
KAN 均衡器变体 — 训练脚本

基于 RKAN 实验的结论：递归结构在固定窗口 IM/DD 均衡中无优势。
本脚本探索三种将 KAN 与前馈结构结合的方案：

  方案 1: KAN-FCNN   — 纯 KAN 前馈网络 (KANLinear 替代 Linear+ReLU)
  方案 2: Hybrid-KAN — FCNN 前端线性特征提取 + KAN 后端非线性映射
  方案 3: ResKAN     — FCNN 主路径 + KAN 残差校正分支

核心假设：
  IM/DD 信道的非线性主要来自方检波 (|·|²)。KAN 的可学习样条激活函数
  擅长拟合单变量非线性函数，若放置在正确的位置（前馈结构而非递归结构），
  可能比 ReLU 更精确地学习方检波的逆映射。

运行方式:
  python train_kan_ideas.py             # 顺序训练全部 3 个模型
  python train_kan_ideas.py kan_fcnn    # 仅训练 KAN-FCNN
  python train_kan_ideas.py hybrid_kan  # 仅训练 Hybrid-KAN
  python train_kan_ideas.py res_kan     # 仅训练 ResKAN
"""

import sys
import torch
import torch.nn as nn
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

from train_rkan import KANLinear


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


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


# ============================================================
#  方案 1:  KAN-FCNN — 纯 KAN 前馈网络
# ============================================================
class KANFCNNEqualizer(nn.Module):
    """
    用 KANLinear (B-样条边激活) 完全替代 FCNN 的 Linear+ReLU。
    KAN 自带非线性 (SiLU 基 + 样条)，无需额外激活函数。
    层间加 LayerNorm 使中间激活落在样条网格范围内。

    与 FCNN([64,32]) 的 3,521 参数相比，
    本模型用 [16,8] + grid=5 + order=3 约 4,344 参数，量级相当。
    """

    def __init__(self, input_dim, hidden_dims=None,
                 grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 8]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(KANLinear(in_dim, h, grid_size, spline_order, grid_range))
            layers.append(nn.LayerNorm(h))
            in_dim = h
        layers.append(KANLinear(in_dim, 1, grid_size, spline_order, grid_range))
        self.net = nn.Sequential(*layers)

    def forward(self, src):
        x = src.squeeze(-1)  # (batch, window_size)
        return self.net(x)


KAN_FCNN_CONFIG = {
    'window_size': 21,
    'sps': 2,
    'hidden_dims': [16, 8],
    'grid_size': 5,
    'spline_order': 3,
    'grid_range': (-2.0, 2.0),
    'batch_size': 256,
    'epochs': 30,
    'lr': 0.001,
    'label_scale': 3.0,
    'eval_interval': 1,
    'device': get_device(),
}


def build_kan_fcnn(config, device):
    return KANFCNNEqualizer(
        input_dim    = config['window_size'],
        hidden_dims  = config['hidden_dims'],
        grid_size    = config['grid_size'],
        spline_order = config['spline_order'],
        grid_range   = tuple(config.get('grid_range', (-2.0, 2.0))),
    ).to(device)


# ============================================================
#  方案 2:  Hybrid-KAN — FCNN 前端 + KAN 后端
# ============================================================
class HybridKANEqualizer(nn.Module):
    """
    混合架构：
      前端 — 标准 Linear+ReLU 做高效线性降维 (与 FCNN 相同)
      后端 — KANLinear 做非线性符号回归

    核心假设: FCNN 的 Linear+ReLU 擅长线性特征提取 (撤色散)，
    但在最终的非线性映射步骤 (逆方检波) 上可能不够精确。
    用 KAN 替换最后一层可以更好地逼近该非线性函数。

    参数量 ≈ FCNN + 一个 KAN 层的额外样条参数。
    """

    def __init__(self, input_dim, linear_dims=None,
                 grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)):
        super().__init__()
        if linear_dims is None:
            linear_dims = [64, 32]

        layers = []
        in_dim = input_dim
        for h in linear_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.linear_front = nn.Sequential(*layers)

        self.kan_out = KANLinear(
            in_dim, 1, grid_size, spline_order, grid_range
        )

    def forward(self, src):
        x = src.squeeze(-1)
        x = self.linear_front(x)
        return self.kan_out(x)


HYBRID_KAN_CONFIG = {
    'window_size': 21,
    'sps': 2,
    'linear_dims': [64, 32],
    'grid_size': 5,
    'spline_order': 3,
    'grid_range': (-2.0, 2.0),
    'batch_size': 256,
    'epochs': 20,
    'lr': 0.001,
    'label_scale': 3.0,
    'eval_interval': 1,
    'device': get_device(),
}


def build_hybrid_kan(config, device):
    return HybridKANEqualizer(
        input_dim    = config['window_size'],
        linear_dims  = config['linear_dims'],
        grid_size    = config['grid_size'],
        spline_order = config['spline_order'],
        grid_range   = tuple(config.get('grid_range', (-2.0, 2.0))),
    ).to(device)


# ============================================================
#  方案 3:  ResKAN — FCNN + KAN 残差校正
# ============================================================
class ResKANEqualizer(nn.Module):
    """
    残差增强架构：
      主路径 — 标准 FCNN (保证基线性能)
      残差路径 — 轻量 KAN 分支 (学习 FCNN 遗漏的非线性校正)
      输出 = FCNN(x) + α · KAN(x)，α 为可学习标量

    核心假设: FCNN 已经能做到不错的均衡 (BER~1e-5)，
    但可能存在系统性的残余非线性误差。KAN 分支专门学习这些
    微小的非线性校正项，而 α 控制校正强度。

    优势: 即使 KAN 分支学不到有用信息 (α→0)，
    性能也不会比 FCNN 差。
    """

    def __init__(self, input_dim, fcnn_dims=None, kan_hidden=8,
                 grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)):
        super().__init__()
        if fcnn_dims is None:
            fcnn_dims = [64, 32]

        # FCNN 主路径
        fcnn_layers = []
        in_dim = input_dim
        for h in fcnn_dims:
            fcnn_layers.append(nn.Linear(in_dim, h))
            fcnn_layers.append(nn.ReLU())
            in_dim = h
        fcnn_layers.append(nn.Linear(in_dim, 1))
        self.fcnn = nn.Sequential(*fcnn_layers)

        # KAN 残差路径
        self.kan_branch = nn.Sequential(
            KANLinear(input_dim, kan_hidden, grid_size, spline_order, grid_range),
            KANLinear(kan_hidden, 1, grid_size, spline_order, grid_range),
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, src):
        x = src.squeeze(-1)
        main_out = self.fcnn(x)
        kan_res  = self.kan_branch(x)
        return main_out + self.alpha * kan_res


RES_KAN_CONFIG = {
    'window_size': 21,
    'sps': 2,
    'fcnn_dims': [64, 32],
    'kan_hidden': 8,
    'grid_size': 5,
    'spline_order': 3,
    'grid_range': (-2.0, 2.0),
    'batch_size': 256,
    'epochs': 20,
    'lr': 0.001,
    'label_scale': 3.0,
    'eval_interval': 1,
    'device': get_device(),
}


def build_res_kan(config, device):
    return ResKANEqualizer(
        input_dim    = config['window_size'],
        fcnn_dims    = config['fcnn_dims'],
        kan_hidden   = config['kan_hidden'],
        grid_size    = config['grid_size'],
        spline_order = config['spline_order'],
        grid_range   = tuple(config.get('grid_range', (-2.0, 2.0))),
    ).to(device)


# ================= 模型注册表 =================
MODEL_TABLE = {
    'kan_fcnn': {
        'display': 'KAN-FCNN',
        'build_fn': build_kan_fcnn,
        'config': KAN_FCNN_CONFIG,
        'ckpt': 'kan_fcnn_model.pth',
    },
    'hybrid_kan': {
        'display': 'Hybrid-KAN',
        'build_fn': build_hybrid_kan,
        'config': HYBRID_KAN_CONFIG,
        'ckpt': 'hybrid_kan_model.pth',
    },
    'res_kan': {
        'display': 'ResKAN',
        'build_fn': build_res_kan,
        'config': RES_KAN_CONFIG,
        'ckpt': 'res_kan_model.pth',
    },
}


# ================= 参数量统计 =================
def count_parameters(model, log):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  总参数量:     {total:,}")
    log.info(f"  可训练参数量: {trainable:,}")
    log.info("  各层参数明细:")
    for name, p in model.named_parameters():
        log.info(f"    {name}: {p.numel():,}  {list(p.shape)}")
    return total


# ================= 评估 =================
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


# ================= 数据加载 (共享) =================
def load_data():
    data = scipy.io.loadmat(str(ROOT / 'dataset_for_python.mat'))

    rx_train = data['rx_train_export'].flatten()
    if np.iscomplexobj(rx_train):
        rx_train = np.abs(rx_train)
    symb_train = data['symb_train_export'].flatten()

    rx_test = data['rx_test_export'].flatten()
    if np.iscomplexobj(rx_test):
        rx_test = np.abs(rx_test)
    symb_test = data['symb_test_export'].flatten()

    rx_mean = float(np.mean(rx_train))
    rx_std  = float(np.std(rx_train))

    return {
        'rx_train': rx_train, 'symb_train': symb_train,
        'rx_test': rx_test,   'symb_test': symb_test,
        'rx_mean': rx_mean,   'rx_std': rx_std,
    }


# ================= 单模型训练 =================
def train_single(key, entry, data_dict):
    """训练单个模型变体。"""
    display_name = entry['display']
    build_fn     = entry['build_fn']
    config       = entry['config']
    ckpt_name    = entry['ckpt']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = LOGS_DIR / f'{key}_train_{timestamp}.log'
    log = logging.getLogger(f'{key}_trainer')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(logging.FileHandler(log_path, mode='w', encoding='utf-8'))
    log.addHandler(logging.StreamHandler())
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for h in log.handlers:
        h.setFormatter(formatter)

    log.info("=" * 60)
    log.info(f"  {display_name} Equalizer — 训练开始")
    log.info("=" * 60)
    dev = config['device']
    log.info(f"运行设备: {dev}")
    log.info(f"配置参数: {config}")

    rx_mean, rx_std = data_dict['rx_mean'], data_dict['rx_std']
    log.info(f"归一化统计: mean={rx_mean:.4f}, std={rx_std:.4f}")
    log.info(f"训练信号点数: {len(data_dict['rx_train']):,}  |  "
             f"训练符号数: {len(data_dict['symb_train']):,}")
    log.info(f"测试信号点数: {len(data_dict['rx_test']):,}   |  "
             f"测试符号数: {len(data_dict['symb_test']):,}")

    train_dataset = OpticalDataset(
        data_dict['rx_train'], data_dict['symb_train'],
        config['window_size'], config['sps'],
        rx_mean=rx_mean, rx_std=rx_std, label_scale=config['label_scale']
    )
    test_dataset = OpticalDataset(
        data_dict['rx_test'], data_dict['symb_test'],
        config['window_size'], config['sps'],
        rx_mean=rx_mean, rx_std=rx_std, label_scale=config['label_scale']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'],
                              shuffle=False, num_workers=0)

    model = build_fn(config, dev)
    log.info(f"\n===== {display_name} 参数量统计 =====")
    count_parameters(model, log)
    log.info("=" * 40 + "\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-5
    )

    train_loss_history = []
    val_loss_history   = []
    val_ber_history    = []
    best_val_loss = float('inf')
    best_val_ber  = float('inf')
    best_epoch    = 0

    model.train()
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.perf_counter()
        epoch_loss  = 0.0

        for inputs, targets in train_loader:
            inputs  = inputs.unsqueeze(-1).to(dev)
            targets = targets.to(dev)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_time     = time.perf_counter() - epoch_start
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % config['eval_interval'] == 0:
            val_loss, val_ber = evaluate(
                model, test_loader, dev, criterion, config['label_scale']
            )
            val_loss_history.append(val_loss)
            val_ber_history.append(val_ber)

            log.info(
                f"Epoch [{epoch:3d}/{config['epochs']}] "
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
                save_path = MODELS_DIR / ckpt_name
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_type': display_name,
                    'rx_mean': rx_mean,
                    'rx_std':  rx_std,
                    'config':  config,
                    'epoch':   epoch,
                    'best_val_loss': best_val_loss,
                    'best_val_ber':  best_val_ber,
                }, str(save_path))
                log.info(f"  ★ 最优模型已更新 → Epoch {epoch}, "
                         f"Val Loss={val_loss:.6f}, Val BER={val_ber:.4e}")
        else:
            log.info(
                f"Epoch [{epoch:3d}/{config['epochs']}] "
                f"Time: {epoch_time:5.1f}s | "
                f"Train Loss: {avg_train_loss:.6f} | LR: {current_lr:.2e}"
            )

    log.info("\n" + "=" * 60)
    log.info(f"训练完成！最优 Epoch: {best_epoch}")
    log.info(f"最优验证损失: {best_val_loss:.6f}")
    log.info(f"最优验证 BER:  {best_val_ber:.4e}")
    log.info(f"模型已保存至:  {MODELS_DIR / ckpt_name}")
    log.info("=" * 60)

    # ---------- 绘制训练曲线 ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_loss_history, label='Train Loss')
    eval_epochs = list(range(config['eval_interval'],
                             config['epochs'] + 1, config['eval_interval']))
    axes[0].plot(eval_epochs, val_loss_history, label='Val Loss')
    axes[0].axvline(x=best_epoch, color='r', linestyle='--',
                    label=f'Best Epoch ({best_epoch})')
    axes[0].set_title('MSE Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].semilogy(eval_epochs, val_ber_history, 'o-', color='C2', label='Val BER')
    axes[1].axvline(x=best_epoch, color='r', linestyle='--',
                    label=f'Best Epoch ({best_epoch})')
    axes[1].set_title('Validation BER (PAM4 硬判决)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('BER')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.4)

    plt.suptitle(f'{display_name} — Training Summary (Best BER={best_val_ber:.4e})',
                 fontsize=12)
    plt.tight_layout()
    fig_path = IMAGES_DIR / f'{key}_training_loss.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"训练曲线已保存: {fig_path}")

    return best_val_ber


# ================= 入口 =================
def train_all(keys=None):
    """
    训练指定 (或全部) KAN 变体。
    keys: 要训练的模型键名列表，None 表示全部。
    """
    if keys is None:
        keys = list(MODEL_TABLE.keys())

    data_dict = load_data()
    print(f"\n数据已加载，将训练以下模型: {[MODEL_TABLE[k]['display'] for k in keys]}\n")

    results = {}
    for k in keys:
        entry = MODEL_TABLE[k]
        best_ber = train_single(k, entry, data_dict)
        results[entry['display']] = best_ber
        print()

    print("\n" + "=" * 50)
    print("         全部训练完成 — 最优 BER 汇总")
    print("=" * 50)
    for name, ber in results.items():
        print(f"  {name:15s}  BER = {ber:.4e}")
    print("=" * 50)


if __name__ == '__main__':
    requested = sys.argv[1:] if len(sys.argv) > 1 else None
    if requested:
        valid = [k for k in requested if k in MODEL_TABLE]
        if not valid:
            print(f"可用模型: {list(MODEL_TABLE.keys())}")
            sys.exit(1)
        train_all(valid)
    else:
        train_all()
