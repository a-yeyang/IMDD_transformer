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

# ================= 配置参数 =================
CONFIG = {
    'window_size': 21,
    'sps': 2,
    'hidden_dims': [64, 32, 16],  # 更深的隐藏层
    'use_batchnorm': True,
    'dropout': 0.1,
    'batch_size': 256,
    'epochs': 50,
    'lr': 0.001,
    'label_scale': 3.0,
    'eval_interval': 1,
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


# ================= DNN 均衡器 =================
class DNNEqualizer(nn.Module):
    """
    深度神经网络 (DNN) 均衡器。
    相比 FCNN，使用更深的层数并加入 BatchNorm + Dropout 正则化，
    以增强非线性建模能力和泛化性能。
    """

    def __init__(self, input_dim, hidden_dims=None, use_batchnorm=True, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, src):
        # src: (batch, window_size, 1)
        x = src.squeeze(-1)  # (batch, window_size)
        return self.net(x)


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
    log_path  = LOGS_DIR / f'dnn_train_{timestamp}.log'
    log = logging.getLogger('dnn_trainer')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(logging.FileHandler(log_path, mode='w', encoding='utf-8'))
    log.addHandler(logging.StreamHandler())
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for h in log.handlers:
        h.setFormatter(formatter)

    log.info("=" * 60)
    log.info("  DNN Equalizer — 训练开始")
    log.info("=" * 60)
    log.info(f"运行设备: {CONFIG['device']}")
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
    model = DNNEqualizer(
        input_dim     = CONFIG['window_size'],
        hidden_dims   = CONFIG['hidden_dims'],
        use_batchnorm = CONFIG['use_batchnorm'],
        dropout       = CONFIG['dropout'],
    ).to(CONFIG['device'])

    log.info("\n===== DNN 参数量统计 =====")
    count_parameters(model, log)
    log.info("==========================\n")

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
                save_path = MODELS_DIR / 'dnn_model.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_type': 'DNN',
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
    log.info(f"模型已保存至:  {MODELS_DIR / 'dnn_model.pth'}")
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

    plt.suptitle(f'DNN — Training Summary (Best BER={best_val_ber:.4e})', fontsize=12)
    plt.tight_layout()
    fig_path = IMAGES_DIR / 'dnn_training_loss.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"训练曲线已保存: {fig_path}")


if __name__ == '__main__':
    train()
