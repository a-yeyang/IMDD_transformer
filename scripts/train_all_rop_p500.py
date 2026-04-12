"""
ROP 泛化实验 — 500 参数量级消融训练脚本

在 ROP=0 dBm 数据上训练六种均衡器（参数量统一到 ~500 量级）：

  对照组:
    1. FCNN           [16, 8]           ~497 params
    2. DNN            [14, 8] + BN      ~481 params
    3. BiLSTM         hs=6, 1 layer     ~517 params

  实验组:
    4. VanillaKAN     h=3               ~594 params  — 纯 KAN 基准
    5. ResKAN         fcnn=[4,2]+kan=2   ~498 params  — FCNN + KAN 残差

  消融实验:
    6. ResFCNN        main=[4,2]+res=17  ~494 params  — FCNN + FCNN 残差
       （验证 ResKAN 中 KAN 分支相比等参数 FCNN 分支的优势）

运行方式:
  python train_all_rop_p500.py                          # 训练全部
  python train_all_rop_p500.py --models res_kan res_fcnn # 指定
  python train_all_rop_p500.py --list                   # 列出 key
"""

import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

# ---------- 复用已有模型定义 ----------
from train_fcnn   import FCNNEqualizer,   OpticalDataset
from train_dnn    import DNNEqualizer
from train_bilstm import BiLSTMEqualizer
from train_kan_ideas import (
    KANLinear,
    build_vanilla_kan,
    build_res_kan,
)

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / 'models'
IMAGES_DIR = ROOT / 'images'
LOGS_DIR   = ROOT / 'logs'
DATA_PATH  = ROOT / 'dataset_rop0_for_python.mat'
MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


DEVICE = get_device()


# ============================================================
#  消融模型: ResFCNN — FCNN + FCNN 残差 (对照 ResKAN)
# ============================================================
class ResFCNNEqualizer(nn.Module):
    """
    FCNN + FCNN 残差消融模型。

    与 ResKAN 结构完全对应，唯一区别是残差分支用 FCNN 替代 KAN：
      主路径 — FCNN (Linear + ReLU)
      残差路径 — FCNN (Linear + ReLU)
      输出 = main(x) + α · residual(x)

    用于回答审稿人的关键质疑：
      "ResKAN 的增益是否来自 KAN 的 B-样条逼近优势，
       还是仅仅来自双分支残差结构？"
    """

    def __init__(self, input_dim, main_dims=None, res_dims=None):
        super().__init__()
        if main_dims is None:
            main_dims = [4, 2]
        if res_dims is None:
            res_dims = [17]

        main_layers = []
        d = input_dim
        for h in main_dims:
            main_layers.append(nn.Linear(d, h))
            main_layers.append(nn.ReLU())
            d = h
        main_layers.append(nn.Linear(d, 1))
        self.main = nn.Sequential(*main_layers)

        res_layers = []
        d = input_dim
        for h in res_dims:
            res_layers.append(nn.Linear(d, h))
            res_layers.append(nn.ReLU())
            d = h
        res_layers.append(nn.Linear(d, 1))
        self.residual = nn.Sequential(*res_layers)

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, src):
        x = src.squeeze(-1)
        return self.main(x) + self.alpha * self.residual(x)


def build_res_fcnn(config, device):
    return ResFCNNEqualizer(
        input_dim = config['window_size'],
        main_dims = config['main_dims'],
        res_dims  = config['res_dims'],
    ).to(device)


# ============================================================
#  500 参数量级配置
# ============================================================
_SHARED = {
    'window_size': 21, 'sps': 2,
    'batch_size': 256, 'lr': 0.001, 'label_scale': 3.0,
}

FCNN_P500 = {
    **_SHARED,
    'hidden_dims': [16, 8],
}

DNN_P500 = {
    **_SHARED,
    'hidden_dims': [14, 8],
    'use_batchnorm': True,
    'dropout': 0.1,
}

BILSTM_P500 = {
    **_SHARED,
    'hidden_size': 6,
    'num_lstm_layers': 1,
    'use_center': True,
}

VANILLA_KAN_P500 = {
    **_SHARED,
    'vanilla_hidden': 3,
    'grid_size': 5, 'spline_order': 3, 'grid_range': (-2.0, 2.0),
}

RES_KAN_P500 = {
    **_SHARED,
    'fcnn_dims': [4, 2],
    'kan_hidden': 2,
    'grid_size': 5, 'spline_order': 3, 'grid_range': (-2.0, 2.0),
}

RES_FCNN_P500 = {
    **_SHARED,
    'main_dims': [4, 2],
    'res_dims': [17],
}


# ================= 模型构建函数 =================
def _build_fcnn(config):
    return FCNNEqualizer(
        input_dim   = config['window_size'],
        hidden_dims = config['hidden_dims'],
    ).to(DEVICE)


def _build_dnn(config):
    return DNNEqualizer(
        input_dim     = config['window_size'],
        hidden_dims   = config['hidden_dims'],
        use_batchnorm = config['use_batchnorm'],
        dropout       = config['dropout'],
    ).to(DEVICE)


def _build_bilstm(config):
    return BiLSTMEqualizer(
        input_size  = 1,
        hidden_size = config['hidden_size'],
        num_layers  = config['num_lstm_layers'],
        use_center  = config['use_center'],
        window_size = config['window_size'],
    ).to(DEVICE)


def _build_vanilla_kan(config):
    return build_vanilla_kan(config, DEVICE)


def _build_res_kan(config):
    return build_res_kan(config, DEVICE)


def _build_res_fcnn(config):
    return build_res_fcnn(config, DEVICE)


# ================= 模型注册表 =================
MODEL_REGISTRY = [
    # --- 对照组 ---
    {
        'key': 'fcnn',  'name': 'FCNN',  'group': 'baseline',
        'build': _build_fcnn, 'cfg': dict(FCNN_P500),
        'ckpt': 'fcnn_p500_rop_model.pth', 'epochs': 20,
    },
    {
        'key': 'dnn',   'name': 'DNN',   'group': 'baseline',
        'build': _build_dnn,  'cfg': dict(DNN_P500),
        'ckpt': 'dnn_p500_rop_model.pth',  'epochs': 20,
    },
    {
        'key': 'bilstm', 'name': 'BiLSTM', 'group': 'baseline',
        'build': _build_bilstm, 'cfg': dict(BILSTM_P500),
        'ckpt': 'bilstm_p500_rop_model.pth', 'epochs': 60,
    },
    # --- 实验组 ---
    {
        'key': 'vanilla_kan', 'name': 'VanillaKAN', 'group': 'kan',
        'build': _build_vanilla_kan, 'cfg': dict(VANILLA_KAN_P500),
        'ckpt': 'vanilla_kan_p500_rop_model.pth', 'epochs': 30,
    },
    {
        'key': 'res_kan', 'name': 'ResKAN', 'group': 'kan',
        'build': _build_res_kan, 'cfg': dict(RES_KAN_P500),
        'ckpt': 'res_kan_p500_rop_model.pth', 'epochs': 25,
    },
    # --- 消融实验 ---
    {
        'key': 'res_fcnn', 'name': 'ResFCNN', 'group': 'ablation',
        'build': _build_res_fcnn, 'cfg': dict(RES_FCNN_P500),
        'ckpt': 'res_fcnn_p500_rop_model.pth', 'epochs': 25,
    },
]

_KEY_TO_ENTRY = {e['key']: e for e in MODEL_REGISTRY}


# ================= 数据加载 =================
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {DATA_PATH}，请先运行 RX_rop.m 生成 ROP=0 数据。"
        )
    data      = scipy.io.loadmat(str(DATA_PATH))
    rx_train  = data['rx_train_export'].flatten().astype(np.float64)
    rx_test   = data['rx_test_export'].flatten().astype(np.float64)
    sym_train = data['symb_train_export'].flatten()
    sym_test  = data['symb_test_export'].flatten()
    if np.iscomplexobj(rx_train): rx_train = np.abs(rx_train)
    if np.iscomplexobj(rx_test):  rx_test  = np.abs(rx_test)
    rx_mean = float(np.mean(rx_train))
    rx_std  = float(np.std(rx_train))
    return rx_train, rx_test, sym_train, sym_test, rx_mean, rx_std


# ================= BER 计算 =================
def compute_ber(preds, targets, label_scale):
    p = preds   * label_scale
    t = targets * label_scale
    pred_lbl = np.select([p < -2, p < 0, p < 2], [-3, -1, 1], default=3)
    true_lbl = np.select([t < -2, t < 0, t < 2], [-3, -1, 1], default=3)
    return float(np.mean(pred_lbl != true_lbl))


# ================= 单模型训练 =================
def train_one(entry, rx_train, rx_test, sym_train, sym_test,
              rx_mean, rx_std, log):
    cfg         = {**entry['cfg'], 'device': DEVICE}
    epochs      = entry['epochs']
    label_scale = cfg.get('label_scale', 3.0)

    model = entry['build'](cfg)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  参数量: {n_params:,}")

    train_ds = OpticalDataset(rx_train, sym_train, cfg['window_size'], cfg['sps'],
                              rx_mean=rx_mean, rx_std=rx_std, label_scale=label_scale)
    test_ds  = OpticalDataset(rx_test,  sym_test,  cfg['window_size'], cfg['sps'],
                              rx_mean=rx_mean, rx_std=rx_std, label_scale=label_scale)
    train_ld = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_hist, val_hist, ber_hist = [], [], []
    best_val_loss = float('inf')
    best_val_ber  = float('inf')
    best_epoch    = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        ep_loss = 0.0
        for x, y in train_ld:
            x = x.unsqueeze(-1).to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        avg_train = ep_loss / len(train_ld)
        train_hist.append(avg_train)

        model.eval()
        preds_list, tgt_list = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, y in test_ld:
                x = x.unsqueeze(-1).to(DEVICE)
                y = y.to(DEVICE)
                out = model(x)
                val_loss += criterion(out, y).item()
                preds_list.append(out.cpu().numpy())
                tgt_list.append(y.cpu().numpy())
        val_loss /= len(test_ld)
        val_hist.append(val_loss)

        val_ber = compute_ber(
            np.concatenate(preds_list).flatten(),
            np.concatenate(tgt_list).flatten(),
            label_scale,
        )
        ber_hist.append(val_ber)

        log.info(
            f"  Epoch [{epoch:3d}/{epochs}] {time.perf_counter()-t0:.1f}s | "
            f"Train={avg_train:.5f} | Val={val_loss:.5f} | BER={val_ber:.3e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ber  = val_ber
            best_epoch    = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'rx_mean': rx_mean, 'rx_std': rx_std,
                'config':  cfg,     'epoch':  epoch,
                'best_val_ber': best_val_ber,
            }, str(MODELS_DIR / entry['ckpt']))

    log.info(f"  ★ Best Epoch={best_epoch}, Val BER={best_val_ber:.3e}\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_hist, label='Train Loss')
    axes[0].plot(val_hist,   label='Val Loss')
    axes[0].axvline(x=best_epoch-1, color='r', linestyle='--',
                    label=f'Best ({best_epoch})')
    axes[0].set_title('MSE Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].semilogy(ber_hist, 'o-', color='C2', label='Val BER')
    axes[1].axvline(x=best_epoch-1, color='r', linestyle='--',
                    label=f'Best ({best_epoch})')
    axes[1].set_title('Val BER')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, which='both')
    plt.suptitle(f"{entry['name']} p500 (ROP=0) — Best BER={best_val_ber:.3e}")
    plt.tight_layout()
    plt.savefig(str(IMAGES_DIR / f"{entry['key']}_p500_rop_training.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    return best_val_ber


# ================= 入口 =================
def main(keys=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log = logging.getLogger('train_rop_p500')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(LOGS_DIR / f'train_rop_p500_{timestamp}.log',
                             encoding='utf-8')
    sh = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    log.addHandler(fh);   log.addHandler(sh)

    log.info("=" * 65)
    log.info("  ROP 泛化实验 — 500 参数量级消融训练")
    log.info(f"  设备: {DEVICE}  全部模型数: {len(MODEL_REGISTRY)}")
    log.info("=" * 65)

    rx_train, rx_test, sym_train, sym_test, rx_mean, rx_std = load_data()
    log.info(f"训练: {len(rx_train):,} pts  测试: {len(rx_test):,} pts")
    log.info(f"归一化: mean={rx_mean:.4f}, std={rx_std:.4f}\n")

    to_run = [_KEY_TO_ENTRY[k] for k in (keys or _KEY_TO_ENTRY.keys())
              if k in _KEY_TO_ENTRY]

    # 打印参数量对比表
    log.info("-" * 55)
    log.info(f"  {'模型':12s} {'组别':10s} {'参数量':>8s}  {'Epochs':>6s}")
    log.info("-" * 55)
    for entry in to_run:
        tmp = entry['build']({**entry['cfg'], 'device': 'cpu'})
        p = sum(pp.numel() for pp in tmp.parameters())
        log.info(f"  {entry['name']:12s} {entry['group']:10s} {p:>8,}  {entry['epochs']:>6d}")
        del tmp
    log.info("-" * 55 + "\n")

    for entry in to_run:
        log.info(f"▶ 训练 {entry['name']} [{entry['group']}] "
                 f"(epochs={entry['epochs']}) ...")
        train_one(entry, rx_train, rx_test, sym_train, sym_test,
                  rx_mean, rx_std, log)

    log.info("全部训练完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ROP 泛化 — 500 参数量级消融训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例:\n'
               '  python train_all_rop_p500.py                          # 全部\n'
               '  python train_all_rop_p500.py --models res_kan res_fcnn # 指定\n'
               '  python train_all_rop_p500.py --list                   # 列出',
    )
    parser.add_argument('--models', nargs='+', metavar='KEY',
                        help='仅训练指定 key 的模型 (可多个)')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用模型 key 后退出')
    parser.add_argument('legacy_keys', nargs='*', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.list:
        print("可用模型 key:")
        for e in MODEL_REGISTRY:
            print(f"  {e['key']:14s} ({e['name']:12s})  [{e['group']}]")
        sys.exit(0)

    keys = args.models or args.legacy_keys or None
    if keys:
        valid = [k for k in keys if k in _KEY_TO_ENTRY]
        invalid = [k for k in keys if k not in _KEY_TO_ENTRY]
        if invalid:
            print(f"[警告] 未知 key: {invalid}")
            print(f"       可用: {sorted(_KEY_TO_ENTRY.keys())}")
        if not valid:
            sys.exit(1)
        main(valid)
    else:
        main()
