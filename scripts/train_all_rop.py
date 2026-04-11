"""
ROP 泛化实验 — 训练脚本

在 ROP=0 dBm 数据（dataset_rop0_for_python.mat）上训练均衡器。

运行方式:
  python train_all_rop.py                              # 训练全部模型
  python train_all_rop.py --models fcnn vanilla_kan     # 仅训练指定模型
  python train_all_rop.py --list                        # 列出所有可用 key
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
from train_fcnn    import FCNNEqualizer,   OpticalDataset, CONFIG as FCNN_CFG
from train_dnn     import DNNEqualizer,    CONFIG as DNN_CFG
from train_bilstm  import BiLSTMEqualizer, CONFIG as BILSTM_CFG
from train_kan_ideas import (
    build_vanilla_kan,    VANILLA_KAN_CONFIG,
    build_kan_fcnn,       KAN_FCNN_CONFIG,
    build_hybrid_kan,     HYBRID_KAN_CONFIG,
    build_res_kan,        RES_KAN_CONFIG,
    build_conv_kan,       CONV_KAN_CONFIG,
    build_kan_attention,  KAN_ATTN_CONFIG,
    build_multiscale_kan, MULTISCALE_KAN_CONFIG,
    build_gated_kan,      GATED_KAN_CONFIG,
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


def _build_kan_fcnn(config):
    return build_kan_fcnn(config, DEVICE)


def _build_hybrid_kan(config):
    return build_hybrid_kan(config, DEVICE)


def _build_res_kan(config):
    return build_res_kan(config, DEVICE)


def _build_conv_kan(config):
    return build_conv_kan(config, DEVICE)


def _build_kan_attn(config):
    return build_kan_attention(config, DEVICE)


def _build_multiscale_kan(config):
    return build_multiscale_kan(config, DEVICE)


def _build_gated_kan(config):
    return build_gated_kan(config, DEVICE)


# ================= 模型注册表 =================
MODEL_REGISTRY = [
    {
        'key': 'fcnn',       'name': 'FCNN',
        'build': _build_fcnn,        'cfg': dict(FCNN_CFG),
        'ckpt': 'fcnn_rop_model.pth', 'epochs': 15,
    },
    {
        'key': 'dnn',        'name': 'DNN',
        'build': _build_dnn,         'cfg': dict(DNN_CFG),
        'ckpt': 'dnn_rop_model.pth',  'epochs': 15,
    },
    {
        'key': 'bilstm',     'name': 'BiLSTM',
        'build': _build_bilstm,      'cfg': dict(BILSTM_CFG),
        'ckpt': 'bilstm_rop_model.pth', 'epochs': 50,
    },
    {
        'key': 'vanilla_kan', 'name': 'VanillaKAN',
        'build': _build_vanilla_kan, 'cfg': dict(VANILLA_KAN_CONFIG),
        'ckpt': 'vanilla_kan_rop_model.pth', 'epochs': 25,
    },
    {
        'key': 'kan_fcnn',   'name': 'KAN-FCNN',
        'build': _build_kan_fcnn,    'cfg': dict(KAN_FCNN_CONFIG),
        'ckpt': 'kan_fcnn_rop_model.pth', 'epochs': 30,
    },
    {
        'key': 'hybrid_kan', 'name': 'Hybrid-KAN',
        'build': _build_hybrid_kan,  'cfg': dict(HYBRID_KAN_CONFIG),
        'ckpt': 'hybrid_kan_rop_model.pth', 'epochs': 20,
    },
    {
        'key': 'res_kan',    'name': 'ResKAN',
        'build': _build_res_kan,     'cfg': dict(RES_KAN_CONFIG),
        'ckpt': 'res_kan_rop_model.pth', 'epochs': 20,
    },
    {
        'key': 'conv_kan',   'name': 'ConvKAN',
        'build': _build_conv_kan,    'cfg': dict(CONV_KAN_CONFIG),
        'ckpt': 'conv_kan_rop_model.pth', 'epochs': 25,
    },
    {
        'key': 'kan_attn',   'name': 'KAN-Attention',
        'build': _build_kan_attn,    'cfg': dict(KAN_ATTN_CONFIG),
        'ckpt': 'kan_attn_rop_model.pth', 'epochs': 25,
    },
    {
        'key': 'multiscale_kan', 'name': 'MultiScale-KAN',
        'build': _build_multiscale_kan, 'cfg': dict(MULTISCALE_KAN_CONFIG),
        'ckpt': 'multiscale_kan_rop_model.pth', 'epochs': 25,
    },
    {
        'key': 'gated_kan',  'name': 'GatedKAN',
        'build': _build_gated_kan,   'cfg': dict(GATED_KAN_CONFIG),
        'ckpt': 'gated_kan_rop_model.pth', 'epochs': 25,
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
def train_one(entry, rx_train, rx_test, sym_train, sym_test, rx_mean, rx_std, log):
    cfg         = {**entry['cfg'], 'device': DEVICE}
    epochs      = entry['epochs']
    label_scale = cfg.get('label_scale', 3.0)

    model     = entry['build'](cfg)
    train_ds  = OpticalDataset(rx_train, sym_train, cfg['window_size'], cfg['sps'],
                               rx_mean=rx_mean, rx_std=rx_std, label_scale=label_scale)
    test_ds   = OpticalDataset(rx_test,  sym_test,  cfg['window_size'], cfg['sps'],
                               rx_mean=rx_mean, rx_std=rx_std, label_scale=label_scale)
    train_ld  = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=0)
    test_ld   = DataLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

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
    axes[0].axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    axes[0].set_title('MSE Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True)
    axes[1].semilogy(ber_hist, 'o-', color='C2', label='Val BER')
    axes[1].axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    axes[1].set_title('Val BER'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True, which='both')
    plt.suptitle(f"{entry['name']} (ROP=0) — Best BER={best_val_ber:.3e}")
    plt.tight_layout()
    plt.savefig(str(IMAGES_DIR / f"{entry['key']}_rop_training.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return best_val_ber


# ================= 入口 =================
def main(keys=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log = logging.getLogger('train_rop')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh  = logging.FileHandler(LOGS_DIR / f'train_rop_{timestamp}.log', encoding='utf-8')
    sh  = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    log.addHandler(fh);   log.addHandler(sh)

    log.info("=" * 65)
    log.info("  ROP 泛化实验 — 模型训练  (数据: ROP=0 dBm)")
    log.info(f"  设备: {DEVICE}  全部模型数: {len(MODEL_REGISTRY)}")
    log.info("=" * 65)

    rx_train, rx_test, sym_train, sym_test, rx_mean, rx_std = load_data()
    log.info(f"训练: {len(rx_train):,} pts  测试: {len(rx_test):,} pts")
    log.info(f"归一化: mean={rx_mean:.4f}, std={rx_std:.4f}\n")

    to_run = [_KEY_TO_ENTRY[k] for k in (keys or _KEY_TO_ENTRY.keys())
              if k in _KEY_TO_ENTRY]

    if keys:
        log.info(f"本次训练: {[e['name'] for e in to_run]}\n")

    for entry in to_run:
        log.info(f"▶ 训练 {entry['name']} (epochs={entry['epochs']}) ...")
        train_one(entry, rx_train, rx_test, sym_train, sym_test, rx_mean, rx_std, log)

    log.info("全部训练完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ROP 泛化实验 — 训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例:\n'
               '  python train_all_rop.py                           # 全部\n'
               '  python train_all_rop.py --models fcnn vanilla_kan # 指定\n'
               '  python train_all_rop.py --list                    # 列出 key',
    )
    parser.add_argument('--models', nargs='+', metavar='KEY',
                        help='仅训练指定 key 的模型 (可多个)')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用模型 key 后退出')
    # 兼容旧的位置参数用法
    parser.add_argument('legacy_keys', nargs='*', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.list:
        print("可用模型 key:")
        for e in MODEL_REGISTRY:
            print(f"  {e['key']:18s}  ({e['name']})")
        sys.exit(0)

    keys = args.models or args.legacy_keys or None
    if keys:
        valid = [k for k in keys if k in _KEY_TO_ENTRY]
        invalid = [k for k in keys if k not in _KEY_TO_ENTRY]
        if invalid:
            print(f"[警告] 未知 key 已忽略: {invalid}")
            print(f"       可用: {sorted(_KEY_TO_ENTRY.keys())}")
        if not valid:
            sys.exit(1)
        main(valid)
    else:
        main()
