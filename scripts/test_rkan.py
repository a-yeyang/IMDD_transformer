"""
RKAN 均衡器 — 独立测试脚本
加载训练好的 rkan_model.pth，在多种 SNR 条件下评估 BER，
并输出汇总表格和 BER vs SNR 曲线。
"""

import torch
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / 'models'
IMAGES_DIR = ROOT / 'images'
LOGS_DIR   = ROOT / 'logs'
IMAGES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from train_rkan import (
    RKANEqualizer,
    KANLinear,
    RKANCell,
    OpticalDataset,
    build_rkan,
    CONFIG as RKAN_CONFIG,
)


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


SNR_LIST    = [0, 5, 10, 15, 20, 25, None]
LABEL_SCALE = 3.0
BATCH_SIZE  = 1024


def snr_label(snr):
    return "无噪声" if snr is None else f"{snr} dB"


def add_awgn(rx_signal, snr_db):
    if snr_db is None or np.isinf(snr_db):
        return rx_signal.copy()
    rx    = np.asarray(rx_signal, dtype=np.float64)
    Ps    = np.mean(rx ** 2)
    Pn    = Ps / (10 ** (snr_db / 10))
    sigma = np.sqrt(Pn)
    noise = np.random.RandomState(seed=42).randn(*rx.shape).astype(np.float64) * sigma
    return (rx + noise).astype(np.float64)


def calculate_ber(pred_scaled, true_scaled):
    pred_labels = np.select(
        [pred_scaled < -2,
         (pred_scaled >= -2) & (pred_scaled < 0),
         (pred_scaled >=  0) & (pred_scaled < 2)],
        [-3, -1, 1], default=3
    )
    errors = np.sum(pred_labels != true_scaled)
    return errors / len(true_scaled)


def run_inference(model, loader, device):
    preds_list, targets_list = [], []
    model.eval()
    with torch.no_grad():
        for inputs, tgt in loader:
            inputs = inputs.unsqueeze(-1).to(device)
            out    = model(inputs).cpu().numpy()
            preds_list.append(out)
            targets_list.append(tgt.numpy())
    preds   = np.concatenate(preds_list).flatten()   * LABEL_SCALE
    targets = np.concatenate(targets_list).flatten() * LABEL_SCALE
    return preds, targets


def test():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = LOGS_DIR / f'test_rkan_{timestamp}.log'
    log = logging.getLogger('test_rkan')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(logging.FileHandler(log_path, mode='w', encoding='utf-8'))
    log.addHandler(logging.StreamHandler())
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for h in log.handlers:
        h.setFormatter(fmt)

    log.info("=" * 60)
    log.info("  RKAN Equalizer — 测试开始")
    log.info("=" * 60)

    device = get_device()
    log.info(f"运行设备: {device}")

    ckpt_path = MODELS_DIR / 'rkan_model.pth'
    if not ckpt_path.exists():
        log.info(f"[ERROR] 未找到 {ckpt_path}，请先运行 train_rkan.py")
        return

    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location=device)
    rx_mean = float(ckpt['rx_mean'])
    rx_std  = float(ckpt['rx_std'])
    config  = ckpt.get('config', dict(RKAN_CONFIG))
    config['device'] = device

    log.info(f"模型来自 Epoch {ckpt.get('epoch', 'N/A')}, "
             f"训练最优 Val BER = {ckpt.get('best_val_ber', 'N/A')}")
    log.info(f"归一化: mean={rx_mean:.4f}, std={rx_std:.4f}")

    total_params = sum(p.numel() for p in build_rkan(config, 'cpu').parameters())
    log.info(f"模型参数量: {total_params:,}")

    # ---------- 加载测试数据 ----------
    data = scipy.io.loadmat(str(ROOT / 'dataset_for_python.mat'))
    rx_test_base = data['rx_test_export'].flatten()
    if np.iscomplexobj(rx_test_base):
        rx_test_base = np.abs(rx_test_base).astype(np.float64)
    else:
        rx_test_base = rx_test_base.astype(np.float64)
    symb_test = data['symb_test_export'].flatten()
    log.info(f"测试信号点数: {len(rx_test_base):,}  |  测试符号数: {len(symb_test):,}")

    # ---------- 逐 SNR 测试 ----------
    model = build_rkan(config, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    ber_results = {}
    for snr in SNR_LIST:
        rx_test = add_awgn(rx_test_base, snr)
        dataset = OpticalDataset(
            rx_test, symb_test,
            config['window_size'], config['sps'],
            rx_mean=rx_mean, rx_std=rx_std,
            label_scale=LABEL_SCALE,
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        preds, targets = run_inference(model, loader, device)
        ber = calculate_ber(preds, targets)
        ber_results[snr] = ber
        log.info(f"  SNR={snr_label(snr):8s}  →  BER = {ber:.4e}")

    # ---------- 汇总 ----------
    log.info("\n" + "=" * 40)
    log.info("         RKAN BER 汇总")
    log.info("=" * 40)
    log.info(f"{'SNR':^12} | {'BER':^16}")
    log.info("-" * 32)
    for snr in SNR_LIST:
        log.info(f"{snr_label(snr):^12} | {ber_results[snr]:^16.4e}")
    log.info("=" * 40)

    # ---------- 绘图 ----------
    x_labels = [snr_label(s) for s in SNR_LIST]
    x_pos    = np.arange(len(x_labels))
    bers     = [ber_results[s] for s in SNR_LIST]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(x_pos, bers, 'o-', color='C0', linewidth=2, markersize=8, label='RKAN')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('RKAN 均衡器 — BER vs SNR', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()

    out_fig = IMAGES_DIR / 'rkan_ber_vs_snr.png'
    plt.savefig(str(out_fig), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"BER 曲线已保存: {out_fig}")
    log.info(f"测试日志已保存: {log_path}")


if __name__ == '__main__':
    test()
