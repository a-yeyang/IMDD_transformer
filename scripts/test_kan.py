"""
KAN-Former vs Transformer TEQ 均衡器对比测试
在 SNR = 0, 5, 10, 15, 20 dB 及无噪声条件下并行评估两个模型的 BER。

模型文件:
  - teq_model.pth        (LightweightTransformerEQ, 由 train_teq.py 生成)
  - kan_teq_model.pth    (KANFormerEQ,              由 train_kan_teq.py 生成)
"""

import torch
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

# 项目根目录（scripts/ 的上一级）
ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / 'models'
IMAGES_DIR = ROOT / 'images'
LOGS_DIR   = ROOT / 'logs'
IMAGES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# 解决 matplotlib 中文乱码
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 从训练脚本导入模型定义和数据集
from train_teq import (
    LightweightTransformerEQ,
    OpticalDataset,
    CONFIG as TEQ_CONFIG,
)
from train_kan_teq import (
    KANFormerEQ,
    FastKANLinear,
    KANTransformerEncoderLayer,
    CONFIG as KAN_CONFIG,
)

# ================= 全局配置 =================
SNR_LIST    = [0, 5, 10, 15, 20, None]  # None 表示无噪声
MAX_WORKERS = 4
LABEL_SCALE = 3.0
BATCH_SIZE  = 1024


# ================= 日志配置 =================
def setup_logger():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = LOGS_DIR / f'test_comparison_{timestamp}.log'
    log = logging.getLogger('test_logger')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(logging.FileHandler(log_path, mode='w', encoding='utf-8'))
    log.addHandler(logging.StreamHandler())
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for h in log.handlers:
        h.setFormatter(formatter)
    return log, log_path


# ================= 工具函数 =================
def snr_label(snr):
    return "无噪声" if snr is None else f"{snr} dB"


def add_awgn(rx_signal, snr_db):
    """按指定 SNR(dB) 向接收信号叠加 AWGN 白噪声。"""
    if snr_db is None or np.isinf(snr_db):
        return rx_signal.copy()
    rx    = np.asarray(rx_signal, dtype=np.float64)
    Ps    = np.mean(rx ** 2)
    Pn    = Ps / (10 ** (snr_db / 10))
    sigma = np.sqrt(Pn)
    noise = np.random.RandomState(seed=42).randn(*rx.shape).astype(np.float64) * sigma
    return (rx + noise).astype(np.float64)


def calculate_ber(pred_scaled, true_scaled):
    """
    PAM4 硬判决（判决门限 -2, 0, 2），返回 BER 及预测符号序列。
    pred_scaled / true_scaled 均已还原到 {-3, -1, 1, 3} 量纲。
    """
    pred_labels = np.select(
        [pred_scaled < -2,
         (pred_scaled >= -2) & (pred_scaled < 0),
         (pred_scaled >=  0) & (pred_scaled < 2)],
        [-3, -1, 1], default=3
    )
    errors = np.sum(pred_labels != true_scaled)
    ber    = errors / len(true_scaled)
    return ber, pred_labels


def _run_inference(model, loader, device):
    """通用推理循环，返回 (preds, targets) numpy 数组（已还原量纲）。"""
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


# ================= 单 SNR 测试函数 =================
def run_teq_snr(snr_db, rx_test_base, symb_test, ckpt_path, config, rx_mean, rx_std):
    """Transformer TEQ 在指定 SNR 下的 BER 测试。"""
    rx_test = add_awgn(rx_test_base, snr_db)
    dataset = OpticalDataset(
        rx_test, symb_test,
        config['window_size'], config['sps'],
        rx_mean=rx_mean, rx_std=rx_std,
        label_scale=LABEL_SCALE,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    device = config['device']

    model = LightweightTransformerEQ(
        input_dim         = config['window_size'],
        d_model           = config['d_model'],
        nhead             = config['nhead'],
        num_layers        = config['num_layers'],
        dim_feedforward   = config['dim_feedforward'],
        num_passes        = config['num_passes'],
        use_weight_sharing= config['use_weight_sharing'],
        use_center_token  = config['use_center_token'],
        use_sinusoidal_pe = config['use_sinusoidal_pe'],
    ).to(device)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    preds, targets = _run_inference(model, loader, device)
    ber, _         = calculate_ber(preds, targets)
    return snr_db, ber


def run_kan_snr(snr_db, rx_test_base, symb_test, ckpt_path, config, rx_mean, rx_std):
    """KAN-Former 在指定 SNR 下的 BER 测试。"""
    rx_test = add_awgn(rx_test_base, snr_db)
    dataset = OpticalDataset(
        rx_test, symb_test,
        config['window_size'], config['sps'],
        rx_mean=rx_mean, rx_std=rx_std,
        label_scale=LABEL_SCALE,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    device = config['device']

    model = KANFormerEQ(
        input_dim         = config['window_size'],
        d_model           = config['d_model'],
        nhead             = config['nhead'],
        num_layers        = config['num_layers'],
        dim_feedforward   = config['dim_feedforward'],
        num_passes        = config['num_passes'],
        use_weight_sharing= config['use_weight_sharing'],
        use_center_token  = config['use_center_token'],
        use_sinusoidal_pe = config['use_sinusoidal_pe'],
        grid_size         = config['kan_grid_size'],
    ).to(device)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    preds, targets = _run_inference(model, loader, device)
    ber, _         = calculate_ber(preds, targets)
    return snr_db, ber


# ================= 主测试流程 =================
def test():
    log, log_path = setup_logger()

    log.info("=" * 68)
    log.info("   KAN-Former  vs  Transformer TEQ  —  BER vs SNR 对比测试")
    log.info("=" * 68)

    # ---------- 加载测试数据 ----------
    data = scipy.io.loadmat(str(ROOT / 'dataset_for_python.mat'))
    rx_test_base = data['rx_test_export'].flatten()
    if np.iscomplexobj(rx_test_base):
        log.info("[INFO] 检测到复数信号，取绝对值作为 IM/DD 包络。")
        rx_test_base = np.abs(rx_test_base).astype(np.float64)
    else:
        rx_test_base = rx_test_base.astype(np.float64)
    symb_test = data['symb_test_export'].flatten()

    log.info(f"[INFO] 测试样本数 (接收信号点数): {len(rx_test_base):,}")
    log.info(f"[INFO] 测试符号数:               {len(symb_test):,}")

    # ---------- 加载 checkpoint 并打印模型信息 ----------
    teq_ckpt_path = str(MODELS_DIR / 'teq_model.pth')
    kan_ckpt_path = str(MODELS_DIR / 'kan_teq_model.pth')

    teq_ckpt = torch.load(teq_ckpt_path, weights_only=False, map_location='cpu')
    kan_ckpt = torch.load(kan_ckpt_path, weights_only=False, map_location='cpu')

    teq_mean = float(teq_ckpt['rx_mean'])
    teq_std  = float(teq_ckpt['rx_std'])
    kan_mean = float(kan_ckpt['rx_mean'])
    kan_std  = float(kan_ckpt['rx_std'])

    teq_epoch = teq_ckpt.get('epoch', 'N/A')
    kan_epoch = kan_ckpt.get('epoch', 'N/A')
    teq_best_ber = teq_ckpt.get('best_val_ber', 'N/A')
    kan_best_ber = kan_ckpt.get('best_val_ber', 'N/A')

    log.info(f"\n[INFO] TEQ      — 来自 Epoch {teq_epoch}, 训练集最优Val BER={teq_best_ber}")
    log.info(f"[INFO] TEQ      — 归一化参数: mean={teq_mean:.4f}, std={teq_std:.4f}")
    log.info(f"[INFO] KAN-Former — 来自 Epoch {kan_epoch}, 训练集最优Val BER={kan_best_ber}")
    log.info(f"[INFO] KAN-Former — 归一化参数: mean={kan_mean:.4f}, std={kan_std:.4f}")
    log.info(f"[INFO] 测试 SNR 条件: {[snr_label(s) for s in SNR_LIST]}")
    log.info(f"[INFO] 并行线程数: {MAX_WORKERS}\n")

    teq_config = dict(TEQ_CONFIG)
    kan_config = dict(KAN_CONFIG)

    # ---------- 并行提交所有 SNR 测试任务 ----------
    teq_results = {}
    kan_results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {}

        for snr in SNR_LIST:
            ft = executor.submit(
                run_teq_snr, snr, rx_test_base, symb_test,
                teq_ckpt_path, teq_config, teq_mean, teq_std,
            )
            future_map[ft] = ('TEQ', snr)

            fk = executor.submit(
                run_kan_snr, snr, rx_test_base, symb_test,
                kan_ckpt_path, kan_config, kan_mean, kan_std,
            )
            future_map[fk] = ('KAN', snr)

        for future in as_completed(future_map):
            model_tag, snr = future_map[future]
            try:
                _, ber = future.result()
                (teq_results if model_tag == 'TEQ' else kan_results)[snr] = ber
                log.info(f"  [{model_tag:3s}] SNR={snr_label(snr):8s}  →  BER = {ber:.4e}")
            except Exception as exc:
                log.info(f"  [{model_tag:3s}] SNR={snr_label(snr):8s}  →  错误: {exc}")

    # ---------- 汇总打印 ----------
    log.info("\n" + "=" * 68)
    log.info("           BER 对比汇总（PAM4 硬判决，门限 -2 / 0 / 2）")
    log.info("=" * 68)
    header = f"{'SNR':^12} | {'Transformer TEQ':^20} | {'KAN-Former':^20} | {'改善幅度':^12}"
    log.info(header)
    log.info("-" * 68)

    csv_rows = []
    for snr in SNR_LIST:
        tb = teq_results.get(snr, float('nan'))
        kb = kan_results.get(snr, float('nan'))
        if not np.isnan(tb) and not np.isnan(kb) and tb > 0:
            improve  = (tb - kb) / tb * 100
            imp_str  = f"{improve:+.1f}%"
        else:
            improve = float('nan')
            imp_str = "  N/A"
        log.info(f"{snr_label(snr):^12} | {tb:^20.4e} | {kb:^20.4e} | {imp_str:^12}")
        csv_rows.append({
            'SNR': snr_label(snr),
            'TEQ_BER': tb,
            'KAN_BER': kb,
            'Improvement(%)': improve,
        })

    log.info("=" * 68)
    log.info("  改善幅度 = (TEQ_BER - KAN_BER) / TEQ_BER × 100%，正值表示 KAN 更优。")

    # ---------- 保存 CSV 结果 ----------
    csv_path = ROOT / 'comparison_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['SNR', 'TEQ_BER', 'KAN_BER', 'Improvement(%)'])
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info(f"\n[INFO] 对比结果已保存至: {csv_path}")

    # ---------- 绘图：BER vs SNR 半对数坐标 ----------
    teq_bers = [teq_results.get(s, np.nan) for s in SNR_LIST]
    kan_bers = [kan_results.get(s, np.nan) for s in SNR_LIST]
    x_labels = [snr_label(s) for s in SNR_LIST]
    x_pos    = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(x_pos, teq_bers, 's--', linewidth=2, markersize=8,
                color='C1', label='Transformer TEQ (基准)')
    ax.semilogy(x_pos, kan_bers, 'o-',  linewidth=2, markersize=8,
                color='C0', label='KAN-Former (创新)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER',      fontsize=12)
    ax.set_title('KAN-Former vs Transformer TEQ — BER vs SNR', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_fig = IMAGES_DIR / 'kan_ber_comparison.png'
    plt.savefig(str(out_fig), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"[INFO] BER 对比图已保存: {out_fig}")
    log.info(f"[INFO] 测试日志已保存:   {log_path}")


if __name__ == '__main__':
    test()
