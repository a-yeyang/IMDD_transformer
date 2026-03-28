"""
七种均衡器综合对比测试 (不含 Transformer 系列)

在 SNR = 0, 5, 10, 15, 20, 25 dB 及无噪声条件下评估所有模型的 BER，
并输出对比表格、CSV 和 BER vs SNR 曲线图。

模型清单:
  基线:
    1. FCNN           (fcnn_model.pth)
    2. DNN            (dnn_model.pth)
    3. BiLSTM         (bilstm_model.pth)
  KAN 变体:
    4. RKAN           (rkan_model.pth)         — 递归 KAN (对照)
    5. KAN-FCNN       (kan_fcnn_model.pth)     — 纯 KAN 前馈
    6. Hybrid-KAN     (hybrid_kan_model.pth)   — FCNN 前端 + KAN 输出
    7. ResKAN         (res_kan_model.pth)      — FCNN + KAN 残差
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


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / 'models'
IMAGES_DIR = ROOT / 'images'
LOGS_DIR   = ROOT / 'logs'
IMAGES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------- 导入各模型定义 ----------
from train_fcnn import (
    FCNNEqualizer,
    OpticalDataset,
    CONFIG as FCNN_CONFIG,
)
from train_dnn import (
    DNNEqualizer,
    CONFIG as DNN_CONFIG,
)
from train_bilstm import (
    BiLSTMEqualizer,
    CONFIG as BILSTM_CONFIG,
)
from train_rkan import (
    RKANEqualizer, KANLinear, RKANCell,
    build_rkan,
    CONFIG as RKAN_CONFIG,
)
from train_kan_ideas import (
    KANFCNNEqualizer,
    HybridKANEqualizer,
    ResKANEqualizer,
    build_kan_fcnn,  KAN_FCNN_CONFIG,
    build_hybrid_kan, HYBRID_KAN_CONFIG,
    build_res_kan,   RES_KAN_CONFIG,
)

# ================= 全局配置 =================
SNR_LIST    = [0, 5, 10, 15, 20, 25, None]
MAX_WORKERS = 4
LABEL_SCALE = 3.0
BATCH_SIZE  = 1024


# ================= 日志 =================
def setup_logger():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = LOGS_DIR / f'compare_all_{timestamp}.log'
    log = logging.getLogger('compare_all')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(logging.FileHandler(log_path, mode='w', encoding='utf-8'))
    log.addHandler(logging.StreamHandler())
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for h in log.handlers:
        h.setFormatter(fmt)
    return log, log_path


# ================= 工具函数 =================
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


# ================= 模型构建函数 (基线) =================
def build_fcnn(config, device):
    return FCNNEqualizer(
        input_dim   = config['window_size'],
        hidden_dims = config['hidden_dims'],
    ).to(device)


def build_dnn(config, device):
    return DNNEqualizer(
        input_dim     = config['window_size'],
        hidden_dims   = config['hidden_dims'],
        use_batchnorm = config['use_batchnorm'],
        dropout       = config['dropout'],
    ).to(device)


def build_bilstm(config, device):
    return BiLSTMEqualizer(
        input_size  = 1,
        hidden_size = config['hidden_size'],
        num_layers  = config['num_lstm_layers'],
        use_center  = config['use_center'],
        window_size = config['window_size'],
    ).to(device)


# ================= 模型注册表 =================
MODEL_REGISTRY = [
    # --- 基线模型 ---
    {
        'name': 'FCNN', 'tag': 'FCNN',
        'ckpt': 'fcnn_model.pth',
        'build_fn': build_fcnn,
        'default_config': dict(FCNN_CONFIG),
        'color': 'C2', 'marker': '^', 'ls': '-.',
        'group': 'baseline',
    },
    {
        'name': 'DNN', 'tag': 'DNN',
        'ckpt': 'dnn_model.pth',
        'build_fn': build_dnn,
        'default_config': dict(DNN_CONFIG),
        'color': 'C4', 'marker': 'v', 'ls': '-.',
        'group': 'baseline',
    },
    {
        'name': 'BiLSTM', 'tag': 'BiLSTM',
        'ckpt': 'bilstm_model.pth',
        'build_fn': build_bilstm,
        'default_config': dict(BILSTM_CONFIG),
        'color': 'C3', 'marker': 'D', 'ls': ':',
        'group': 'baseline',
    },
    # --- KAN 变体 ---
    {
        'name': 'RKAN', 'tag': 'RKAN',
        'ckpt': 'rkan_model.pth',
        'build_fn': build_rkan,
        'default_config': dict(RKAN_CONFIG),
        'color': 'C5', 'marker': 'x', 'ls': '--',
        'group': 'kan',
    },
    {
        'name': 'KAN-FCNN', 'tag': 'KANFCNN',
        'ckpt': 'kan_fcnn_model.pth',
        'build_fn': build_kan_fcnn,
        'default_config': dict(KAN_FCNN_CONFIG),
        'color': 'C0', 'marker': 'o', 'ls': '-',
        'group': 'kan',
    },
    {
        'name': 'Hybrid-KAN', 'tag': 'HybKAN',
        'ckpt': 'hybrid_kan_model.pth',
        'build_fn': build_hybrid_kan,
        'default_config': dict(HYBRID_KAN_CONFIG),
        'color': 'C1', 'marker': 's', 'ls': '-',
        'group': 'kan',
    },
    {
        'name': 'ResKAN', 'tag': 'ResKAN',
        'ckpt': 'res_kan_model.pth',
        'build_fn': build_res_kan,
        'default_config': dict(RES_KAN_CONFIG),
        'color': 'C6', 'marker': 'p', 'ls': '-',
        'group': 'kan',
    },
]


# ================= 单任务 =================
def run_single(model_entry, snr_db, rx_test_base, symb_test, rx_mean, rx_std):
    ckpt_path = MODELS_DIR / model_entry['ckpt']
    config = model_entry['default_config']
    device = config['device']

    rx_test = add_awgn(rx_test_base, snr_db)
    dataset = OpticalDataset(
        rx_test, symb_test,
        config['window_size'], config['sps'],
        rx_mean=rx_mean, rx_std=rx_std,
        label_scale=LABEL_SCALE,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = model_entry['build_fn'](config, device)
    ckpt  = torch.load(str(ckpt_path), weights_only=False, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    preds, targets = run_inference(model, loader, device)
    ber = calculate_ber(preds, targets)
    return model_entry['tag'], snr_db, ber


# ================= 主测试流程 =================
def test():
    log, log_path = setup_logger()

    log.info("=" * 80)
    log.info("   七种均衡器综合对比 — BER vs SNR")
    log.info("   (FCNN / DNN / BiLSTM / RKAN / KAN-FCNN / Hybrid-KAN / ResKAN)")
    log.info("=" * 80)

    _device = get_device()
    log.info(f"[INFO] 运行设备: {_device}")

    for entry in MODEL_REGISTRY:
        entry['default_config']['device'] = _device

    # ---------- 检查可用模型 ----------
    available = []
    for entry in MODEL_REGISTRY:
        ckpt_path = MODELS_DIR / entry['ckpt']
        if ckpt_path.exists():
            available.append(entry)
            log.info(f"  [OK] {entry['name']:15s}  ← {entry['ckpt']}")
        else:
            log.info(f"  [--] {entry['name']:15s}  ← 未找到 {entry['ckpt']}，跳过")

    if not available:
        log.info("\n[ERROR] 没有找到任何模型 checkpoint。")
        return

    # ---------- 加载测试数据 ----------
    data = scipy.io.loadmat(str(ROOT / 'dataset_for_python.mat'))
    rx_test_base = data['rx_test_export'].flatten()
    if np.iscomplexobj(rx_test_base):
        log.info("[INFO] 检测到复数信号，取绝对值作为 IM/DD 包络。")
        rx_test_base = np.abs(rx_test_base).astype(np.float64)
    else:
        rx_test_base = rx_test_base.astype(np.float64)
    symb_test = data['symb_test_export'].flatten()

    log.info(f"\n[INFO] 测试信号点数: {len(rx_test_base):,}")
    log.info(f"[INFO] 测试符号数:   {len(symb_test):,}")

    # ---------- 加载归一化参数 ----------
    model_meta = {}
    for entry in available:
        ckpt = torch.load(str(MODELS_DIR / entry['ckpt']),
                          weights_only=False, map_location='cpu')
        rx_mean = float(ckpt['rx_mean'])
        rx_std  = float(ckpt['rx_std'])
        epoch   = ckpt.get('epoch', 'N/A')
        best_ber = ckpt.get('best_val_ber', 'N/A')

        saved_config = ckpt.get('config', None)
        if saved_config is not None:
            entry['default_config'].update(saved_config)
        entry['default_config']['device'] = _device

        model_meta[entry['tag']] = {
            'rx_mean': rx_mean, 'rx_std': rx_std,
            'epoch': epoch, 'best_ber': best_ber,
        }
        log.info(f"  {entry['name']:15s} — Epoch {epoch}, "
                 f"训练 Val BER={best_ber}, "
                 f"mean={rx_mean:.4f}, std={rx_std:.4f}")

    # ---------- 参数量统计 ----------
    log.info("\n" + "=" * 72)
    log.info("                    模型参数量对比")
    log.info("=" * 72)
    log.info(f"{'模型':^20} | {'总参数量':^12} | {'可训练参数量':^12}")
    log.info("-" * 50)
    for entry in available:
        config = entry['default_config']
        tmp_model = entry['build_fn'](config, 'cpu')
        total = sum(p.numel() for p in tmp_model.parameters())
        trainable = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad)
        log.info(f"{entry['name']:^20} | {total:^12,} | {trainable:^12,}")
        del tmp_model
    log.info("=" * 72)

    log.info(f"\n[INFO] 测试 SNR: {[snr_label(s) for s in SNR_LIST]}")
    log.info(f"[INFO] 可用模型: {len(available)}")
    log.info(f"[INFO] 并行线程: {MAX_WORKERS}\n")

    # ---------- 并行测试 ----------
    results = {entry['tag']: {} for entry in available}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {}
        for entry in available:
            meta = model_meta[entry['tag']]
            for snr in SNR_LIST:
                ft = executor.submit(
                    run_single, entry, snr, rx_test_base, symb_test,
                    meta['rx_mean'], meta['rx_std'],
                )
                future_map[ft] = (entry['tag'], snr)

        for future in as_completed(future_map):
            tag, snr = future_map[future]
            try:
                _, _, ber = future.result()
                results[tag][snr] = ber
                log.info(f"  [{tag:8s}] SNR={snr_label(snr):8s}  →  BER = {ber:.4e}")
            except Exception as exc:
                log.info(f"  [{tag:8s}] SNR={snr_label(snr):8s}  →  错误: {exc}")

    # ---------- 汇总表格 ----------
    names = [e['name'] for e in available]

    col_width = max(16, max(len(n) for n in names) + 4)
    sep_len   = 14 + col_width * len(available)

    log.info("\n" + "=" * sep_len)
    log.info("         BER 对比汇总（PAM4 硬判决，门限 -2 / 0 / 2）")
    log.info("=" * sep_len)

    header = f"{'SNR':^12} |"
    for n in names:
        header += f" {n:^{col_width - 2}} |"
    log.info(header)
    log.info("-" * len(header))

    csv_rows = []
    for snr in SNR_LIST:
        row_str = f"{snr_label(snr):^12} |"
        csv_row = {'SNR': snr_label(snr)}
        for entry in available:
            ber = results[entry['tag']].get(snr, float('nan'))
            row_str += f" {ber:^{col_width - 2}.4e} |"
            csv_row[entry['name']] = ber
        log.info(row_str)
        csv_rows.append(csv_row)

    log.info("=" * len(header))

    # ---------- 保存 CSV ----------
    csv_path = ROOT / 'all_equalizer_comparison.csv'
    fieldnames = ['SNR'] + names
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info(f"\n[INFO] CSV 已保存: {csv_path}")

    # ---------- 绘图 ----------
    x_labels = [snr_label(s) for s in SNR_LIST]
    x_pos    = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(13, 7))
    for entry in available:
        bers = [results[entry['tag']].get(s, np.nan) for s in SNR_LIST]
        ax.semilogy(
            x_pos, bers,
            marker=entry['marker'], linestyle=entry['ls'],
            linewidth=2, markersize=8,
            color=entry['color'], label=entry['name'],
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('IM/DD PAM4 均衡器全面对比 — BER vs SNR', fontsize=14)
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_fig = IMAGES_DIR / 'all_equalizer_comparison.png'
    plt.savefig(str(out_fig), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"[INFO] 对比图已保存: {out_fig}")
    log.info(f"[INFO] 日志已保存:   {log_path}")


if __name__ == '__main__':
    test()
