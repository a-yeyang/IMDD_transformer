"""
五种均衡器综合对比测试
在 SNR = 0, 5, 10, 15, 20 dB 及无噪声条件下评估所有模型的 BER，
并输出对比表格、CSV 和 BER vs SNR 曲线图。

模型清单:
  1. FCNN              (fcnn_model.pth)
  2. DNN               (dnn_model.pth)
  3. BiLSTM            (bilstm_model.pth)
  4. Transformer TEQ   (teq_model.pth)
  5. KAN-Former        (kan_teq_model.pth)
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

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / 'models'
IMAGES_DIR = ROOT / 'images'
LOGS_DIR   = ROOT / 'logs'
IMAGES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------- 从各训练脚本导入模型定义 ----------
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
from train_fcnn import (
    FCNNEqualizer,
    CONFIG as FCNN_CONFIG,
)
from train_bilstm import (
    BiLSTMEqualizer,
    CONFIG as BILSTM_CONFIG,
)
from train_dnn import (
    DNNEqualizer,
    CONFIG as DNN_CONFIG,
)

# ================= 全局配置 =================
SNR_LIST    = [0, 5, 10, 15, 20, None]
MAX_WORKERS = 4
LABEL_SCALE = 3.0
BATCH_SIZE  = 1024


# ================= 日志 =================
def setup_logger():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = LOGS_DIR / f'test_all_{timestamp}.log'
    log = logging.getLogger('test_all')
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


# ================= 模型构建函数 =================
def build_teq(config, device):
    return LightweightTransformerEQ(
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


def build_kan(config, device):
    return KANFormerEQ(
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


def build_fcnn(config, device):
    return FCNNEqualizer(
        input_dim   = config['window_size'],
        hidden_dims = config['hidden_dims'],
    ).to(device)


def build_bilstm(config, device):
    return BiLSTMEqualizer(
        input_size  = 1,
        hidden_size = config['hidden_size'],
        num_layers  = config['num_lstm_layers'],
        use_center  = config['use_center'],
        window_size = config['window_size'],
    ).to(device)


def build_dnn(config, device):
    return DNNEqualizer(
        input_dim     = config['window_size'],
        hidden_dims   = config['hidden_dims'],
        use_batchnorm = config['use_batchnorm'],
        dropout       = config['dropout'],
    ).to(device)


# ================= 模型注册表 =================
MODEL_REGISTRY = [
    {
        'name': 'FCNN',
        'tag':  'FCNN',
        'ckpt': 'fcnn_model.pth',
        'build_fn': build_fcnn,
        'default_config': dict(FCNN_CONFIG),
        'color': 'C2', 'marker': '^', 'ls': '-.',
    },
    {
        'name': 'DNN',
        'tag':  'DNN',
        'ckpt': 'dnn_model.pth',
        'build_fn': build_dnn,
        'default_config': dict(DNN_CONFIG),
        'color': 'C4', 'marker': 'v', 'ls': '-.',
    },
    {
        'name': 'BiLSTM',
        'tag':  'BiLSTM',
        'ckpt': 'bilstm_model.pth',
        'build_fn': build_bilstm,
        'default_config': dict(BILSTM_CONFIG),
        'color': 'C3', 'marker': 'D', 'ls': ':',
    },
    {
        'name': 'Transformer TEQ',
        'tag':  'TEQ',
        'ckpt': 'teq_model.pth',
        'build_fn': build_teq,
        'default_config': dict(TEQ_CONFIG),
        'color': 'C1', 'marker': 's', 'ls': '--',
    },
    {
        'name': 'KAN-Former',
        'tag':  'KAN',
        'ckpt': 'kan_teq_model.pth',
        'build_fn': build_kan,
        'default_config': dict(KAN_CONFIG),
        'color': 'C0', 'marker': 'o', 'ls': '-',
    },
]


# ================= 单任务：特定模型 + 特定 SNR =================
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

    log.info("=" * 72)
    log.info("   五种均衡器综合对比 — BER vs SNR")
    log.info("=" * 72)

    # ---------- 检查可用模型 ----------
    available = []
    for entry in MODEL_REGISTRY:
        ckpt_path = MODELS_DIR / entry['ckpt']
        if ckpt_path.exists():
            available.append(entry)
            log.info(f"  [OK] {entry['name']:20s}  ← {entry['ckpt']}")
        else:
            log.info(f"  [--] {entry['name']:20s}  ← 未找到 {entry['ckpt']}，跳过")

    if not available:
        log.info("\n[ERROR] 没有找到任何模型 checkpoint，请先运行对应的训练脚本。")
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

    log.info(f"\n[INFO] 测试样本数 (接收信号点数): {len(rx_test_base):,}")
    log.info(f"[INFO] 测试符号数:               {len(symb_test):,}")

    # ---------- 加载各模型归一化参数 & 元信息 ----------
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

        model_meta[entry['tag']] = {
            'rx_mean': rx_mean, 'rx_std': rx_std,
            'epoch': epoch, 'best_ber': best_ber,
        }
        log.info(f"  {entry['name']:20s} — Epoch {epoch}, "
                 f"训练最优Val BER={best_ber}, "
                 f"mean={rx_mean:.4f}, std={rx_std:.4f}")

    # ---------- 参数量统计 ----------
    log.info("\n" + "=" * 72)
    log.info("                    模型参数量对比")
    log.info("=" * 72)
    log.info(f"{'模型':^20} | {'总参数量':^12} | {'可训练参数量':^12}")
    log.info("-" * 50)
    for entry in available:
        config = entry['default_config']
        device = 'cpu'
        tmp_model = entry['build_fn'](config, device)
        total = sum(p.numel() for p in tmp_model.parameters())
        trainable = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad)
        log.info(f"{entry['name']:^20} | {total:^12,} | {trainable:^12,}")
        del tmp_model
    log.info("=" * 72)

    log.info(f"\n[INFO] 测试 SNR 条件: {[snr_label(s) for s in SNR_LIST]}")
    log.info(f"[INFO] 可用模型数: {len(available)}")
    log.info(f"[INFO] 并行线程数: {MAX_WORKERS}\n")

    # ---------- 并行提交所有测试任务 ----------
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
                log.info(f"  [{tag:7s}] SNR={snr_label(snr):8s}  →  BER = {ber:.4e}")
            except Exception as exc:
                log.info(f"  [{tag:7s}] SNR={snr_label(snr):8s}  →  错误: {exc}")

    # ---------- 汇总表格 ----------
    tags = [e['tag'] for e in available]
    names = [e['name'] for e in available]

    log.info("\n" + "=" * (20 + 18 * len(available)))
    log.info("             BER 对比汇总（PAM4 硬判决，门限 -2 / 0 / 2）")
    log.info("=" * (20 + 18 * len(available)))

    header = f"{'SNR':^12} |"
    for n in names:
        header += f" {n:^16} |"
    log.info(header)
    log.info("-" * len(header))

    csv_rows = []
    for snr in SNR_LIST:
        row_str = f"{snr_label(snr):^12} |"
        csv_row = {'SNR': snr_label(snr)}
        for entry in available:
            ber = results[entry['tag']].get(snr, float('nan'))
            row_str += f" {ber:^16.4e} |"
            csv_row[entry['name']] = ber
        log.info(row_str)
        csv_rows.append(csv_row)

    log.info("=" * len(header))

    # ---------- 保存 CSV ----------
    csv_path = ROOT / 'all_comparison_results.csv'
    fieldnames = ['SNR'] + names
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info(f"\n[INFO] 对比结果已保存至: {csv_path}")

    # ---------- 绘图：BER vs SNR ----------
    x_labels = [snr_label(s) for s in SNR_LIST]
    x_pos    = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(11, 6))

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
    ax.set_ylabel('BER',      fontsize=12)
    ax.set_title('IM/DD PAM4 均衡器对比 — BER vs SNR', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_fig = IMAGES_DIR / 'all_ber_comparison.png'
    plt.savefig(str(out_fig), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"[INFO] BER 对比图已保存: {out_fig}")
    log.info(f"[INFO] 测试日志已保存:   {log_path}")


if __name__ == '__main__':
    test()
