"""
均衡器综合对比测试 (不含 Transformer 系列)

在 SNR = 0, 5, 10, 15, 20, 25 dB 及无噪声条件下评估所有模型的 BER，
并输出对比表格、CSV 和 BER vs SNR 曲线图。

模型清单:
  基线:
    1. FCNN           (fcnn_model.pth)
    2. DNN            (dnn_model.pth)
    3. BiLSTM         (bilstm_model.pth)
  KAN 变体:
    4. VanillaKAN     (vanilla_kan_model.pth) — 纯 KAN 基准
    5. KAN-FCNN       (kan_fcnn_model.pth)    — KAN 前馈
    6. Hybrid-KAN     (hybrid_kan_model.pth)  — FCNN 前端 + KAN 输出
    7. ResKAN         (res_kan_model.pth)     — FCNN + KAN 残差
    8. ConvKAN        (conv_kan_model.pth)    — 1D 卷积 + KAN 管道
    9. KAN-Attention  (kan_attn_model.pth)    — 注意力池化 KAN
   10. MultiScale-KAN (multiscale_kan_model.pth) — 多尺度并行 KAN
   11. GatedKAN       (gated_kan_model.pth)   — 门控双路径 KAN

运行方式:
  python compare_all_equalizers.py                     # 测试全部可用模型
  python compare_all_equalizers.py --models FCNN DNN   # 仅测试指定模型
  python compare_all_equalizers.py --list              # 列出所有可用 tag

增量 CSV:
  仅覆盖本次测试的模型列，保留 CSV 中已有的其他模型结果。
"""

import os
import sys
import argparse
import torch
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
import csv
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from train_kan_ideas import (
    KANLinear,
    VanillaKANEqualizer,
    KANFCNNEqualizer,
    HybridKANEqualizer,
    ResKANEqualizer,
    ConvKANEqualizer,
    KANAttentionEqualizer,
    MultiScaleKANEqualizer,
    GatedKANEqualizer,
    build_vanilla_kan,    VANILLA_KAN_CONFIG,
    build_kan_fcnn,       KAN_FCNN_CONFIG,
    build_hybrid_kan,     HYBRID_KAN_CONFIG,
    build_res_kan,        RES_KAN_CONFIG,
    build_conv_kan,       CONV_KAN_CONFIG,
    build_kan_attention,  KAN_ATTN_CONFIG,
    build_multiscale_kan, MULTISCALE_KAN_CONFIG,
    build_gated_kan,      GATED_KAN_CONFIG,
)

# ================= 全局配置 =================
SNR_LIST    = [0, 5, 10, 15, 20, 25, None]
LABEL_SCALE = 3.0
BATCH_SIZE  = 4096

FORCE_CPU = True
CPU_WORKERS = None


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
    with torch.inference_mode():
        for inputs, tgt in loader:
            inputs = inputs.unsqueeze(-1).to(device, non_blocking=True)
            preds_list.append(model(inputs))
            targets_list.append(tgt)
    preds   = torch.cat(preds_list).flatten().cpu().numpy() * LABEL_SCALE
    targets = torch.cat(targets_list).flatten().numpy()     * LABEL_SCALE
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
        'name': 'VanillaKAN', 'tag': 'VanKAN',
        'ckpt': 'vanilla_kan_model.pth',
        'build_fn': build_vanilla_kan,
        'default_config': dict(VANILLA_KAN_CONFIG),
        'color': 'C9', 'marker': 'P', 'ls': '-',
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
    {
        'name': 'ConvKAN', 'tag': 'ConvKAN',
        'ckpt': 'conv_kan_model.pth',
        'build_fn': build_conv_kan,
        'default_config': dict(CONV_KAN_CONFIG),
        'color': 'C7', 'marker': 'H', 'ls': '-',
        'group': 'kan',
    },
    {
        'name': 'KAN-Attention', 'tag': 'KANAttn',
        'ckpt': 'kan_attn_model.pth',
        'build_fn': build_kan_attention,
        'default_config': dict(KAN_ATTN_CONFIG),
        'color': 'C8', 'marker': '*', 'ls': '-',
        'group': 'kan',
    },
    {
        'name': 'MultiScale-KAN', 'tag': 'MSKAN',
        'ckpt': 'multiscale_kan_model.pth',
        'build_fn': build_multiscale_kan,
        'default_config': dict(MULTISCALE_KAN_CONFIG),
        'color': 'tab:brown', 'marker': 'd', 'ls': '-',
        'group': 'kan',
    },
    {
        'name': 'GatedKAN', 'tag': 'GatedKAN',
        'ckpt': 'gated_kan_model.pth',
        'build_fn': build_gated_kan,
        'default_config': dict(GATED_KAN_CONFIG),
        'color': 'tab:pink', 'marker': 'h', 'ls': '-',
        'group': 'kan',
    },
]

_TAG_TO_ENTRY = {e['tag']: e for e in MODEL_REGISTRY}
_NAME_TO_ENTRY = {e['name']: e for e in MODEL_REGISTRY}


# ================= CPU 多进程工作函数 =================
def _model_worker(task_args):
    (tag, build_fn, config, ckpt_path_str,
     snr_list, rx_test_base, symb_test, rx_mean, rx_std) = task_args

    torch.set_num_threads(1)

    model = build_fn(config, 'cpu')
    ckpt  = torch.load(ckpt_path_str, weights_only=False, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    model_results = {}
    for snr in snr_list:
        rx_test = add_awgn(rx_test_base, snr)
        dataset = OpticalDataset(
            rx_test, symb_test,
            config['window_size'], config['sps'],
            rx_mean=rx_mean, rx_std=rx_std,
            label_scale=LABEL_SCALE,
        )
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=0, pin_memory=False,
        )
        preds, targets = run_inference(model, loader, 'cpu')
        model_results[snr] = calculate_ber(preds, targets)

    return tag, model_results


# ================= 增量 CSV 读写 =================
def load_existing_csv(csv_path):
    """读取已有 CSV，返回 {snr_label: {model_name: ber}} 的嵌套字典。"""
    existing = OrderedDict()
    existing_cols = []
    if not csv_path.exists():
        return existing, existing_cols

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'SNR' not in reader.fieldnames:
            return existing, existing_cols
        existing_cols = [c for c in reader.fieldnames if c != 'SNR']
        for row in reader:
            snr_key = row.get('SNR', '')
            if not snr_key:
                continue
            existing[snr_key] = {}
            for col in existing_cols:
                val = row.get(col, '')
                if val and val.lower() != 'nan':
                    try:
                        existing[snr_key][col] = float(val)
                    except ValueError:
                        pass
    return existing, existing_cols


def save_merged_csv(csv_path, new_results, available_entries,
                    external_results, log):
    """将本次测试结果增量合并到已有 CSV 中。"""
    existing, existing_cols = load_existing_csv(csv_path)

    new_names = [e['name'] for e in available_entries]
    all_names = list(OrderedDict.fromkeys(
        existing_cols + new_names + list(external_results.keys())
    ))

    csv_rows = []
    for snr in SNR_LIST:
        snr_key = snr_label(snr)
        row = {'SNR': snr_key}
        old_row = existing.get(snr_key, {})
        for name in all_names:
            matched_entry = _NAME_TO_ENTRY.get(name)
            if matched_entry and matched_entry['tag'] in new_results:
                ber = new_results[matched_entry['tag']].get(snr, float('nan'))
                row[name] = ber
            elif name in external_results:
                row[name] = external_results[name].get(snr_key, float('nan'))
            elif name in old_row:
                row[name] = old_row[name]
            else:
                row[name] = float('nan')
        csv_rows.append(row)

    fieldnames = ['SNR'] + all_names
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info(f"\n[INFO] CSV 已保存 (增量合并): {csv_path}")
    return all_names, csv_rows


def load_external_classical_results(csv_path):
    """从已有 CSV 读取外部生成的经典均衡器结果。"""
    external_results = {}
    if not csv_path.exists():
        return external_results

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        registry_names = {e['name'] for e in MODEL_REGISTRY}
        external_names = [name for name in fieldnames
                          if name != 'SNR' and name not in registry_names]
        if not external_names:
            return external_results

        for name in external_names:
            external_results[name] = {}

        for row in reader:
            snr_name = row.get('SNR')
            if not snr_name:
                continue
            for name in external_names:
                value = row.get(name, '')
                if value in ('', None):
                    continue
                try:
                    external_results[name][snr_name] = float(value)
                except ValueError:
                    continue

    return external_results


# ================= 主测试流程 =================
def test(model_tags=None):
    """
    model_tags: 要测试的 tag 列表。None 表示测试全部可用模型。
    """
    log, log_path = setup_logger()

    log.info("=" * 80)
    log.info("   均衡器综合对比 — BER vs SNR")
    if model_tags:
        log.info(f"   指定模型: {model_tags}")
    else:
        log.info("   模式: 测试全部可用模型")
    log.info("=" * 80)

    if FORCE_CPU:
        _device = 'cpu'
        n_workers = CPU_WORKERS or os.cpu_count() or 1
        log.info(f"[INFO] 运行模式: CPU 多进程并行（FORCE_CPU=True）")
        log.info(f"[INFO] 逻辑核心数: {os.cpu_count()}  并行进程数: {n_workers}")
    else:
        _device = get_device()
        log.info(f"[INFO] 运行模式: 自动设备串行（FORCE_CPU=False）")
        log.info(f"[INFO] 运行设备: {_device}")

    # ---------- 筛选注册表 ----------
    if model_tags:
        candidates = [e for e in MODEL_REGISTRY if e['tag'] in model_tags]
    else:
        candidates = list(MODEL_REGISTRY)

    for entry in candidates:
        entry['default_config']['device'] = _device

    available = []
    for entry in candidates:
        ckpt_path = MODELS_DIR / entry['ckpt']
        if ckpt_path.exists():
            available.append(entry)
            log.info(f"  [OK] {entry['name']:17s}  ← {entry['ckpt']}")
        else:
            log.info(f"  [--] {entry['name']:17s}  ← 未找到 {entry['ckpt']}，跳过")

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

    # ---------- 加载模型 checkpoint ----------
    log.info("\n[INFO] 正在加载模型 checkpoint...")
    loaded_models = {}
    for entry in available:
        ckpt_path = MODELS_DIR / entry['ckpt']
        ckpt = torch.load(str(ckpt_path), weights_only=False, map_location='cpu')

        rx_mean  = float(ckpt['rx_mean'])
        rx_std   = float(ckpt['rx_std'])
        epoch    = ckpt.get('epoch', 'N/A')
        best_ber = ckpt.get('best_val_ber', 'N/A')

        saved_config = ckpt.get('config', None)
        if saved_config is not None:
            entry['default_config'].update(saved_config)
        entry['default_config']['device'] = _device

        model = entry['build_fn'](entry['default_config'], 'cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        loaded_models[entry['tag']] = {
            'model':    model,
            'rx_mean':  rx_mean,
            'rx_std':   rx_std,
            'epoch':    epoch,
            'best_ber': best_ber,
        }
        log.info(f"  {entry['name']:17s} — Epoch {epoch}, "
                 f"训练 Val BER={best_ber}, "
                 f"mean={rx_mean:.4f}, std={rx_std:.4f}")

    # ---------- 参数量统计 ----------
    log.info("\n" + "=" * 72)
    log.info("                    模型参数量对比")
    log.info("=" * 72)
    log.info(f"{'模型':^20} | {'总参数量':^12} | {'可训练参数量':^12}")
    log.info("-" * 50)
    for entry in available:
        model = loaded_models[entry['tag']]['model']
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"{entry['name']:^20} | {total:^12,} | {trainable:^12,}")
    log.info("=" * 72)

    log.info(f"\n[INFO] 测试 SNR: {[snr_label(s) for s in SNR_LIST]}")
    log.info(f"[INFO] 本次测试模型数: {len(available)}")
    log.info(f"[INFO] batch 大小: {BATCH_SIZE}\n")

    csv_path = ROOT / 'all_equalizer_comparison.csv'
    external_results = load_external_classical_results(csv_path)
    if external_results:
        log.info(f"[INFO] 检测到外部经典均衡器结果，将保留: {list(external_results)}")

    # ---------- 推理 ----------
    results = {entry['tag']: {} for entry in available}

    if FORCE_CPU:
        actual_workers = min(n_workers, len(available))
        log.info(f"[INFO] 启动 {actual_workers} 个子进程...\n")

        task_list = []
        for entry in available:
            info   = loaded_models[entry['tag']]
            config = {k: v for k, v in entry['default_config'].items()}
            config['device'] = 'cpu'
            task_list.append((
                entry['tag'],
                entry['build_fn'],
                config,
                str(MODELS_DIR / entry['ckpt']),
                SNR_LIST,
                rx_test_base,
                symb_test,
                info['rx_mean'],
                info['rx_std'],
            ))

        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            future_to_tag = {
                executor.submit(_model_worker, args): args[0]
                for args in task_list
            }
            for future in as_completed(future_to_tag):
                tag = future_to_tag[future]
                try:
                    _, model_results = future.result()
                    results[tag] = model_results
                    for snr in SNR_LIST:
                        ber = model_results.get(snr, float('nan'))
                        log.info(f"  [{tag:8s}] SNR={snr_label(snr):8s}  →  BER = {ber:.4e}")
                except Exception as exc:
                    log.error(f"  [{tag:8s}] 子进程推理失败: {exc}")

    else:
        pin_mem = (_device != 'cpu')
        for entry in available:
            info    = loaded_models[entry['tag']]
            model   = info['model'].to(_device)
            config  = entry['default_config']
            rx_mean = info['rx_mean']
            rx_std  = info['rx_std']

            for snr in SNR_LIST:
                rx_test = add_awgn(rx_test_base, snr)
                dataset = OpticalDataset(
                    rx_test, symb_test,
                    config['window_size'], config['sps'],
                    rx_mean=rx_mean, rx_std=rx_std,
                    label_scale=LABEL_SCALE,
                )
                loader = DataLoader(
                    dataset, batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=0, pin_memory=pin_mem,
                )
                preds, targets = run_inference(model, loader, _device)
                ber = calculate_ber(preds, targets)
                results[entry['tag']][snr] = ber
                log.info(f"  [{entry['tag']:8s}] SNR={snr_label(snr):8s}  →  BER = {ber:.4e}")

    # ---------- 增量合并 CSV ----------
    all_names, csv_rows = save_merged_csv(
        csv_path, results, available, external_results, log
    )

    # ---------- 汇总表格 ----------
    col_width = max(16, max(len(n) for n in all_names) + 4)
    sep_len   = 14 + col_width * len(all_names)

    log.info("\n" + "=" * sep_len)
    log.info("         BER 对比汇总（PAM4 硬判决，门限 -2 / 0 / 2）")
    log.info("=" * sep_len)

    header = f"{'SNR':^12} |"
    for n in all_names:
        header += f" {n:^{col_width - 2}} |"
    log.info(header)
    log.info("-" * len(header))

    for row in csv_rows:
        row_str = f"{row['SNR']:^12} |"
        for n in all_names:
            v = row.get(n, float('nan'))
            try:
                row_str += f" {float(v):^{col_width - 2}.4e} |"
            except (ValueError, TypeError):
                row_str += f" {'N/A':^{col_width - 2}} |"
        log.info(row_str)

    log.info("=" * len(header))

    # ---------- 绘图（使用 CSV 中全部数据）----------
    existing_data, existing_cols = load_existing_csv(csv_path)
    x_labels = [snr_label(s) for s in SNR_LIST]
    x_pos    = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(10, 8))

    for entry in MODEL_REGISTRY:
        col_name = entry['name']
        if col_name not in existing_cols and col_name not in all_names:
            continue
        bers = []
        for snr_key in x_labels:
            row_data = existing_data.get(snr_key, {})
            bers.append(row_data.get(col_name, np.nan))
        if all(np.isnan(b) for b in bers):
            continue
        ax.semilogy(
            x_pos, bers,
            marker=entry['marker'], linestyle=entry['ls'],
            linewidth=2, markersize=8,
            color=entry['color'], label=entry['name'],
        )

    external_styles = {
        'CMA': {'color': 'C5', 'marker': 'X', 'ls': ':'},
        'Volterra': {'color': 'tab:olive', 'marker': 'P', 'ls': ':'},
    }
    for name in existing_cols:
        if name in _NAME_TO_ENTRY or name == 'SNR':
            continue
        style = external_styles.get(name, {'color': None, 'marker': 'o', 'ls': ':'})
        bers = []
        for snr_key in x_labels:
            row_data = existing_data.get(snr_key, {})
            bers.append(row_data.get(name, np.nan))
        if all(np.isnan(b) for b in bers):
            continue
        ax.semilogy(
            x_pos, bers,
            marker=style['marker'], linestyle=style['ls'],
            linewidth=2, markersize=8,
            color=style['color'], label=name,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('IM/DD PAM4 均衡器全面对比 — BER vs SNR', fontsize=14)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_fig = IMAGES_DIR / 'all_equalizer_comparison.png'
    plt.savefig(str(out_fig), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"[INFO] 对比图已保存: {out_fig}")
    log.info(f"[INFO] 日志已保存:   {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='均衡器综合对比 — BER vs SNR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例:\n'
               '  python compare_all_equalizers.py                    # 全部\n'
               '  python compare_all_equalizers.py --models FCNN DNN  # 指定 tag\n'
               '  python compare_all_equalizers.py --list             # 列出 tag',
    )
    parser.add_argument('--models', nargs='+', metavar='TAG',
                        help='仅测试指定 tag 的模型 (可多个)')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用模型 tag 后退出')
    args = parser.parse_args()

    if args.list:
        print("可用模型 tag:")
        for e in MODEL_REGISTRY:
            ckpt_path = MODELS_DIR / e['ckpt']
            status = "✓" if ckpt_path.exists() else "✗"
            print(f"  {status}  {e['tag']:12s}  ({e['name']})  ← {e['ckpt']}")
        sys.exit(0)

    tags = None
    if args.models:
        valid_tags = {e['tag'] for e in MODEL_REGISTRY}
        tags = [t for t in args.models if t in valid_tags]
        invalid = [t for t in args.models if t not in valid_tags]
        if invalid:
            print(f"[警告] 未知 tag 已忽略: {invalid}")
            print(f"       可用: {sorted(valid_tags)}")
        if not tags:
            sys.exit(1)

    test(model_tags=tags)
