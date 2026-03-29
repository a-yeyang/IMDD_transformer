"""
六种均衡器综合对比测试 (不含 Transformer 系列)

在 SNR = 0, 5, 10, 15, 20, 25 dB 及无噪声条件下评估所有模型的 BER，
并输出对比表格、CSV 和 BER vs SNR 曲线图。

模型清单:
  基线:
    1. FCNN           (fcnn_model.pth)
    2. DNN            (dnn_model.pth)
    3. BiLSTM         (bilstm_model.pth)
  KAN 变体:
    4. KAN-FCNN       (kan_fcnn_model.pth)     — 纯 KAN 前馈
    5. Hybrid-KAN     (hybrid_kan_model.pth)   — FCNN 前端 + KAN 输出
    6. ResKAN         (res_kan_model.pth)      — FCNN + KAN 残差

性能优化说明:
  - 每个模型的 checkpoint 只从磁盘加载一次（原来每个 SNR 点都重复加载）
  - 去掉 ThreadPoolExecutor：GIL 使线程无法真正并行 CPU 推理，改为串行循环
  - run_inference 改为在设备上拼接 tensor 后一次性 .cpu()，避免逐 batch PCIe 传输
  - 使用 torch.inference_mode() 代替 no_grad()，节省梯度追踪开销
  - BATCH_SIZE 增大以减少 kernel 启动次数；GPU 时启用 pin_memory

CPU 多核并行说明 (FORCE_CPU 开关):
  - 打开 FORCE_CPU=True 后，强制全程使用 CPU，并用 ProcessPoolExecutor 实现
    真正的多进程并行（绕过 GIL），每个进程独立负责一个模型的全部 SNR 测试
  - CPU_WORKERS 控制并行进程数，默认取 os.cpu_count()（您的机器为 8 核）
  - 每个子进程内调用 torch.set_num_threads(1)，防止多进程 × 多线程的过度订阅
    （例如 8 进程×8线程 = 64 线程争用，反而更慢）
"""

import os
import torch
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
import csv
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
    KANFCNNEqualizer,
    HybridKANEqualizer,
    ResKANEqualizer,
    build_kan_fcnn,  KAN_FCNN_CONFIG,
    build_hybrid_kan, HYBRID_KAN_CONFIG,
    build_res_kan,   RES_KAN_CONFIG,
)

# ================= 全局配置 =================
SNR_LIST    = [0, 5, 10, 15, 20, 25, None]
LABEL_SCALE = 3.0
BATCH_SIZE  = 4096  # 增大 batch 减少 kernel 启动次数

# ----- CPU 多核并行开关 -----
# True : 强制使用 CPU，并用 ProcessPoolExecutor 多进程并行（每个进程负责一个模型）
# False: 自动选择设备（优先 GPU），串行逐模型推理
FORCE_CPU = True

# 并行进程数。None 表示自动取 os.cpu_count()（通常等于逻辑核心数）
# 您的机器有 8 核，可设为 8 或更小的值；设为 1 退化为单进程 CPU 串行
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
    """在设备上完成所有 batch 推理后，统一转 CPU，避免逐 batch PCIe 传输。"""
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


# ================= CPU 多进程工作函数 =================
# 必须定义在模块顶层，才能在 Windows spawn 模式下被 pickle 序列化。
# 每个子进程独立负责一个模型的全部 SNR 推理，避免重复加载 checkpoint。
def _model_worker(task_args):
    """
    子进程任务：加载一个模型，遍历所有 SNR 点，返回该模型的 BER 字典。

    设计要点:
      - torch.set_num_threads(1): 防止 N进程 × M线程 的过度订阅。
        每个进程独占一个核，PyTorch 内部 BLAS 也只用单线程。
      - 模型在子进程内仅加载一次，然后复用到所有 SNR 点。
    """
    (tag, build_fn, config, ckpt_path_str,
     snr_list, rx_test_base, symb_test, rx_mean, rx_std) = task_args

    # 限制每个子进程只使用 1 个 OpenMP/MKL 线程，防止过度订阅
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


# ================= 主测试流程 =================
def test():
    log, log_path = setup_logger()

    log.info("=" * 80)
    log.info("   六种均衡器综合对比 — BER vs SNR")
    log.info("   (FCNN / DNN / BiLSTM / KAN-FCNN / Hybrid-KAN / ResKAN)")
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

    # ---------- 加载所有模型（主进程，用于参数量统计和归一化参数读取）----------
    log.info("\n[INFO] 正在加载模型 checkpoint（主进程，仅用于参数统计）...")
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

        # 主进程只需加载模型用于参数量统计，不用于推理（推理在子进程中完成）
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
        log.info(f"  {entry['name']:15s} — Epoch {epoch}, "
                 f"训练 Val BER={best_ber}, "
                 f"mean={rx_mean:.4f}, std={rx_std:.4f}")

    # ---------- 参数量统计（复用主进程已加载的模型）----------
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
    log.info(f"[INFO] 可用模型: {len(available)}")
    log.info(f"[INFO] batch 大小: {BATCH_SIZE}\n")

    # ---------- 推理：根据 FORCE_CPU 选择并行或串行路径 ----------
    results = {entry['tag']: {} for entry in available}

    if FORCE_CPU:
        # ===== CPU 多进程并行路径 =====
        # 每个子进程负责一个模型的所有 SNR 点，进程间真正并行（无 GIL）。
        # 注意：rx_test_base 和 symb_test 通过 pickle 传给每个子进程（一次性开销）。
        actual_workers = min(n_workers, len(available))
        log.info(f"[INFO] 启动 {actual_workers} 个子进程（共 {len(available)} 个模型任务）...\n")

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
                    # 按 SNR 顺序打印该模型的所有结果
                    for snr in SNR_LIST:
                        ber = model_results.get(snr, float('nan'))
                        log.info(f"  [{tag:8s}] SNR={snr_label(snr):8s}  →  BER = {ber:.4e}")
                except Exception as exc:
                    log.error(f"  [{tag:8s}] 子进程推理失败: {exc}")

    else:
        # ===== GPU/CPU 串行路径 =====
        # 模型在主进程中复用，无需重复加载 checkpoint。
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


# Windows 的 spawn 模式要求多进程程序必须在此守卫下启动，
# 否则子进程会反复执行顶层代码导致递归创建进程。
if __name__ == '__main__':
    test()
