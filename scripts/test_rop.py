"""
ROP 泛化实验 — 测试脚本（CPU 多进程并行版）

加载由 train_all_rop.py 训练好的六种均衡器，在 ROP=0~-15 dBm 数据上并行推理，
将 BER 结果保存到 ROP测试数据/ 文件夹（CSV + 图）。

并行策略（绕过 GIL 的多进程，适合 8 核 CPU）:
  - 每个子进程负责一个模型的全部 ROP 点推理
  - 进程数 = min(CPU_WORKERS, 可用模型数)，默认取 os.cpu_count()=8
  - 每进程内 torch.set_num_threads(1)，防止 8进程×8线程=64线程争抢

图表规格:
  - 1:1 正方形 (6 in × 6 in)
  - Times New Roman 字体，14 pt
  - BER vs ROP（半对数，y 轴对数刻度）

运行前请先:
  1. 运行 RX_rop.m          ← 生成各 ROP 的 .mat 测试文件
  2. 运行 train_all_rop.py  ← 在 ROP=0 数据上训练所有模型
"""

import os
import csv
import sys
import numpy as np
import torch
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import DataLoader

# ---------- 复用已有模型定义 ----------
from train_fcnn    import FCNNEqualizer,   OpticalDataset, CONFIG as FCNN_CFG
from train_dnn     import DNNEqualizer,    CONFIG as DNN_CFG
from train_bilstm  import BiLSTMEqualizer, CONFIG as BILSTM_CFG
from train_kan_ideas import (
    build_kan_fcnn,   KAN_FCNN_CONFIG,
    build_hybrid_kan, HYBRID_KAN_CONFIG,
    build_res_kan,    RES_KAN_CONFIG,
)

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / 'models'
ROP_DIR    = ROOT / 'ROP测试数据'
ROP_DIR.mkdir(exist_ok=True)

ROP_LIST    = list(range(0, -16, -1))   # 0, -1, -2, ..., -15
LABEL_SCALE = 3.0
BATCH_SIZE  = 4096
CPU_WORKERS = None   # None → 自动取 os.cpu_count()（您的机器为 8 核）


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


DEVICE = get_device()


# ================= 模型构建函数（主进程 & 子进程均可调用）=================
def _build_fcnn(config):
    return FCNNEqualizer(
        input_dim   = config['window_size'],
        hidden_dims = config['hidden_dims'],
    ).to('cpu')


def _build_dnn(config):
    return DNNEqualizer(
        input_dim     = config['window_size'],
        hidden_dims   = config['hidden_dims'],
        use_batchnorm = config['use_batchnorm'],
        dropout       = config['dropout'],
    ).to('cpu')


def _build_bilstm(config):
    return BiLSTMEqualizer(
        input_size  = 1,
        hidden_size = config['hidden_size'],
        num_layers  = config['num_lstm_layers'],
        use_center  = config['use_center'],
        window_size = config['window_size'],
    ).to('cpu')


def _build_kan_fcnn(config):
    return build_kan_fcnn(config, 'cpu')


def _build_hybrid_kan(config):
    return build_hybrid_kan(config, 'cpu')


def _build_res_kan(config):
    return build_res_kan(config, 'cpu')


# ================= 模型注册表 =================
MODEL_REGISTRY = [
    {
        'name': 'FCNN',       'build': _build_fcnn,
        'default_cfg': dict(FCNN_CFG),
        'ckpt': 'fcnn_rop_model.pth',
        'color': 'C2', 'marker': '^', 'ls': '-.',
    },
    {
        'name': 'DNN',        'build': _build_dnn,
        'default_cfg': dict(DNN_CFG),
        'ckpt': 'dnn_rop_model.pth',
        'color': 'C4', 'marker': 'v', 'ls': '-.',
    },
    {
        'name': 'BiLSTM',     'build': _build_bilstm,
        'default_cfg': dict(BILSTM_CFG),
        'ckpt': 'bilstm_rop_model.pth',
        'color': 'C3', 'marker': 'D', 'ls': ':',
    },
    {
        'name': 'KAN-FCNN',   'build': _build_kan_fcnn,
        'default_cfg': dict(KAN_FCNN_CONFIG),
        'ckpt': 'kan_fcnn_rop_model.pth',
        'color': 'C0', 'marker': 'o', 'ls': '-',
    },
    {
        'name': 'Hybrid-KAN', 'build': _build_hybrid_kan,
        'default_cfg': dict(HYBRID_KAN_CONFIG),
        'ckpt': 'hybrid_kan_rop_model.pth',
        'color': 'C1', 'marker': 's', 'ls': '-',
    },
    {
        'name': 'ResKAN',     'build': _build_res_kan,
        'default_cfg': dict(RES_KAN_CONFIG),
        'ckpt': 'res_kan_rop_model.pth',
        'color': 'C6', 'marker': 'p', 'ls': '-',
    },
]


# ================= 子进程工作函数（必须在模块顶层，Windows spawn 可 pickle）=================
def _worker(args):
    """
    每个子进程负责一个模型的全部 ROP 点推理。
    args: (name, build_fn, config, ckpt_path_str, rop_data_paths, rx_mean, rx_std)
    返回: (name, {rop: ber})
    """
    name, build_fn, config, ckpt_path_str, rop_data_paths, rx_mean, rx_std = args

    # 限制每个子进程的 PyTorch 内部线程数，防止过度订阅
    torch.set_num_threads(1)

    # 加载模型（每个子进程只加载一次）
    ckpt  = torch.load(ckpt_path_str, weights_only=False, map_location='cpu')
    model = build_fn(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    model_results = {}
    for rop, mat_path in rop_data_paths.items():
        data = scipy.io.loadmat(mat_path)
        rx   = data['rx_test_export'].flatten().astype(np.float64)
        sym  = data['symb_test_export'].flatten()
        if np.iscomplexobj(rx):
            rx = np.abs(rx)

        ds = OpticalDataset(
            rx, sym,
            config['window_size'], config['sps'],
            rx_mean=rx_mean, rx_std=rx_std, label_scale=LABEL_SCALE,
        )
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

        preds_list, tgt_list = [], []
        with torch.inference_mode():
            for x, y in ld:
                x = x.unsqueeze(-1)
                preds_list.append(model(x))
                tgt_list.append(y)

        preds   = torch.cat(preds_list).flatten().numpy() * LABEL_SCALE
        targets = torch.cat(tgt_list).flatten().numpy()   * LABEL_SCALE
        pred_lbl = np.select([preds   < -2, preds   < 0, preds   < 2], [-3, -1, 1], default=3)
        true_lbl = np.select([targets < -2, targets < 0, targets < 2], [-3, -1, 1], default=3)
        model_results[rop] = float(np.mean(pred_lbl != true_lbl))

    return name, model_results


# ================= 主流程 =================
def main():
    n_workers = CPU_WORKERS or os.cpu_count() or 1

    print("=" * 65)
    print("  ROP 泛化实验 — 多进程并行测试")
    print(f"  逻辑核心数: {os.cpu_count()}  并行进程数: {n_workers}")
    print("=" * 65)

    # ── 确认数据文件 & 构建 rop_data_paths ──
    rop_data_paths = {}
    for rop in ROP_LIST:
        if rop == 0:
            path = ROP_DIR / 'dataset_rop0_test.mat'
            if not path.exists():
                path = ROOT / 'dataset_rop0_for_python.mat'
        else:
            path = ROP_DIR / f'dataset_rop{rop}_test.mat'
        if path.exists():
            rop_data_paths[rop] = str(path)
        else:
            print(f"  [跳过] ROP={rop:4d} dBm: 数据文件不存在")

    if not rop_data_paths:
        print("\n没有可用的 ROP 测试数据，请先运行 RX_rop.m。")
        sys.exit(1)

    rop_tested = sorted(rop_data_paths.keys(), reverse=True)   # 0, -1, ..., -15

    # ── 确认模型 checkpoint & 构建任务列表 ──
    task_list = []
    available_entries = []
    for entry in MODEL_REGISTRY:
        ckpt_path = MODELS_DIR / entry['ckpt']
        if not ckpt_path.exists():
            print(f"  [跳过] {entry['name']}: {entry['ckpt']} 不存在")
            continue

        ckpt = torch.load(str(ckpt_path), weights_only=False, map_location='cpu')
        cfg  = {**entry['default_cfg']}
        if 'config' in ckpt and ckpt['config']:
            cfg.update(ckpt['config'])
        cfg['device'] = 'cpu'

        rx_mean = float(ckpt['rx_mean'])
        rx_std  = float(ckpt['rx_std'])
        print(f"  [OK]   {entry['name']} 已加载  (mean={rx_mean:.4f}, std={rx_std:.4f})")

        task_list.append((
            entry['name'],
            entry['build'],
            cfg,
            str(ckpt_path),
            rop_data_paths,
            rx_mean,
            rx_std,
        ))
        available_entries.append(entry)

    if not task_list:
        print("\n没有找到任何 ROP 模型 checkpoint，请先运行 train_all_rop.py。")
        sys.exit(1)

    # ── 多进程并行推理 ──
    actual_workers = min(n_workers, len(task_list))
    print(f"\n[INFO] 启动 {actual_workers} 个子进程（共 {len(task_list)} 个模型任务）...\n")

    results = {}
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        future_to_name = {executor.submit(_worker, args): args[0] for args in task_list}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                _, model_results = future.result()
                results[name] = model_results
                for rop in rop_tested:
                    ber = model_results.get(rop, float('nan'))
                    print(f"  [{name:12s}] ROP={rop:4d} dBm | BER = {ber:.4e}")
            except Exception as exc:
                print(f"  [{name:12s}] 子进程失败: {exc}")

    # ── 保存 CSV ──
    names    = [e['name'] for e in available_entries if e['name'] in results]
    csv_path = ROP_DIR / 'rop_ber_comparison.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['ROP(dBm)'] + names)
        writer.writeheader()
        for rop in rop_tested:
            row = {'ROP(dBm)': rop}
            for name in names:
                row[name] = results.get(name, {}).get(rop, float('nan'))
            writer.writerow(row)
    print(f"\nCSV 已保存: {csv_path}")

    # ── 绘图 ──
    matplotlib.rcParams.update({
        'font.family':     'Times New Roman',
        'font.size':       14,
        'axes.titlesize':  14,
        'axes.labelsize':  14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'axes.unicode_minus': False,
    })

    fig, ax = plt.subplots(figsize=(6, 6))   # 1:1 正方形

    for entry in available_entries:
        name = entry['name']
        if name not in results:
            continue
        bers = [results[name].get(rop, float('nan')) for rop in rop_tested]
        ax.semilogy(
            rop_tested, bers,
            marker    = entry['marker'],
            linestyle = entry['ls'],
            color     = entry['color'],
            linewidth = 1.5,
            markersize= 7,
            label     = name,
        )

    ax.set_xlabel('Received Optical Power (dBm)')
    ax.set_ylabel('Bit Error Rate (BER)')
    ax.set_title('BER vs. Received Optical Power')
    ax.set_xticks(rop_tested)
    ax.invert_xaxis()                          # 0 dBm 在左，-15 dBm 在右
    ax.legend(loc='upper right', framealpha=0.8)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_path = ROP_DIR / 'rop_ber_comparison.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {fig_path}")


# Windows spawn 模式要求多进程入口必须在此守卫下
if __name__ == '__main__':
    main()
