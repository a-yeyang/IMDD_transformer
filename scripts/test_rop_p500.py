"""
ROP 泛化实验 — 500 参数量级消融测试脚本（CPU 多进程并行）

加载由 train_all_rop_p500.py 训练的六种 ~500 参数均衡器，
在 ROP=0~-15 dBm 数据上并行推理，将 BER 结果增量保存。

运行方式:
  python test_rop_p500.py                                # 全部
  python test_rop_p500.py --models ResKAN ResFCNN         # 指定
  python test_rop_p500.py --list                          # 列出
  python test_rop_p500.py --plot-only                     # 仅重绘图

增量 CSV:
  仅覆盖本次测试的模型列，保留 CSV 中已有的其他模型结果。
"""

import argparse
import os
import csv
import sys
import numpy as np
import torch
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import DataLoader

# ---------- 复用已有模型定义 ----------
from train_fcnn   import FCNNEqualizer,   OpticalDataset
from train_dnn    import DNNEqualizer
from train_bilstm import BiLSTMEqualizer
from train_kan_ideas import build_vanilla_kan, build_res_kan
from train_all_rop_p500 import (
    ResFCNNEqualizer, build_res_fcnn,
    FCNN_P500, DNN_P500, BILSTM_P500,
    VANILLA_KAN_P500, RES_KAN_P500, RES_FCNN_P500,
)

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / 'models'
ROP_DIR    = ROOT / 'ROP测试数据'
ROP_DIR.mkdir(exist_ok=True)

ROP_LIST    = list(range(0, -16, -1))
LABEL_SCALE = 3.0
BATCH_SIZE  = 4096
CPU_WORKERS = None

ROP_PLOT_ROPS = [-9, -10, -11, -12, -13, -14, -15]


# ================= 模型构建函数 (CPU) =================
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


def _build_vanilla_kan(config):
    return build_vanilla_kan(config, 'cpu')


def _build_res_kan(config):
    return build_res_kan(config, 'cpu')


def _build_res_fcnn(config):
    return build_res_fcnn(config, 'cpu')


# ================= 模型注册表 =================
MODEL_REGISTRY = [
    {
        'name': 'FCNN',       'build': _build_fcnn,
        'default_cfg': dict(FCNN_P500),
        'ckpt': 'fcnn_p500_rop_model.pth',
        'color': 'C2', 'marker': '^', 'ls': '-.',
    },
    {
        'name': 'DNN',        'build': _build_dnn,
        'default_cfg': dict(DNN_P500),
        'ckpt': 'dnn_p500_rop_model.pth',
        'color': 'C4', 'marker': 'v', 'ls': '-.',
    },
    {
        'name': 'BiLSTM',     'build': _build_bilstm,
        'default_cfg': dict(BILSTM_P500),
        'ckpt': 'bilstm_p500_rop_model.pth',
        'color': 'C3', 'marker': 'D', 'ls': ':',
    },
    {
        'name': 'VanillaKAN', 'build': _build_vanilla_kan,
        'default_cfg': dict(VANILLA_KAN_P500),
        'ckpt': 'vanilla_kan_p500_rop_model.pth',
        'color': 'C9', 'marker': 'P', 'ls': '-',
    },
    {
        'name': 'ResKAN',     'build': _build_res_kan,
        'default_cfg': dict(RES_KAN_P500),
        'ckpt': 'res_kan_p500_rop_model.pth',
        'color': 'C0', 'marker': 'o', 'ls': '-',
    },
    {
        'name': 'ResFCNN',    'build': _build_res_fcnn,
        'default_cfg': dict(RES_FCNN_P500),
        'ckpt': 'res_fcnn_p500_rop_model.pth',
        'color': 'C1', 'marker': 's', 'ls': '--',
    },
]

_NAME_TO_ENTRY = {e['name']: e for e in MODEL_REGISTRY}
_FALLBACK_STYLE = [
    ('C5', 'x', '--'), ('tab:olive', '+', ':'), ('tab:cyan', '1', '-.'),
]


# ================= 增量 CSV =================
def load_existing_csv(csv_path):
    existing = OrderedDict()
    existing_cols = []
    if not csv_path.exists():
        return existing, existing_cols
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'ROP(dBm)' not in reader.fieldnames:
            return existing, existing_cols
        existing_cols = [c for c in reader.fieldnames if c != 'ROP(dBm)']
        for row in reader:
            raw = (row.get('ROP(dBm)') or '').strip()
            if not raw:
                continue
            rop = int(float(raw))
            existing[rop] = {}
            for col in existing_cols:
                val = row.get(col, '')
                if val and val.lower() != 'nan':
                    try:
                        existing[rop][col] = float(val)
                    except ValueError:
                        pass
    return existing, existing_cols


def save_merged_csv(csv_path, new_results, rop_tested):
    existing, existing_cols = load_existing_csv(csv_path)
    new_names = list(new_results.keys())
    all_names = list(OrderedDict.fromkeys(existing_cols + new_names))
    all_rops = sorted(set(list(existing.keys()) + rop_tested), reverse=True)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['ROP(dBm)'] + all_names)
        writer.writeheader()
        for rop in all_rops:
            row = {'ROP(dBm)': rop}
            old_row = existing.get(rop, {})
            for name in all_names:
                if name in new_results and rop in new_results[name]:
                    row[name] = new_results[name][rop]
                elif name in old_row:
                    row[name] = old_row[name]
                else:
                    row[name] = float('nan')
            writer.writerow(row)
    print(f"\nCSV 已保存 (增量合并): {csv_path}")
    return all_names, all_rops


# ================= 绘图 =================
def _entries_for_column_names(names):
    out = []
    for i, n in enumerate(names):
        if n in _NAME_TO_ENTRY:
            out.append(_NAME_TO_ENTRY[n])
        else:
            c, m, ls = _FALLBACK_STYLE[i % len(_FALLBACK_STYLE)]
            out.append({'name': n, 'color': c, 'marker': m, 'ls': ls})
    return out


def _resolve_rop_for_plot(rop_tested, valid_rops):
    valid = set(valid_rops)
    if ROP_PLOT_ROPS is None:
        return list(rop_tested)
    rop_for_plot = [r for r in ROP_PLOT_ROPS if r in valid]
    if not rop_for_plot:
        print("  [警告] ROP_PLOT_ROPS 与有效数据无交集，绘图改用全部 ROP。")
        return list(rop_tested)
    print(f"\n[INFO] 绘图仅使用 ROP = {rop_for_plot}（共 {len(rop_for_plot)} 点）")
    return rop_for_plot


def _render_rop_plot(csv_path, rop_for_plot, fig_path):
    existing, existing_cols = load_existing_csv(csv_path)
    if not existing_cols:
        print("[警告] CSV 为空，无法绘图。")
        return

    matplotlib.rcParams.update({
        'font.family':     'Times New Roman',
        'font.size':       14,
        'axes.titlesize':  14,
        'axes.labelsize':  14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 11,
        'axes.unicode_minus': False,
    })

    fig, ax = plt.subplots(figsize=(7, 6))
    entries = _entries_for_column_names(existing_cols)

    for entry in entries:
        name = entry['name']
        bers = [existing.get(rop, {}).get(name, float('nan'))
                for rop in rop_for_plot]
        if all(np.isnan(b) for b in bers):
            continue
        ax.semilogy(
            rop_for_plot, bers,
            marker=entry['marker'], linestyle=entry['ls'],
            color=entry['color'], linewidth=1.8, markersize=8,
            label=name,
        )

    ax.set_xlabel('Received Optical Power (dBm)')
    ax.set_ylabel('Bit Error Rate (BER)')
    ax.set_title('BER vs. ROP — ~500 Params Ablation')
    ax.set_xticks(rop_for_plot)
    ax.invert_xaxis()
    ax.legend(loc='upper right', framealpha=0.8)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {fig_path}")


# ================= 子进程工作函数 =================
def _worker(args):
    name, build_fn, config, ckpt_path_str, rop_data_paths, rx_mean, rx_std = args

    torch.set_num_threads(1)

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
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)

        preds_list, tgt_list = [], []
        with torch.inference_mode():
            for x, y in ld:
                x = x.unsqueeze(-1)
                preds_list.append(model(x))
                tgt_list.append(y)

        preds   = torch.cat(preds_list).flatten().numpy() * LABEL_SCALE
        targets = torch.cat(tgt_list).flatten().numpy()   * LABEL_SCALE
        pred_lbl = np.select(
            [preds   < -2, preds   < 0, preds   < 2], [-3, -1, 1], default=3)
        true_lbl = np.select(
            [targets < -2, targets < 0, targets < 2], [-3, -1, 1], default=3)
        model_results[rop] = float(np.mean(pred_lbl != true_lbl))

    return name, model_results


# ================= 主流程 =================
def main(plot_only=False, model_names=None):
    csv_path = ROP_DIR / 'rop_ber_comparison_p500.csv'
    fig_path = ROP_DIR / 'rop_ber_comparison_p500.png'

    if plot_only:
        print("=" * 65)
        print("  ROP p500 — 仅绘图（从 CSV 读取，不运行推理）")
        print("=" * 65)
        if not csv_path.exists():
            print(f"\n未找到 {csv_path}。")
            sys.exit(1)
        existing, _ = load_existing_csv(csv_path)
        if not existing:
            print("\nCSV 中没有有效数据。")
            sys.exit(1)
        all_rops = sorted(existing.keys(), reverse=True)
        rop_for_plot = _resolve_rop_for_plot(all_rops, set(all_rops))
        _render_rop_plot(csv_path, rop_for_plot, fig_path)
        return

    n_workers = CPU_WORKERS or os.cpu_count() or 1

    print("=" * 65)
    print("  ROP 泛化 — 500 参数量级消融测试 (多进程)")
    print(f"  逻辑核心数: {os.cpu_count()}  并行进程数: {n_workers}")
    if model_names:
        print(f"  指定模型: {model_names}")
    print("=" * 65)

    # ── 数据文件 ──
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

    rop_tested = sorted(rop_data_paths.keys(), reverse=True)

    # ── 筛选注册表 ──
    if model_names:
        candidates = [e for e in MODEL_REGISTRY if e['name'] in model_names]
    else:
        candidates = list(MODEL_REGISTRY)

    # ── 加载 checkpoint ──
    task_list = []
    available_entries = []
    for entry in candidates:
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

        tmp = entry['build'](cfg)
        n_params = sum(p.numel() for p in tmp.parameters())
        del tmp
        print(f"  [OK]   {entry['name']:12s} ({n_params:,} params)  "
              f"mean={rx_mean:.4f}, std={rx_std:.4f}")

        task_list.append((
            entry['name'], entry['build'], cfg,
            str(ckpt_path), rop_data_paths, rx_mean, rx_std,
        ))
        available_entries.append(entry)

    if not task_list:
        print("\n没有找到任何 p500 模型 checkpoint。")
        sys.exit(1)

    # ── 多进程推理 ──
    actual_workers = min(n_workers, len(task_list))
    print(f"\n[INFO] 启动 {actual_workers} 个子进程 "
          f"（共 {len(task_list)} 个模型）...\n")

    results = {}
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        future_to_name = {
            executor.submit(_worker, args): args[0] for args in task_list
        }
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

    # ── 增量合并 CSV ──
    save_merged_csv(csv_path, results, rop_tested)

    # ── 绘图 ──
    existing_data, _ = load_existing_csv(csv_path)
    all_rops = sorted(existing_data.keys(), reverse=True)
    rop_for_plot = _resolve_rop_for_plot(
        all_rops, set(rop_data_paths.keys()) | set(all_rops))
    _render_rop_plot(csv_path, rop_for_plot, fig_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ROP 泛化 — 500 参数量级消融测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例:\n'
               '  python test_rop_p500.py                             # 全部\n'
               '  python test_rop_p500.py --models ResKAN ResFCNN     # 指定\n'
               '  python test_rop_p500.py --list                      # 列出\n'
               '  python test_rop_p500.py --plot-only                 # 仅绘图',
    )
    parser.add_argument('--models', nargs='+', metavar='NAME',
                        help='仅测试指定 name 的模型 (可多个)')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用模型名后退出')
    parser.add_argument('--plot-only', action='store_true',
                        help='不运行推理，仅根据已有 CSV 重新出图')
    args = parser.parse_args()

    if args.list:
        print("可用模型 name:")
        for e in MODEL_REGISTRY:
            ckpt_path = MODELS_DIR / e['ckpt']
            status = "✓" if ckpt_path.exists() else "✗"
            print(f"  {status}  {e['name']:12s}  ← {e['ckpt']}")
        sys.exit(0)

    names = None
    if args.models:
        valid_names = {e['name'] for e in MODEL_REGISTRY}
        names = [n for n in args.models if n in valid_names]
        invalid = [n for n in args.models if n not in valid_names]
        if invalid:
            print(f"[警告] 未知模型名: {invalid}")
            print(f"       可用: {sorted(valid_names)}")
        if not names:
            sys.exit(1)

    main(plot_only=args.plot_only, model_names=names)
