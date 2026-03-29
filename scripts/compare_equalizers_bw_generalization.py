"""
多带宽泛化对比：在「泛化测试数据」目录下各 dataset_for_python_bw*.mat 上
评估已训练均衡器在不同带宽条件下的 BER（与 compare_all_equalizers 相同的 SNR 扫描），
结果写入单一 CSV，并按带宽分组绘制 BER vs SNR（semilogy）。

前置：在 MATLAB 中运行 RX_generalization.m，生成各带宽对应的 .mat。

仅绘图不重测：
  python compare_equalizers_bw_generalization.py --plot-only

绘图前会弹出窗口勾选曲线；10 秒内未点「确定」则绘制全部。无界面请加 --no-gui。
"""

import argparse
import re
import sys
import torch
import scipy.io
import numpy as np
import logging
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
# 注意：不在此处 import pyplot，避免先于 Tk 初始化 Matplotlib 后端导致 Windows 上弹窗不显示。
# pyplot 仅在弹窗结束后再在 plot_generalization_results 内导入。

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
    build_kan_fcnn,
    KAN_FCNN_CONFIG,
    build_hybrid_kan,
    HYBRID_KAN_CONFIG,
    build_res_kan,
    RES_KAN_CONFIG,
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
GEN_DIR = ROOT / "泛化测试数据"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
GEN_DIR.mkdir(exist_ok=True)

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

DEFAULT_CSV_NAME = "equalizer_bw_generalization_results.csv"
DEFAULT_FIG_NAME = "equalizer_bw_generalization_ber.png"

# 开关：设为 True 时，直接运行脚本将只根据已有 CSV 绘图、不重测（等价于命令行 --plot-only）
PLOT_ONLY_NO_RETEST = False

# SNR 轴：无噪声用略大于 25 的刻度位置，便于与 dB 点分开
SNR_X_NUMERIC = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

# semilogy 无法显示 BER=0，绘图时将所有 0 映射到该纵坐标
BER_ZERO_AS_PLOT = 1e-6

# 绘图前弹窗：未在时限内点「确定」则绘制全部模型
PLOT_MODEL_PICK_TIMEOUT_SEC = 10


def select_models_via_dialog(model_names, timeout_sec=PLOT_MODEL_PICK_TIMEOUT_SEC):
    """
    弹出勾选框选择要绘制的模型名（顺序与 CSV 列一致）。
    点「确定」：仅绘制已勾选；若未勾选任何一项则提示且保持窗口。
    超时未点「确定」：视为全部勾选。
    关闭窗口：与超时相同，绘制全部。
    无 GUI 环境或 tkinter 不可用时向 stderr 说明原因并退回全部模型名。
    """
    if not model_names:
        return []
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception as exc:
        print(
            "[compare_bw_gen] 无法加载 tkinter，已跳过勾选弹窗（将绘制全部曲线）。"
            f"原因: {exc}\n"
            "  若需弹窗，请使用带 Tcl/Tk 的 Python 安装，或运行: python -m tkinter",
            file=sys.stderr,
        )
        return list(model_names)

    state = {"timeout_id": None, "picked": None, "finished": False}

    try:
        root = tk.Tk()
    except Exception as exc:
        print(
            "[compare_bw_gen] 无法创建 Tk 窗口，已跳过勾选弹窗（将绘制全部曲线）。"
            f"原因: {exc}",
            file=sys.stderr,
        )
        return list(model_names)

    root.title("选择要绘制的均衡器")
    root.resizable(True, True)
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass

    vars_by_name = {n: tk.BooleanVar(value=True) for n in model_names}

    outer = tk.Frame(root, padx=12, pady=10)
    outer.pack(fill=tk.BOTH, expand=True)

    tk.Label(
        outer,
        text=(
            f"请勾选要显示的曲线（{timeout_sec} 秒内未点「确定」将自动绘制全部）"
        ),
        justify=tk.LEFT,
    ).pack(anchor=tk.W)

    cb_frame = tk.Frame(outer)
    cb_frame.pack(fill=tk.BOTH, expand=True, pady=6)
    for n in model_names:
        tk.Checkbutton(cb_frame, text=n, variable=vars_by_name[n]).pack(anchor=tk.W)

    def finish_with_all():
        if state["finished"]:
            return
        state["finished"] = True
        state["picked"] = list(model_names)
        if state["timeout_id"] is not None:
            try:
                root.after_cancel(state["timeout_id"])
            except tk.TclError:
                pass
        root.destroy()

    def on_ok():
        picked = [n for n in model_names if vars_by_name[n].get()]
        if not picked:
            messagebox.showwarning(
                "提示",
                "请至少勾选一条曲线；或关闭窗口 / 等待倒计时结束以绘制全部。",
            )
            return
        if state["finished"]:
            return
        state["finished"] = True
        state["picked"] = picked
        if state["timeout_id"] is not None:
            try:
                root.after_cancel(state["timeout_id"])
            except tk.TclError:
                pass
        root.destroy()

    btn_row = tk.Frame(outer)
    btn_row.pack(fill=tk.X, pady=4)
    tk.Button(btn_row, text="确定", command=on_ok, width=12).pack(side=tk.LEFT)

    state["timeout_id"] = root.after(timeout_sec * 1000, finish_with_all)

    def on_user_close():
        finish_with_all()

    root.protocol("WM_DELETE_WINDOW", on_user_close)

    try:
        root.deiconify()
        root.update_idletasks()
        root.update()
        root.lift()
        root.focus_force()
    except tk.TclError:
        pass

    root.mainloop()

    if state["picked"] is None:
        print(
            "[compare_bw_gen] 警告：弹窗异常结束（未收到选择），将绘制全部曲线。"
            " 若从未看到窗口，请尝试在系统终端（非远程 SSH）中运行，并确认已安装 tkinter。",
            file=sys.stderr,
        )
        return list(model_names)
    return state["picked"]


def ber_for_semilogy_plot(y):
    """将 BER 数组转为可画 semilogy 的值：有限且为 0 的点统一画在 BER_ZERO_AS_PLOT。"""
    a = np.asarray(y, dtype=np.float64).copy()
    a[np.isfinite(a) & (a == 0)] = BER_ZERO_AS_PLOT
    return a

SNR_LIST = [0, 5, 10, 15, 20, 25, None]
MAX_WORKERS = 4
LABEL_SCALE = 3.0
BATCH_SIZE = 1024

BW_MAT_PATTERN = re.compile(r"dataset_for_python_bw([\d.]+)\.mat$", re.I)


def setup_logger():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"compare_bw_gen_{timestamp}.log"
    log = logging.getLogger("compare_bw_gen")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    log.addHandler(logging.StreamHandler())
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for h in log.handlers:
        h.setFormatter(fmt)
    return log, log_path


def snr_label(snr):
    return "无噪声" if snr is None else f"{snr} dB"


def match_snr_cell(cell, snr_ref):
    """判断 CSV 中 SNR 列是否对应 SNR_LIST 中的 snr_ref（兼容「无噪声」/ No / 科学计数误写等）。"""
    c = str(cell).strip()
    if snr_ref is None:
        cl = c.lower()
        return c in ("无噪声",) or cl in ("no", "none", "noiseless", "inf")
    lab = snr_label(snr_ref)
    return c == lab or c.replace(" ", "") == lab.replace(" ", "")


def _parse_ber_cell(val):
    if val is None or str(val).strip() == "":
        return float("nan")
    s = str(val).strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_generalization_csv_rows(csv_path):
    """读取泛化结果 CSV，返回 (model_columns, rows 列表)。"""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "bandwidth_GHz" not in fieldnames or "SNR" not in fieldnames:
        raise ValueError(f"CSV 格式无效（缺少 bandwidth_GHz 或 SNR）: {csv_path}")
    model_cols = [c for c in fieldnames if c not in ("bandwidth_GHz", "SNR")]
    return model_cols, rows


def build_ber_matrix_for_bw(rows_one_bw, model_cols):
    """对单一 bandwidth 的多行记录，按 SNR_LIST 顺序拼 BER 矩阵 (len(SNR_LIST), n_models)。"""
    mat = np.full((len(SNR_LIST), len(model_cols)), np.nan, dtype=np.float64)
    for i, snr in enumerate(SNR_LIST):
        row = next((r for r in rows_one_bw if match_snr_cell(r["SNR"], snr)), None)
        if row is None:
            continue
        for j, m in enumerate(model_cols):
            mat[i, j] = _parse_ber_cell(row.get(m))
    return mat


def plot_generalization_results(csv_path, out_fig_path=None, pick_models=True):
    """
    按带宽分组子图：x 为 SNR（无噪声在 30 GHz 刻度位置）, y 为 BER，semilogy；
    每个子图内多条曲线对应各均衡器。

    pick_models=True 时先弹出窗口勾选要画的模型；False 时画 CSV 中全部模型列（适合无界面环境）。
    """
    model_cols, rows = load_generalization_csv_rows(csv_path)
    if not model_cols:
        raise ValueError("CSV 中未找到模型列")

    if pick_models:
        selected = select_models_via_dialog(model_cols)
    else:
        selected = list(model_cols)
    if not selected:
        raise ValueError("没有可绘制的模型列")

    import matplotlib.pyplot as plt

    by_bw = defaultdict(list)
    for row in rows:
        try:
            bw = float(row["bandwidth_GHz"])
        except (TypeError, ValueError):
            continue
        by_bw[bw].append(row)

    bws = sorted(by_bw.keys())
    if not bws:
        raise ValueError("CSV 中没有有效的 bandwidth_GHz 数据")

    n = len(bws)
    ncols = 2 if n > 2 else 1
    nrows = (n + ncols - 1) // ncols
    fig_w = 12 if ncols > 1 else 10
    fig_h = max(3.2 * nrows, 4.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.flatten()

    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(model_cols), 2)))
    col_index = {name: j for j, name in enumerate(model_cols)}

    for idx, bw in enumerate(bws):
        ax = axes_flat[idx]
        mat = build_ber_matrix_for_bw(by_bw[bw], model_cols)
        for name in selected:
            j = col_index[name]
            y = mat[:, j]
            if np.all(np.isnan(y)):
                continue
            ax.semilogy(
                SNR_X_NUMERIC,
                ber_for_semilogy_plot(y),
                marker="o",
                linewidth=1.8,
                markersize=5,
                color=cmap[j % len(cmap)],
                label=name,
            )
        ax.set_title(f"带宽 = {bw:g} GHz", fontsize=11)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.set_xticks(SNR_X_NUMERIC)
        ax.set_xticklabels(["0", "5", "10", "15", "20", "25", "无噪声"])
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right", ncol=2)

    for k in range(len(bws), len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle("多带宽泛化：各均衡器 BER vs SNR", fontsize=13, y=1.02)
    plt.tight_layout()
    if out_fig_path is None:
        out_fig_path = GEN_DIR / DEFAULT_FIG_NAME
    out_fig_path = Path(out_fig_path)
    out_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    return out_fig_path


def add_awgn(rx_signal, snr_db):
    if snr_db is None or np.isinf(snr_db):
        return rx_signal.copy()
    rx = np.asarray(rx_signal, dtype=np.float64)
    ps = np.mean(rx ** 2)
    pn = ps / (10 ** (snr_db / 10))
    sigma = np.sqrt(pn)
    noise = np.random.RandomState(seed=42).randn(*rx.shape).astype(np.float64) * sigma
    return (rx + noise).astype(np.float64)


def calculate_ber(pred_scaled, true_scaled):
    pred_labels = np.select(
        [
            pred_scaled < -2,
            (pred_scaled >= -2) & (pred_scaled < 0),
            (pred_scaled >= 0) & (pred_scaled < 2),
        ],
        [-3, -1, 1],
        default=3,
    )
    errors = np.sum(pred_labels != true_scaled)
    return errors / len(true_scaled)


def run_inference(model, loader, device):
    preds_list, targets_list = [], []
    model.eval()
    with torch.no_grad():
        for inputs, tgt in loader:
            inputs = inputs.unsqueeze(-1).to(device)
            out = model(inputs).cpu().numpy()
            preds_list.append(out)
            targets_list.append(tgt.numpy())
    preds = np.concatenate(preds_list).flatten() * LABEL_SCALE
    targets = np.concatenate(targets_list).flatten() * LABEL_SCALE
    return preds, targets


def build_fcnn(config, device):
    return FCNNEqualizer(
        input_dim=config["window_size"],
        hidden_dims=config["hidden_dims"],
    ).to(device)


def build_dnn(config, device):
    return DNNEqualizer(
        input_dim=config["window_size"],
        hidden_dims=config["hidden_dims"],
        use_batchnorm=config["use_batchnorm"],
        dropout=config["dropout"],
    ).to(device)


def build_bilstm(config, device):
    return BiLSTMEqualizer(
        input_size=1,
        hidden_size=config["hidden_size"],
        num_layers=config["num_lstm_layers"],
        use_center=config["use_center"],
        window_size=config["window_size"],
    ).to(device)


MODEL_REGISTRY = [
    {
        "name": "FCNN",
        "tag": "FCNN",
        "ckpt": "fcnn_model.pth",
        "build_fn": build_fcnn,
        "default_config": dict(FCNN_CONFIG),
        "group": "baseline",
    },
    {
        "name": "DNN",
        "tag": "DNN",
        "ckpt": "dnn_model.pth",
        "build_fn": build_dnn,
        "default_config": dict(DNN_CONFIG),
        "group": "baseline",
    },
    {
        "name": "BiLSTM",
        "tag": "BiLSTM",
        "ckpt": "bilstm_model.pth",
        "build_fn": build_bilstm,
        "default_config": dict(BILSTM_CONFIG),
        "group": "baseline",
    },
    {
        "name": "KAN-FCNN",
        "tag": "KANFCNN",
        "ckpt": "kan_fcnn_model.pth",
        "build_fn": build_kan_fcnn,
        "default_config": dict(KAN_FCNN_CONFIG),
        "group": "kan",
    },
    {
        "name": "Hybrid-KAN",
        "tag": "HybKAN",
        "ckpt": "hybrid_kan_model.pth",
        "build_fn": build_hybrid_kan,
        "default_config": dict(HYBRID_KAN_CONFIG),
        "group": "kan",
    },
    {
        "name": "ResKAN",
        "tag": "ResKAN",
        "ckpt": "res_kan_model.pth",
        "build_fn": build_res_kan,
        "default_config": dict(RES_KAN_CONFIG),
        "group": "kan",
    },
]


def run_single(model_entry, snr_db, rx_test_base, symb_test, rx_mean, rx_std):
    config = model_entry["default_config"]
    device = config["device"]
    ckpt_path = MODELS_DIR / model_entry["ckpt"]

    rx_test = add_awgn(rx_test_base, snr_db)
    dataset = OpticalDataset(
        rx_test,
        symb_test,
        config["window_size"],
        config["sps"],
        rx_mean=rx_mean,
        rx_std=rx_std,
        label_scale=LABEL_SCALE,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = model_entry["build_fn"](config, device)
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    preds, targets = run_inference(model, loader, device)
    ber = calculate_ber(preds, targets)
    return model_entry["tag"], snr_db, ber


def discover_mat_files():
    if not GEN_DIR.is_dir():
        return []
    mats = []
    for p in sorted(GEN_DIR.glob("dataset_for_python_bw*.mat")):
        m = BW_MAT_PATTERN.search(p.name)
        bw = float(m.group(1)) if m else None
        mats.append((bw, p))
    mats.sort(key=lambda x: (x[0] is None, x[0] or 0.0))
    return mats


def load_test_vectors(mat_path):
    data = scipy.io.loadmat(str(mat_path))
    rx_test_base = data["rx_test_export"].flatten()
    if np.iscomplexobj(rx_test_base):
        rx_test_base = np.abs(rx_test_base).astype(np.float64)
    else:
        rx_test_base = rx_test_base.astype(np.float64)
    symb_test = data["symb_test_export"].flatten()
    bw = None
    if "bandwidth_ghz" in data:
        bw = float(np.squeeze(data["bandwidth_ghz"]))
    return rx_test_base, symb_test, bw


def main():
    parser = argparse.ArgumentParser(
        description="多带宽泛化 BER 对比；--plot-only 时仅根据 CSV 绘图，不重新推理。"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="关闭重测：只读取已有 CSV 并绘制 BER–SNR 曲线（适合快速改图）",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=f"结果 CSV 路径，默认: {GEN_DIR / DEFAULT_CSV_NAME}",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="不保存 PNG 图像（仅重测模式默认可保存图）",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="绘图前不弹窗，直接绘制 CSV 中全部模型（服务器/无界面环境使用）",
    )
    args = parser.parse_args()
    if PLOT_ONLY_NO_RETEST:
        args.plot_only = True

    csv_path = args.csv if args.csv is not None else (GEN_DIR / DEFAULT_CSV_NAME)

    if args.plot_only:
        log, log_path = setup_logger()
        if not csv_path.is_file():
            log.info(f"[ERROR] 未找到 CSV: {csv_path}")
            log.info("        请先完整跑一次本脚本生成结果，或检查 --csv 路径。")
            return
        log.info(f"[INFO] --plot-only：跳过重测，从 CSV 绘图: {csv_path}")
        if not args.no_figure:
            out_fig = plot_generalization_results(
                csv_path, pick_models=not args.no_gui
            )
            log.info(f"[INFO] 图像已保存: {out_fig}")
        log.info(f"[INFO] 日志: {log_path}")
        return

    log, log_path = setup_logger()
    log.info("=" * 72)
    log.info("  多带宽泛化 BER 对比（各 bw 的 dataset_for_python_bw*.mat）")
    log.info("=" * 72)

    _device = get_device()
    log.info(f"[INFO] 运行设备: {_device}")

    mat_entries = discover_mat_files()
    if not mat_entries:
        log.info(f"[ERROR] 在 {GEN_DIR} 未找到 dataset_for_python_bw*.mat。")
        log.info("        请先在 MATLAB 中运行 RX_generalization.m 生成数据。")
        return

    for _, p in mat_entries:
        log.info(f"  [数据] {p.name}")

    available = []
    for entry in MODEL_REGISTRY:
        ckpt_path = MODELS_DIR / entry["ckpt"]
        if ckpt_path.exists():
            available.append(entry)
            log.info(f"  [OK] {entry['name']:15s}  ← {entry['ckpt']}")
        else:
            log.info(f"  [--] {entry['name']:15s}  ← 未找到 {entry['ckpt']}，跳过")

    if not available:
        log.info("\n[ERROR] 没有找到任何模型 checkpoint。")
        return

    model_meta = {}
    for entry in available:
        ckpt = torch.load(
            str(MODELS_DIR / entry["ckpt"]), weights_only=False, map_location="cpu"
        )
        rx_mean = float(ckpt["rx_mean"])
        rx_std = float(ckpt["rx_std"])
        saved_config = ckpt.get("config", None)
        if saved_config is not None:
            entry["default_config"].update(saved_config)
        entry["default_config"]["device"] = _device
        model_meta[entry["tag"]] = {"rx_mean": rx_mean, "rx_std": rx_std}

    # 预加载各带宽测试波形（避免多线程重复读盘）
    bw_cases = []
    for bw_key, mat_path in mat_entries:
        rx_base, symb_t, bw_mat = load_test_vectors(mat_path)
        bw = bw_mat if bw_mat is not None else bw_key
        if bw is None:
            m = BW_MAT_PATTERN.search(mat_path.name)
            bw = float(m.group(1)) if m else float("nan")
        bw_cases.append(
            {
                "bandwidth_ghz": bw,
                "mat_path": mat_path,
                "rx_test_base": rx_base,
                "symb_test": symb_t,
            }
        )
        log.info(
            f"[INFO] 已加载 {mat_path.name}: bw={bw} GHz, "
            f"len(rx)={len(rx_base):,}, len(symb)={len(symb_t):,}"
        )

    results = {e["tag"]: {} for e in available}
    for e in available:
        for c in bw_cases:
            results[e["tag"]][c["bandwidth_ghz"]] = {}

    log.info(f"\n[INFO] SNR: {[snr_label(s) for s in SNR_LIST]}")
    log.info(f"[INFO] 并行线程: {MAX_WORKERS}\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut_map = {}
        for case in bw_cases:
            bw = case["bandwidth_ghz"]
            rx_base = case["rx_test_base"]
            symb_t = case["symb_test"]
            for entry in available:
                meta = model_meta[entry["tag"]]
                for snr in SNR_LIST:
                    ft = ex.submit(
                        run_single,
                        entry,
                        snr,
                        rx_base,
                        symb_t,
                        meta["rx_mean"],
                        meta["rx_std"],
                    )
                    fut_map[ft] = (entry["tag"], bw, snr)

        for fut in as_completed(fut_map):
            tag, bw, snr = fut_map[fut]
            try:
                _, _, ber = fut.result()
                results[tag][bw][snr] = ber
                log.info(
                    f"  [{tag:8s}] bw={bw:g} GHz  SNR={snr_label(snr):8s}  BER={ber:.4e}"
                )
            except Exception as exc:
                log.info(
                    f"  [{tag:8s}] bw={bw:g} GHz  SNR={snr_label(snr):8s}  错误: {exc}"
                )

    names = [e["name"] for e in available]
    csv_path = GEN_DIR / "equalizer_bw_generalization_results.csv"
    fieldnames = ["bandwidth_GHz", "SNR"] + names

    csv_rows = []
    for case in sorted(bw_cases, key=lambda x: x["bandwidth_ghz"]):
        bw = case["bandwidth_ghz"]
        for snr in SNR_LIST:
            row = {"bandwidth_GHz": bw, "SNR": snr_label(snr)}
            for entry in available:
                ber = results[entry["tag"]][bw].get(snr, float("nan"))
                row[entry["name"]] = ber
            csv_rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    log.info(f"\n[INFO] 结果已写入: {csv_path}")
    if not args.no_figure:
        out_fig = plot_generalization_results(
            csv_path, pick_models=not args.no_gui
        )
        log.info(f"[INFO] BER–SNR 图已保存: {out_fig}")
    log.info(f"[INFO] 日志: {log_path}")


if __name__ == "__main__":
    main()
