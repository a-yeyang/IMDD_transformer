"""
加载 sweep 已训练 checkpoint，在指定 SNR 下仅测试（CPU），不写回训练。

SNR: 25 / 20 / 15 / 10 dB
模型: KAN 三系 + FCNN / DNN / BiLSTM（与 run_kan_sweep / run_baseline_sweep 一致）

输出:
  muti_test/multi_snr_test_results.csv
  muti_test/*.png

用法:
  python scripts/test_sweep_checkpoints_muti_snr.py
  python scripts/test_sweep_checkpoints_muti_snr.py --plot-only
      # 仅读取 muti_test/multi_snr_test_results.csv 重绘 PNG，不跑测试、不写 CSV
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "muti_test"
CKPT_DIR = ROOT / "sweep_results" / "checkpoints"
SCRIPTS = Path(__file__).resolve().parent
CSV_NAME = "multi_snr_test_results.csv"
sys.path.insert(0, str(SCRIPTS))

from run_baseline_sweep import (  # noqa: E402
    BASELINE_MODELS,
    tune_by_building,
)
from run_kan_sweep import (  # noqa: E402
    COMBOS,
    MODEL_KEYS,
    MODELS,
    PARAM_SCALES,
    WINDOW_SIZE,
    build_sweep_model,
)
from train_fcnn import OpticalDataset  # noqa: E402

import scipy.io
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
SPS = 2
LABEL_SCALE = 3.0
SNR_LIST = [25, 20, 15, 10]
DEVICE = "cpu"
BER_FLOOR = 1e-7


def configure_matplotlib_fonts() -> None:
    """
    避免「Times New Roman 不含汉字」导致的缺字警告：无衬线字体列表把 CJK 字体放首位，
    由 matplotlib 按字符回退；英文与数字仍正常显示。
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            # 勿把 Times New Roman 放在首位，否则中文会触发 missing glyph 警告
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
                "Arial",
            ],
            "axes.unicode_minus": False,
        }
    )


def snr_tag(snr: int) -> str:
    return str(snr)


def add_awgn(rx: np.ndarray, snr_db: float) -> np.ndarray:
    ps = np.mean(rx.astype(np.float64) ** 2)
    sigma = np.sqrt(ps / 10 ** (snr_db / 10))
    rng = np.random.RandomState(42)
    return rx + rng.randn(*rx.shape) * sigma


def compute_ber(pred: np.ndarray, true: np.ndarray) -> float:
    def _labels(x: np.ndarray) -> np.ndarray:
        return np.select(
            [x < -2, (x >= -2) & (x < 0), (x >= 0) & (x < 2)],
            [-3, -1, 1],
            default=3,
        )

    return float(np.mean(_labels(pred) != _labels(true)))


def load_mat():
    data = scipy.io.loadmat(str(ROOT / "dataset_for_python.mat"))
    rxe = data["rx_test_export"].flatten()
    if np.iscomplexobj(rxe):
        rxe = np.abs(rxe)
    se = data["symb_test_export"].flatten()
    rx_mean = float(np.mean(rxe))
    rx_std = float(np.std(rxe))
    return rxe, se, rx_mean, rx_std


def load_state(model: torch.nn.Module, path: Path) -> None:
    try:
        state = torch.load(str(path), map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(str(path), map_location=DEVICE)
    model.load_state_dict(state, strict=True)


def test_one_snr(
    model: torch.nn.Module,
    rx_base: np.ndarray,
    symb: np.ndarray,
    rx_mean: float,
    rx_std: float,
    snr_db: float,
) -> float:
    model.eval()
    rx = add_awgn(rx_base, snr_db)
    ds = OpticalDataset(rx, symb, WINDOW_SIZE, SPS, rx_mean, rx_std, LABEL_SCALE)
    ld = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    preds, trues = [], []
    with torch.inference_mode():
        for x, y in ld:
            preds.append(model(x.unsqueeze(-1).to(DEVICE)))
            trues.append(y)
    pr = torch.cat(preds).flatten().cpu().numpy() * LABEL_SCALE
    tr = torch.cat(trues).flatten().numpy() * LABEL_SCALE
    return compute_ber(pr, tr)


def run_kan_case(mk: str, ps: int, combo: dict) -> dict | None:
    cid, G, k = combo["id"], combo["G"], combo["k"]
    ckpt = CKPT_DIR / f"sweep_{mk}_{ps}_{cid}.pth"
    if not ckpt.is_file():
        logging.warning("缺少 checkpoint: %s", ckpt)
        return None
    model, nparams, cfg = build_sweep_model(mk, ps, G, k, DEVICE)
    load_state(model, ckpt)
    return {
        "family": "KAN",
        "model_key": mk,
        "model": MODELS[mk]["name"],
        "param_scale": ps,
        "combo": cid,
        "grid_size": G,
        "spline_order": k,
        "hidden_config": cfg,
        "actual_params": nparams,
        "_model": model,
    }


def run_baseline_case(entry: dict, ps: int, combo: dict) -> dict | None:
    mk = entry["key"]
    cid = combo["id"]
    ckpt = CKPT_DIR / f"sweep_baseline_{mk}_{ps}_{cid}.pth"
    if not ckpt.is_file():
        logging.warning("缺少 checkpoint: %s", ckpt)
        return None
    best_h, actual_p = tune_by_building(combo["build"], ps)
    model = combo["build"](best_h).to(DEVICE)
    load_state(model, ckpt)
    cfg_str = combo["cfg"](best_h)
    return {
        "family": "baseline",
        "model_key": mk,
        "model": entry["name"],
        "param_scale": ps,
        "combo": cid,
        "grid_size": "",
        "spline_order": "",
        "hidden_config": cfg_str,
        "actual_params": actual_p,
        "_model": model,
    }


def fill_ber_row(meta: dict, rx_base, symb, rx_mean, rx_std) -> dict:
    m = meta["_model"]
    row = {k: v for k, v in meta.items() if k != "_model"}
    for snr in SNR_LIST:
        row[f"ber_{snr_tag(snr)}dB"] = test_one_snr(m, rx_base, symb, rx_mean, rx_std, snr)
    return row


def plot_histograms(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, snr in zip(axes.flat, SNR_LIST):
        key = f"ber_{snr_tag(snr)}dB"
        vals = [max(r[key], BER_FLOOR) for r in rows]
        logv = np.log10(np.array(vals))
        ax.hist(logv, bins=24, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(f"SNR = {snr} dB  (N={len(vals)})")
        ax.set_xlabel(r"$\log_{10}$(BER)")
        ax.set_ylabel("配置数量")
        ax.grid(True, axis="y", ls="--", alpha=0.35)
    fig.suptitle("各 SNR 下全部模型×参数量级×超参组合的 BER 分布（直方图）", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)


def plot_block_heatmaps(rows: list[dict], out: Path) -> None:
    """行=SNR，列=模型，子图内为 param_scale × combo 的 BER 方块图。"""
    kan_order = [MODELS[k]["name"] for k in MODEL_KEYS]
    base_order = [e["name"] for e in BASELINE_MODELS]
    model_order = base_order + kan_order

    def mat_for_model(name: str, snr: int) -> np.ndarray:
        key = f"ber_{snr_tag(snr)}dB"
        data = np.full((len(PARAM_SCALES), len(COMBOS)), np.nan)
        for i, ps in enumerate(PARAM_SCALES):
            for j, c in enumerate(COMBOS):
                cid = c["id"]
                for r in rows:
                    if r["model"] == name and r["param_scale"] == ps and r["combo"] == cid:
                        data[i, j] = max(r[key], BER_FLOOR)
                        break
        return data

    # 全图统一对数色标
    all_flat = []
    for snr in SNR_LIST:
        for name in model_order:
            d = mat_for_model(name, snr)
            all_flat.append(d[np.isfinite(d) & (d > 0)])
    if all_flat:
        merged = np.concatenate(all_flat)
        g_vmin = max(float(np.min(merged)), BER_FLOOR)
        g_vmax = max(float(np.max(merged)), g_vmin * 10)
    else:
        g_vmin, g_vmax = BER_FLOOR, 1.0
    norm = LogNorm(vmin=g_vmin, vmax=g_vmax)

    fig, axes = plt.subplots(len(SNR_LIST), len(model_order), figsize=(22, 14))
    combo_lbl = [f"{c['id']}\n(G={c['G']},k={c['k']})" for c in COMBOS]
    last_im = None

    for si, snr in enumerate(SNR_LIST):
        for mi, name in enumerate(model_order):
            ax = axes[si, mi]
            data = mat_for_model(name, snr)
            data_plot = np.where(np.isfinite(data), data, BER_FLOOR)
            last_im = ax.imshow(data_plot, cmap="RdYlGn_r", aspect="auto", norm=norm)
            ax.set_xticks(range(len(COMBOS)))
            if si == len(SNR_LIST) - 1:
                ax.set_xticklabels(combo_lbl, fontsize=6)
            else:
                ax.set_xticklabels([])
            ax.set_yticks(range(len(PARAM_SCALES)))
            if mi == 0:
                ax.set_yticklabels(PARAM_SCALES)
            else:
                ax.set_yticklabels([])
            if si == 0:
                ax.set_title(name, fontsize=10)
            if mi == 0:
                ax.set_ylabel(f"SNR {snr} dB\n目标参数量", fontsize=9)
            med = np.nanmedian(data_plot)
            for i in range(data_plot.shape[0]):
                for j in range(data_plot.shape[1]):
                    if np.isfinite(data[i, j]):
                        v = data_plot[i, j]
                        c = "white" if v > med else "black"
                        ax.text(
                            j,
                            i,
                            f"{v:.1e}",
                            ha="center",
                            va="center",
                            fontsize=5,
                            color=c,
                        )

    fig.suptitle(
        "各信噪比 × 模型：参数量级 × 超参组合的 BER 方块图（共用色标；颜色越深越好）",
        fontsize=14,
        y=1.01,
    )
    fig.subplots_adjust(right=0.92)
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.65, pad=0.02)
    cbar.set_label("BER")
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bars(rows: list[dict], out: Path) -> None:
    """各 SNR：横轴为模型，分组为四档目标参数量级（Combo A）。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    model_order = [e["name"] for e in BASELINE_MODELS] + [MODELS[k]["name"] for k in MODEL_KEYS]
    n_mod = len(model_order)
    bw = 0.18

    for ax, snr in zip(axes.flat, SNR_LIST):
        key = f"ber_{snr_tag(snr)}dB"
        x = np.arange(n_mod)
        for pi, ps in enumerate(PARAM_SCALES):
            offs = (pi - 1.5) * bw
            hs = []
            for name in model_order:
                v = BER_FLOOR
                for r in rows:
                    if (
                        r["param_scale"] == ps
                        and r["model"] == name
                        and r["combo"] == "A"
                    ):
                        v = max(r[key], BER_FLOOR)
                        break
                hs.append(v)
            ax.bar(x + offs, hs, bw * 0.9, label=f"~{ps} params")
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("BER")
        ax.set_yscale("log")
        ax.set_title(f"SNR = {snr} dB（Combo A，按参数量级分组）")
        ax.legend(fontsize=7, ncol=4)
        ax.grid(True, axis="y", ls="--", alpha=0.3)

    fig.suptitle("各 SNR 下模型误码率柱状图（默认 Combo A，四档参数量级）", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)


def save_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "family",
        "model_key",
        "model",
        "param_scale",
        "combo",
        "grid_size",
        "spline_order",
        "hidden_config",
        "actual_params",
    ] + [f"ber_{snr_tag(s)}dB" for s in SNR_LIST]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def load_results_csv(path: Path) -> list[dict]:
    """从 save_csv 写出的文件读回，供 --plot-only 重绘。"""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r: dict = dict(row)
            r["param_scale"] = int(r["param_scale"])
            r["actual_params"] = int(r["actual_params"])
            gs = str(r.get("grid_size", "")).strip()
            r["grid_size"] = int(gs) if gs else ""
            so = str(r.get("spline_order", "")).strip()
            r["spline_order"] = int(so) if so else ""
            for snr in SNR_LIST:
                k = f"ber_{snr_tag(snr)}dB"
                r[k] = float(r[k])
            rows.append(r)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多 SNR 均衡器测试（CPU）或仅根据 CSV 重绘图表",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="关闭测试：只读取 muti_test/multi_snr_test_results.csv 并输出 PNG，不加载模型、不计算 BER",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    configure_matplotlib_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / CSV_NAME
    rows_out: list[dict]

    if args.plot_only:
        if not csv_path.is_file():
            logging.error("未找到 %s，无法使用 --plot-only", csv_path)
            sys.exit(1)
        rows_out = load_results_csv(csv_path)
        logging.info("--plot-only: 已从 CSV 加载 %d 条记录", len(rows_out))
    else:
        torch.set_num_threads(8)

        rx_base, symb, rx_mean, rx_std = load_mat()
        logging.info("Test samples: %s, CPU threads: 8", len(symb))

        rows_out = []

        for ps in PARAM_SCALES:
            for combo in COMBOS:
                for mk in MODEL_KEYS:
                    meta = run_kan_case(mk, ps, combo)
                    if meta is None:
                        continue
                    rows_out.append(fill_ber_row(meta, rx_base, symb, rx_mean, rx_std))

        for ps in PARAM_SCALES:
            for entry in BASELINE_MODELS:
                for combo in entry["combos"]:
                    meta = run_baseline_case(entry, ps, combo)
                    if meta is None:
                        continue
                    rows_out.append(fill_ber_row(meta, rx_base, symb, rx_mean, rx_std))

        save_csv(rows_out, csv_path)
        logging.info("CSV -> %s", csv_path)

    plot_histograms(rows_out, OUT_DIR / "hist_ber_distribution.png")
    plot_block_heatmaps(rows_out, OUT_DIR / "heatmap_ber_blocks_by_snr.png")
    plot_grouped_bars(rows_out, OUT_DIR / "bar_ber_comboA_by_scale.png")
    logging.info("图已保存至 %s", OUT_DIR)


if __name__ == "__main__":
    main()
