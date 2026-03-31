import csv
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).parent
CSV_PATH = ROOT / "equalizer_bw_generalization_results.csv"

# 按要求：SNR 从小到大，最后是 No（无额外噪声）
SNR_ORDER = ["0 dB", "5 dB", "10 dB", "15 dB", "20 dB", "25 dB", "No"]

MODEL_ORDER = ["FCNN", "DNN", "BiLSTM", "RKAN", "KAN-FCNN", "Hybrid-KAN", "ResKAN"]
MODEL_COLORS = {
    "FCNN": "C0",
    "DNN": "C1",
    "BiLSTM": "C2",
    "RKAN": "C3",
    "KAN-FCNN": "C4",
    "Hybrid-KAN": "C5",
    "ResKAN": "C6",
}


def setup_chinese_font():
    """
    自动选择可用中文字体，修复中文标题/坐标轴显示问题。
    Windows 常见可用字体: Microsoft YaHei, SimHei, SimSun 等。
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    picked = [name for name in candidates if name in installed]
    if picked:
        matplotlib.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
    else:
        # 没找到中文字体时保留默认，避免程序报错
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_rows():
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def get_bandwidths(rows):
    vals = sorted({to_float(r["bandwidth_GHz"]) for r in rows})
    return vals


def plot_fixed_bw_vary_snr(rows):
    """
    固定带限，改变 SNR（从左到右: 0,5,10,15,20,25,No）
    每个带限一张图，图中包含所有模型。
    """
    bws = get_bandwidths(rows)

    for bw in bws:
        sub = [r for r in rows if to_float(r["bandwidth_GHz"]) == bw]
        # 构造 snr->row 映射，按 SNR_ORDER 排序取值
        snr_to_row = {r["SNR"].strip(): r for r in sub}

        x = np.arange(len(SNR_ORDER))
        fig, ax = plt.subplots(figsize=(8, 8))  # 正方形

        for model in MODEL_ORDER:
            y = []
            for snr in SNR_ORDER:
                row = snr_to_row.get(snr)
                if row is None:
                    y.append(np.nan)
                    continue
                v = to_float(row.get(model, np.nan))
                if v == 0:
                    v = 1e-6
                y.append(v)
            y = np.array(y, dtype=float)
            ax.semilogy(
                x,
                y,
                marker="o",
                linewidth=2,
                markersize=6,
                label=model,
                color=MODEL_COLORS.get(model, None),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(SNR_ORDER)
        ax.set_xlabel("SNR")
        ax.set_ylabel("BER")
        ax.set_title(f"固定带限 {bw:g} GHz：不同模型 BER vs SNR")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, ncol=2)
        plt.tight_layout()

        out_path = ROOT / f"fixedBW_{bw:g}GHz_varySNR.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out_path}")


def plot_fixed_snr_vary_bw(rows):
    """
    固定 SNR，改变带限（从小到大）。
    每个 SNR 一张图，图中包含所有模型。
    """
    bws = get_bandwidths(rows)
    x = np.array(bws, dtype=float)

    for snr in SNR_ORDER:
        sub = [r for r in rows if r["SNR"].strip() == snr]
        if not sub:
            continue

        # 构造 bw->row 映射
        bw_to_row = {to_float(r["bandwidth_GHz"]): r for r in sub}

        fig, ax = plt.subplots(figsize=(8, 8))  # 正方形

        for model in MODEL_ORDER:
            y = []
            for bw in bws:
                row = bw_to_row.get(bw)
                if row is None:
                    y.append(np.nan)
                    continue
                v = to_float(row.get(model, np.nan))
                if v == 0:
                    v = 1e-6
                y.append(v)
            y = np.array(y, dtype=float)
            ax.semilogy(
                x,
                y,
                marker="o",
                linewidth=2,
                markersize=6,
                label=model,
                color=MODEL_COLORS.get(model, None),
            )

        ax.set_xticks(x)
        ax.set_xlabel("带限带宽 (GHz)")
        ax.set_ylabel("BER")
        ax.set_title(f"固定 SNR={snr}：不同模型 BER vs 带限")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, ncol=2)
        plt.tight_layout()

        safe_snr = snr.replace(" ", "").replace("dB", "dB")
        out_path = ROOT / f"fixedSNR_{safe_snr}_varyBW.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out_path}")


def main():
    setup_chinese_font()
    rows = load_rows()
    plot_fixed_bw_vary_snr(rows)
    plot_fixed_snr_vary_bw(rows)
    print("Done.")


if __name__ == "__main__":
    main()
