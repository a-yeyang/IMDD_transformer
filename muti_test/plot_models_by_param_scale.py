import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv


ROOT = Path(__file__).parent
CSV_PATH = ROOT / "multi_snr_test_results.csv"
OUT_DIR = ROOT

# 用户要求固定横轴顺序: 0, 5, 10, 15, 20, 25, No
# 仅根据数据表绘图：若某个 SNR 列不存在，则该点记为 NaN（曲线断开）。
SNR_SPEC = [
    # ("0 dB", "ber_0dB"),
    # ("5 dB", "ber_5dB"),
    ("10 dB", "ber_10dB"),
    ("15 dB", "ber_15dB"),
    ("20 dB", "ber_20dB"),
    ("25 dB", "ber_25dB"),
  #  ("No", "ber_No"),
]

MODEL_ORDER = [
    "KAN-FCNN", "Hybrid-KAN", "ResKAN",
    "FCNN", "DNN", "BiLSTM",
]

COLOR_MAP = {
    "KAN-FCNN": "C0",
    "Hybrid-KAN": "C1",
    "ResKAN": "C2",
    "FCNN": "C3",
    "DNN": "C4",
    "BiLSTM": "C5",
}


def main():
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # 按 (param_scale, model) 聚合，A/B/C/D 取均值
    # store[(scale, model)][snr_col] = [values...]
    store = {}
    for r in rows:
        scale = int(float(r["param_scale"]))
        model = r["model"]
        key = (scale, model)
        if key not in store:
            store[key] = {col: [] for _, col in SNR_SPEC}
        for _, col in SNR_SPEC:
            raw = r.get(col, "")
            if raw is None or raw == "":
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            # 0 BER -> 1e-6
            if v == 0:
                v = 1e-6
            store[key][col].append(v)

    scales = sorted({k[0] for k in store.keys()})
    snr_labels = [x[0] for x in SNR_SPEC]
    snr_cols = [x[1] for x in SNR_SPEC]

    for scale in scales:

        fig, ax = plt.subplots(figsize=(8, 8))
        x = np.arange(len(snr_cols))

        for model in MODEL_ORDER:
            key = (scale, model)
            if key not in store:
                continue
            y = []
            for col in snr_cols:
                vals = store[key].get(col, [])
                if not vals:
                    y.append(np.nan)
                else:
                    y.append(float(np.mean(vals)))
            y = np.array(y, dtype=float)
            y = np.where(y == 0, 1e-6, y)
            ax.semilogy(
                x, y,
                marker="o",
                linewidth=2,
                markersize=7,
                label=model,
                color=COLOR_MAP.get(model, None),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(snr_labels)
        ax.set_xlabel("SNR")
        ax.set_ylabel("BER")
        ax.set_title(
            f"BER Curves by Model @ Param Scale {scale} (A/B/C/D mean)\n"
            "SNR order: 0, 5, 10, 15, 20, 25, No"
        )
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(ncol=2, fontsize=9)
        plt.tight_layout()

        out_png = OUT_DIR / f"line_ber_models_scale_{scale}.png"
        plt.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out_png}")

    print("Done.")


if __name__ == "__main__":
    main()
