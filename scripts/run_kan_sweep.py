"""
KAN 均衡器超参数扫描 — 训练 + 测试 + 分析一体化程序

扫描维度:
  参数量级:    500 / 1000 / 2000 / 3000
  模型架构:    KAN-FCNN / Hybrid-KAN / ResKAN
  超参数组合:  A(G=5,k=3) / B(G=8,k=3) / C(G=3,k=5) / D(G=10,k=2)

流程:
  1. 自动搜索隐层维度使实际参数量逼近目标
  2. 逐组训练 + 多 SNR 测试 (GPU 加速)
  3. 每完成一个参数量级保存 CSV + 中间图表
  4. 全部完成后生成科研论文级综合分析图
  5. 通过 QQ 邮箱发送完成通知

用法:
  python run_kan_sweep.py
"""

import sys
import csv
import time
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.io
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ============================================================
#  路径 & 导入
# ============================================================
ROOT      = Path(__file__).parent.parent
SWEEP_DIR = ROOT / 'sweep_results'
CKPT_DIR  = SWEEP_DIR / 'checkpoints'
FIG_DIR   = SWEEP_DIR / 'figures'
LOGS_DIR  = ROOT / 'logs'
for _d in (SWEEP_DIR, CKPT_DIR, FIG_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from train_kan_ideas import (
    KANLinear, KANFCNNEqualizer, HybridKANEqualizer, ResKANEqualizer,
    OpticalDataset,
)

# ============================================================
#  全局常量
# ============================================================
WINDOW_SIZE  = 21
SPS          = 2
LABEL_SCALE  = 3.0
BATCH_SIZE   = 256
EPOCHS       = 30
LR           = 1e-3
GRID_RANGE   = (-2.0, 2.0)

PARAM_SCALES = [500, 1000, 2000, 3000]
SNR_LIST     = [0, 5, 10, 15, 20, 25, None]

COMBOS = [
    {'id': 'A', 'G': 5,  'k': 3, 'desc': 'G=5,k=3 (cubic, default)'},
    {'id': 'B', 'G': 8,  'k': 3, 'desc': 'G=8,k=3 (fine cubic)'},
    {'id': 'C', 'G': 3,  'k': 5, 'desc': 'G=3,k=5 (quintic)'},
    {'id': 'D', 'G': 10, 'k': 2, 'desc': 'G=10,k=2 (quadratic)'},
]

MODEL_KEYS = ['kan_fcnn', 'hybrid_kan', 'res_kan']
MODELS = {
    'kan_fcnn':   {'name': 'KAN-FCNN',   'color': '#1f77b4', 'marker': 'o'},
    'hybrid_kan': {'name': 'Hybrid-KAN', 'color': '#ff7f0e', 'marker': 's'},
    'res_kan':    {'name': 'ResKAN',     'color': '#2ca02c', 'marker': '^'},
}

COMBO_LS = {'A': '-', 'B': '--', 'C': '-.', 'D': ':'}

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'savefig.dpi': 300,
    'font.sans-serif': ['Times New Roman', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
})


# ============================================================
#  工具函数
# ============================================================
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def snr_tag(snr):
    return 'inf' if snr is None else str(snr)


def snr_display(snr):
    return 'Noiseless' if snr is None else f'{snr} dB'


def add_awgn(rx, snr_db):
    if snr_db is None:
        return rx.copy()
    ps = np.mean(rx.astype(np.float64) ** 2)
    sigma = np.sqrt(ps / 10 ** (snr_db / 10))
    return rx + np.random.RandomState(42).randn(*rx.shape) * sigma


def compute_ber(pred, true):
    def _labels(x):
        return np.select(
            [x < -2, (x >= -2) & (x < 0), (x >= 0) & (x < 2)],
            [-3, -1, 1], default=3,
        )
    return float(np.mean(_labels(pred) != _labels(true)))


# ============================================================
#  参数量自动搜索
# ============================================================
def _F(G, k):
    """KANLinear 每条连接的参数数: spline_weight(G+k) + base_weight(1)"""
    return G + k + 1


def tune_kan_fcnn(target, W, G, k):
    F = _F(G, k)
    best, diff = [2, 1], float('inf')
    for h1 in range(1, 200):
        for h2 in range(1, h1 + 1):
            p = F * (W * h1 + h1 * h2 + h2) + 2 * (h1 + h2)
            d = abs(p - target)
            if d < diff:
                diff, best = d, [h1, h2]
            if p > 2 * target:
                break
    return best


def tune_hybrid_kan(target, W, G, k):
    F = _F(G, k)
    best, diff = [4, 2], float('inf')
    for h1 in range(2, 300):
        for h2 in range(1, h1 + 1):
            p = h1 * (W + 1) + h2 * (h1 + 1 + F)
            d = abs(p - target)
            if d < diff:
                diff, best = d, [h1, h2]
            if p > 2 * target:
                break
    return best


def tune_res_kan(target, W, G, k):
    F = _F(G, k)
    best, diff = ([4, 2], 2), float('inf')
    for fh1 in range(2, 150):
        for fh2 in range(1, fh1 + 1):
            fcnn_p = fh1 * (W + 1) + fh2 * (fh1 + 2) + 1
            rem = target - fcnn_p - 1
            if rem < F * (W + 1):
                continue
            kh_f = rem / (F * (W + 1))
            for kh in (max(1, int(kh_f)), max(1, int(kh_f) + 1)):
                p = fcnn_p + kh * F * (W + 1) + 1
                d = abs(p - target)
                if d < diff:
                    diff, best = d, ([fh1, fh2], kh)
    return best


# ============================================================
#  模型构建
# ============================================================
def build_sweep_model(mt, target, G, k, device):
    W = WINDOW_SIZE
    if mt == 'kan_fcnn':
        dims = tune_kan_fcnn(target, W, G, k)
        m = KANFCNNEqualizer(W, dims, G, k, GRID_RANGE).to(device)
        cfg = f'hidden={dims}'
    elif mt == 'hybrid_kan':
        dims = tune_hybrid_kan(target, W, G, k)
        m = HybridKANEqualizer(W, dims, G, k, GRID_RANGE).to(device)
        cfg = f'linear={dims}'
    else:
        fd, kh = tune_res_kan(target, W, G, k)
        m = ResKANEqualizer(W, fd, kh, G, k, GRID_RANGE).to(device)
        cfg = f'fcnn={fd},kh={kh}'
    nparams = sum(p.numel() for p in m.parameters())
    return m, nparams, cfg


# ============================================================
#  训练（单次配置）
# ============================================================
def train_once(model, train_ld, val_ld, device, log, ckpt_path):
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    best_vl, best_ber, best_ep, last_tl = float('inf'), float('inf'), 0, 0.0

    model.train()
    for ep in range(1, EPOCHS + 1):
        ep_loss = 0.0
        for x, y in train_ld:
            x, y = x.unsqueeze(-1).to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sched.step()
        last_tl = ep_loss / len(train_ld)

        model.eval()
        vl_sum, pa, ta = 0.0, [], []
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.unsqueeze(-1).to(device), y.to(device)
                o = model(x)
                vl_sum += crit(o, y).item()
                pa.append(o.cpu().numpy())
                ta.append(y.cpu().numpy())
        model.train()

        vl = vl_sum / len(val_ld)
        pr = np.concatenate(pa).flatten() * LABEL_SCALE
        tr = np.concatenate(ta).flatten() * LABEL_SCALE
        ber = compute_ber(pr, tr)

        if vl < best_vl:
            best_vl, best_ber, best_ep = vl, ber, ep
            torch.save(model.state_dict(), str(ckpt_path))

        if ep % 10 == 0 or ep == EPOCHS:
            log.info(f'      ep {ep:3d}/{EPOCHS}  '
                     f'train={last_tl:.6f}  val={vl:.6f}  ber={ber:.4e}')

    return best_ep, best_vl, best_ber, last_tl


# ============================================================
#  多 SNR 测试
# ============================================================
def test_snrs(model, rx_base, symb, rxm, rxs, device):
    model.eval()
    out = {}
    for snr in SNR_LIST:
        rx = add_awgn(rx_base, snr)
        ds = OpticalDataset(rx, symb, WINDOW_SIZE, SPS, rxm, rxs, LABEL_SCALE)
        ld = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0,
                        pin_memory=(device == 'cuda'))
        pa, ta = [], []
        with torch.inference_mode():
            for x, y in ld:
                pa.append(model(x.unsqueeze(-1).to(device, non_blocking=True)))
                ta.append(y)
        pr = torch.cat(pa).flatten().cpu().numpy() * LABEL_SCALE
        tr = torch.cat(ta).flatten().numpy() * LABEL_SCALE
        out[snr] = compute_ber(pr, tr)
    return out


# ============================================================
#  数据加载
# ============================================================
def load_data():
    data = scipy.io.loadmat(str(ROOT / 'dataset_for_python.mat'))
    rxt = data['rx_train_export'].flatten()
    if np.iscomplexobj(rxt):
        rxt = np.abs(rxt)
    st = data['symb_train_export'].flatten()
    rxe = data['rx_test_export'].flatten()
    if np.iscomplexobj(rxe):
        rxe = np.abs(rxe)
    se = data['symb_test_export'].flatten()
    return rxt, st, rxe, se, float(np.mean(rxt)), float(np.std(rxt))


# ============================================================
#  CSV 写入
# ============================================================
CSV_FIELDS = [
    'param_scale', 'model', 'combo', 'grid_size', 'spline_order',
    'hidden_config', 'actual_params', 'best_epoch',
    'train_loss', 'val_loss', 'val_ber',
] + [f'ber_{snr_tag(s)}dB' for s in SNR_LIST]


def save_csv(results, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for row in results:
            line = []
            for field in CSV_FIELDS:
                v = row.get(field, '')
                if isinstance(v, float):
                    line.append(f'{v:.6e}')
                else:
                    line.append(v)
            w.writerow(line)


# ============================================================
#  绘图: 单一参数量级 BER vs SNR (默认组合 A)
# ============================================================
def plot_per_scale(results, ps, log):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(SNR_LIST))

    for mk in MODEL_KEYS:
        mi = MODELS[mk]
        for r in results:
            if r['param_scale'] == ps and r['model'] == mi['name'] and r['combo'] == 'A':
                bers = [max(r[f'ber_{snr_tag(s)}dB'], 1e-7) for s in SNR_LIST]
                ax.semilogy(x, bers, marker=mi['marker'], color=mi['color'],
                            linewidth=2, markersize=8, label=mi['name'])
                break

    ax.set_xticks(x)
    ax.set_xticklabels([snr_display(s) for s in SNR_LIST])
    ax.set_xlabel('SNR')
    ax.set_ylabel('BER')
    ax.set_title(f'BER vs SNR  |  Target ~ {ps} params (Combo A: G=5, k=3)')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.4)

    path = FIG_DIR / f'sweep_scale_{ps}.png'
    fig.savefig(str(path))
    plt.close(fig)
    log.info(f'  [Plot] {path.name}')


# ============================================================
#  绘图: 2x2 总览 BER vs SNR
# ============================================================
def plot_overview(results, log):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(SNR_LIST))

    for idx, ps in enumerate(PARAM_SCALES):
        ax = axes[idx // 2][idx % 2]
        for mk in MODEL_KEYS:
            mi = MODELS[mk]
            for r in results:
                if r['param_scale'] == ps and r['model'] == mi['name'] and r['combo'] == 'A':
                    bers = [max(r[f'ber_{snr_tag(s)}dB'], 1e-7) for s in SNR_LIST]
                    ax.semilogy(x, bers, marker=mi['marker'], color=mi['color'],
                                lw=2, ms=7, label=mi['name'])
                    break
        ax.set_xticks(x)
        ax.set_xticklabels([snr_display(s) for s in SNR_LIST], fontsize=9)
        ax.set_ylabel('BER')
        ax.set_title(f'Target ~ {ps} params')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', ls='--', alpha=0.4)

    fig.suptitle('KAN Equalizer — BER vs SNR Overview (Combo A)', fontsize=14, y=1.01)
    fig.tight_layout()
    path = FIG_DIR / 'sweep_overview.png'
    fig.savefig(str(path))
    plt.close(fig)
    log.info(f'  [Plot] {path.name}')


# ============================================================
#  绘图: BER vs 参数量 (Scaling)
# ============================================================
def plot_scaling(results, log):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for mk in MODEL_KEYS:
        mi = MODELS[mk]
        xs, ys = [], []
        for ps in PARAM_SCALES:
            for r in results:
                if r['param_scale'] == ps and r['model'] == mi['name'] and r['combo'] == 'A':
                    xs.append(r['actual_params'])
                    ys.append(max(r[f'ber_{snr_tag(None)}dB'], 1e-7))
                    break
        if xs:
            ax.semilogy(xs, ys, marker=mi['marker'], color=mi['color'],
                        lw=2, ms=10, label=mi['name'])

    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('BER (Noiseless)')
    ax.set_title('Performance Scaling — BER vs Model Size (Combo A)')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.4)

    path = FIG_DIR / 'sweep_scaling.png'
    fig.savefig(str(path))
    plt.close(fig)
    log.info(f'  [Plot] {path.name}')


# ============================================================
#  绘图: 超参数灵敏度热力图
# ============================================================
def plot_sensitivity(results, log):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    combo_labels = [f"Combo {c['id']}\n(G={c['G']},k={c['k']})" for c in COMBOS]

    for idx, mk in enumerate(MODEL_KEYS):
        ax = axes[idx]
        mi = MODELS[mk]
        data = np.full((len(PARAM_SCALES), len(COMBOS)), np.nan)

        for i, ps in enumerate(PARAM_SCALES):
            for j, combo in enumerate(COMBOS):
                for r in results:
                    if (r['param_scale'] == ps and
                            r['model'] == mi['name'] and
                            r['combo'] == combo['id']):
                        data[i, j] = r[f'ber_{snr_tag(None)}dB']
                        break

        valid = data[np.isfinite(data) & (data > 0)]
        if len(valid) == 0:
            continue
        vmin = max(valid.min(), 1e-7)
        vmax = valid.max()

        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto',
                       norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)))
        ax.set_xticks(range(len(COMBOS)))
        ax.set_xticklabels(combo_labels, fontsize=8)
        ax.set_yticks(range(len(PARAM_SCALES)))
        ax.set_yticklabels(PARAM_SCALES)
        ax.set_xlabel('Hyperparameter Combo')
        ax.set_ylabel('Target Parameters')
        ax.set_title(mi['name'])

        med = np.nanmedian(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isfinite(data[i, j]):
                    c = 'white' if data[i, j] > med else 'black'
                    ax.text(j, i, f'{data[i, j]:.2e}', ha='center', va='center',
                            fontsize=8, color=c, fontweight='bold')

        plt.colorbar(im, ax=ax, label='BER (Noiseless)', shrink=0.8)

    fig.suptitle('Hyperparameter Sensitivity — BER Heatmap (Noiseless)', fontsize=14, y=1.02)
    fig.tight_layout()
    path = FIG_DIR / 'sweep_sensitivity.png'
    fig.savefig(str(path))
    plt.close(fig)
    log.info(f'  [Plot] {path.name}')


# ============================================================
#  绘图: 参数组合对比柱状图
# ============================================================
def plot_combo_bars(results, log):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    combo_ids = [c['id'] for c in COMBOS]
    n_models = len(MODEL_KEYS)
    bw = 0.22

    for idx, ps in enumerate(PARAM_SCALES):
        ax = axes[idx // 2][idx % 2]
        x = np.arange(len(combo_ids))

        for mi_idx, mk in enumerate(MODEL_KEYS):
            mi = MODELS[mk]
            bers = []
            for cid in combo_ids:
                val = np.nan
                for r in results:
                    if (r['param_scale'] == ps and
                            r['model'] == mi['name'] and
                            r['combo'] == cid):
                        val = max(r[f'ber_{snr_tag(None)}dB'], 1e-7)
                        break
                bers.append(val)
            ax.bar(x + mi_idx * bw, bers, bw,
                   label=mi['name'], color=mi['color'], alpha=0.85)

        ax.set_xticks(x + bw * (n_models - 1) / 2)
        ax.set_xticklabels([f'Combo {c}' for c in combo_ids])
        ax.set_ylabel('BER (Noiseless)')
        ax.set_title(f'Target ~ {ps} params')
        ax.legend(fontsize=8)
        ax.set_yscale('log')
        ax.grid(True, which='both', axis='y', ls='--', alpha=0.3)

    fig.suptitle('Combo Comparison — BER by Hyperparameter Configuration', fontsize=14, y=1.01)
    fig.tight_layout()
    path = FIG_DIR / 'sweep_combo_bars.png'
    fig.savefig(str(path))
    plt.close(fig)
    log.info(f'  [Plot] {path.name}')


# ============================================================
#  绘图: Pareto 前沿 (参数量 vs BER, 所有配置)
# ============================================================
def plot_pareto(results, log):
    fig, ax = plt.subplots(figsize=(9, 6))

    for mk in MODEL_KEYS:
        mi = MODELS[mk]
        for combo in COMBOS:
            xs, ys = [], []
            for ps in PARAM_SCALES:
                for r in results:
                    if (r['param_scale'] == ps and
                            r['model'] == mi['name'] and
                            r['combo'] == combo['id']):
                        xs.append(r['actual_params'])
                        ys.append(max(r[f'ber_{snr_tag(None)}dB'], 1e-7))
                        break
            if xs:
                label = f"{mi['name']} ({combo['id']})" if combo['id'] == 'A' else None
                alpha = 1.0 if combo['id'] == 'A' else 0.4
                ax.semilogy(xs, ys, marker=mi['marker'], color=mi['color'],
                            ls=COMBO_LS[combo['id']], lw=1.5, ms=7,
                            alpha=alpha, label=label)

    all_pts = [(r['actual_params'], max(r[f'ber_{snr_tag(None)}dB'], 1e-7))
               for r in results if np.isfinite(r[f'ber_{snr_tag(None)}dB'])]
    if all_pts:
        all_pts.sort(key=lambda t: t[0])
        pareto_x, pareto_y = [all_pts[0][0]], [all_pts[0][1]]
        best_y = all_pts[0][1]
        for px, py in all_pts[1:]:
            if py < best_y:
                pareto_x.append(px)
                pareto_y.append(py)
                best_y = py
        ax.semilogy(pareto_x, pareto_y, 'k--', lw=2.5, alpha=0.5, label='Pareto frontier')

    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('BER (Noiseless)')
    ax.set_title('Pareto Analysis — All Configurations')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', ls='--', alpha=0.4)

    path = FIG_DIR / 'sweep_pareto.png'
    fig.savefig(str(path))
    plt.close(fig)
    log.info(f'  [Plot] {path.name}')


# ============================================================
#  邮件正文格式化
# ============================================================
def format_email_body(results):
    lines = [
        'KAN 均衡器超参数扫描已完成!',
        '',
        f'扫描范围: 参数量级 {PARAM_SCALES}',
        f'模型类型: {[MODELS[k]["name"] for k in MODEL_KEYS]}',
        f'超参组合: {[c["id"] + "(" + c["desc"] + ")" for c in COMBOS]}',
        '',
        '=' * 50,
        '  结果摘要 (Noiseless BER, 默认组合 A)',
        '=' * 50,
    ]
    for ps in PARAM_SCALES:
        lines.append(f'\n--- 参数量级 ~{ps} ---')
        for mk in MODEL_KEYS:
            mi = MODELS[mk]
            for r in results:
                if r['param_scale'] == ps and r['model'] == mi['name'] and r['combo'] == 'A':
                    lines.append(
                        f"  {mi['name']:12s}  actual={r['actual_params']:5d}  "
                        f"BER={r[f'ber_{snr_tag(None)}dB']:.4e}"
                    )
                    break

    lines.append('\n\n详细结果见附件 CSV 和分析图表。')
    return '\n'.join(lines)


# ============================================================
#  主扫描流程
# ============================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log = logging.getLogger('kan_sweep')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(LOGS_DIR / f'kan_sweep_{timestamp}.log', 'w', 'utf-8')
    sh = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)

    device = get_device()
    log.info(f'Device: {device}')
    if device == 'cuda':
        log.info(f'GPU: {torch.cuda.get_device_name()}')

    rx_train, symb_train, rx_test, symb_test, rx_mean, rx_std = load_data()
    log.info(f'Train: {len(rx_train):,} samples, {len(symb_train):,} symbols')
    log.info(f'Test:  {len(rx_test):,} samples, {len(symb_test):,} symbols')

    train_ds = OpticalDataset(rx_train, symb_train, WINDOW_SIZE, SPS,
                              rx_mean=rx_mean, rx_std=rx_std, label_scale=LABEL_SCALE)
    test_ds = OpticalDataset(rx_test, symb_test, WINDOW_SIZE, SPS,
                             rx_mean=rx_mean, rx_std=rx_std, label_scale=LABEL_SCALE)
    pin = (device == 'cuda')
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=pin)
    val_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=pin)

    all_results = []
    csv_path = SWEEP_DIR / 'kan_sweep_results.csv'
    sweep_start = time.perf_counter()

    for ps in PARAM_SCALES:
        group_start = time.perf_counter()
        log.info(f'\n{"=" * 60}')
        log.info(f'  PARAMETER SCALE: {ps}')
        log.info(f'{"=" * 60}')

        for combo in COMBOS:
            cid, G, k = combo['id'], combo['G'], combo['k']
            log.info(f'\n  Combo {cid}: {combo["desc"]}')

            for mk in MODEL_KEYS:
                mi = MODELS[mk]
                tag = f'{mk}_{ps}_{cid}'
                ckpt = CKPT_DIR / f'sweep_{tag}.pth'

                try:
                    model, nparams, cfg = build_sweep_model(mk, ps, G, k, device)
                    log.info(f'    {mi["name"]:12s}  target={ps:5d}  '
                             f'actual={nparams:5d}  {cfg}')

                    t0 = time.perf_counter()
                    best_ep, best_vl, best_ber, last_tl = train_once(
                        model, train_ld, val_ld, device, log, ckpt)
                    train_time = time.perf_counter() - t0
                    log.info(f'    -> best_ep={best_ep}  val_ber={best_ber:.4e}  '
                             f'time={train_time:.1f}s')

                    model.load_state_dict(
                        torch.load(str(ckpt), map_location=device, weights_only=True))
                    snr_bers = test_snrs(model, rx_test, symb_test,
                                        rx_mean, rx_std, device)

                    row = {
                        'param_scale': ps,
                        'model': mi['name'],
                        'combo': cid,
                        'grid_size': G,
                        'spline_order': k,
                        'hidden_config': cfg,
                        'actual_params': nparams,
                        'best_epoch': best_ep,
                        'train_loss': last_tl,
                        'val_loss': best_vl,
                        'val_ber': best_ber,
                    }
                    for snr in SNR_LIST:
                        row[f'ber_{snr_tag(snr)}dB'] = snr_bers[snr]
                        log.info(f'      SNR={snr_display(snr):10s} -> '
                                 f'BER={snr_bers[snr]:.4e}')
                    all_results.append(row)

                except Exception as exc:
                    log.error(f'    [FAIL] {mi["name"]} @ {ps}/{cid}: {exc}')

                if device == 'cuda':
                    torch.cuda.empty_cache()

        group_time = time.perf_counter() - group_start
        log.info(f'\n  Scale {ps} 完成, 耗时 {group_time:.0f}s')

        save_csv(all_results, csv_path)
        log.info(f'  [CSV] {csv_path}')

        plot_per_scale(all_results, ps, log)

    total_time = time.perf_counter() - sweep_start
    log.info(f'\n\n{"=" * 60}')
    log.info(f'  全部扫描完成！总耗时 {total_time:.0f}s ({total_time / 60:.1f} min)')
    log.info(f'{"=" * 60}\n')

    log.info('生成综合分析图...')
    plot_overview(all_results, log)
    plot_scaling(all_results, log)
    plot_sensitivity(all_results, log)
    plot_combo_bars(all_results, log)
    plot_pareto(all_results, log)

    log.info('\n发送邮件通知...')
    try:
        from sendmail import send_notification
        figures = sorted(FIG_DIR.glob('*.png'))
        body = format_email_body(all_results)
        send_notification(
            subject=f'KAN 超参数扫描完成 (耗时 {total_time / 60:.0f} min)',
            body=body,
            attachments=[str(csv_path)] + [str(f) for f in figures],
        )
        log.info('[Email] 通知邮件已发送')
    except Exception as exc:
        log.error(f'[Email] 发送失败: {exc}')

    log.info('Done.')


if __name__ == '__main__':
    main()
