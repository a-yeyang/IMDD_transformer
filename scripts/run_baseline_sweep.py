"""
基线均衡器 (FCNN / DNN / BiLSTM) 超参数扫描

与 KAN 扫描对齐: 参数量级 500/1000/2000/3000, 每种 4 组超参组合。
完成后合并 KAN 结果，生成综合论文图表并发送邮件通知。

FCNN   组合:  A=2层平衡 / B=3层 / C=1层 / D=2层陡降
DNN    组合:  A=2L+BN+drop0.1 / B=3L+BN+drop0.1 / C=2L无正则 / D=2L+BN+drop0.2
BiLSTM 组合:  A=1L-center / B=2L-center / C=1L-last / D=1L-center-minhead

用法:  python run_baseline_sweep.py
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
from train_fcnn import FCNNEqualizer, OpticalDataset
from train_dnn import DNNEqualizer
from train_bilstm import BiLSTMEqualizer

# ============================================================
#  全局常量
# ============================================================
WINDOW_SIZE  = 21
SPS          = 2
LABEL_SCALE  = 3.0
BATCH_SIZE   = 256
EPOCHS       = 30
LR           = 1e-3

PARAM_SCALES = [500, 1000, 2000, 3000]
SNR_LIST     = [0, 5, 10, 15, 20, 25, None]

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
#  Combo 定义: 模型工厂函数
# ============================================================
W = WINDOW_SIZE

# ----- FCNN -----
def _fcnn_a(h):
    return FCNNEqualizer(W, [h, max(1, h // 2)])

def _fcnn_b(h):
    return FCNNEqualizer(W, [h, max(1, h * 2 // 3), max(1, h // 3)])

def _fcnn_c(h):
    return FCNNEqualizer(W, [h])

def _fcnn_d(h):
    return FCNNEqualizer(W, [h, max(1, h // 4)])

# ----- DNN -----
def _dnn_a(h):
    return DNNEqualizer(W, [h, max(1, h // 2)], use_batchnorm=True, dropout=0.1)

def _dnn_b(h):
    return DNNEqualizer(W, [h, max(1, h * 2 // 3), max(1, h // 3)],
                        use_batchnorm=True, dropout=0.1)

def _dnn_c(h):
    return DNNEqualizer(W, [h, max(1, h // 2)], use_batchnorm=False, dropout=0.0)

def _dnn_d(h):
    return DNNEqualizer(W, [h, max(1, h // 2)], use_batchnorm=True, dropout=0.2)

# ----- BiLSTM -----
def _bilstm_a(h):
    return BiLSTMEqualizer(1, h, num_layers=1, use_center=True, window_size=W)

def _bilstm_b(h):
    return BiLSTMEqualizer(1, h, num_layers=2, use_center=True, window_size=W)

def _bilstm_c(h):
    return BiLSTMEqualizer(1, h, num_layers=1, use_center=False, window_size=W)

def _bilstm_d(h):
    m = BiLSTMEqualizer(1, h, num_layers=1, use_center=True, window_size=W)
    m.head = nn.Sequential(nn.Linear(h * 2, 1))
    return m


# ----- 配置描述 -----
def _cfg_fcnn_a(h):
    return f'dims=[{h}, {max(1, h // 2)}]'

def _cfg_fcnn_b(h):
    return f'dims=[{h}, {max(1, h * 2 // 3)}, {max(1, h // 3)}]'

def _cfg_fcnn_c(h):
    return f'dims=[{h}]'

def _cfg_fcnn_d(h):
    return f'dims=[{h}, {max(1, h // 4)}]'

def _cfg_dnn(h, combo_id):
    bn_map = {'A': 'BN+d0.1', 'B': 'BN+d0.1', 'C': 'no-reg', 'D': 'BN+d0.2'}
    if combo_id == 'B':
        dims = f'[{h}, {max(1, h * 2 // 3)}, {max(1, h // 3)}]'
    else:
        dims = f'[{h}, {max(1, h // 2)}]'
    return f'dims={dims}, {bn_map[combo_id]}'

def _cfg_bilstm(h, combo_id):
    info = {'A': '1L,center', 'B': '2L,center', 'C': '1L,last', 'D': '1L,center,min-head'}
    return f'h={h}, {info[combo_id]}'


# ----- 模型注册表 -----
BASELINE_MODELS = [
    {
        'key': 'fcnn', 'name': 'FCNN',
        'combos': [
            {'id': 'A', 'desc': '2-layer balanced',  'build': _fcnn_a, 'cfg': _cfg_fcnn_a},
            {'id': 'B', 'desc': '3-layer',           'build': _fcnn_b, 'cfg': _cfg_fcnn_b},
            {'id': 'C', 'desc': '1-layer wide',      'build': _fcnn_c, 'cfg': _cfg_fcnn_c},
            {'id': 'D', 'desc': '2-layer steep 4:1',  'build': _fcnn_d, 'cfg': _cfg_fcnn_d},
        ],
    },
    {
        'key': 'dnn', 'name': 'DNN',
        'combos': [
            {'id': 'A', 'desc': 'BN + dropout=0.1',
             'build': _dnn_a, 'cfg': lambda h: _cfg_dnn(h, 'A')},
            {'id': 'B', 'desc': '3-layer BN+drop0.1',
             'build': _dnn_b, 'cfg': lambda h: _cfg_dnn(h, 'B')},
            {'id': 'C', 'desc': 'no regularization',
             'build': _dnn_c, 'cfg': lambda h: _cfg_dnn(h, 'C')},
            {'id': 'D', 'desc': 'BN + dropout=0.2',
             'build': _dnn_d, 'cfg': lambda h: _cfg_dnn(h, 'D')},
        ],
    },
    {
        'key': 'bilstm', 'name': 'BiLSTM',
        'combos': [
            {'id': 'A', 'desc': '1L center (default)',
             'build': _bilstm_a, 'cfg': lambda h: _cfg_bilstm(h, 'A')},
            {'id': 'B', 'desc': '2L center',
             'build': _bilstm_b, 'cfg': lambda h: _cfg_bilstm(h, 'B')},
            {'id': 'C', 'desc': '1L last-step',
             'build': _bilstm_c, 'cfg': lambda h: _cfg_bilstm(h, 'C')},
            {'id': 'D', 'desc': '1L center min-head',
             'build': _bilstm_d, 'cfg': lambda h: _cfg_bilstm(h, 'D')},
        ],
    },
]


# ============================================================
#  参数量自动搜索 (通过实际构建模型来计数)
# ============================================================
def tune_by_building(build_fn, target, max_h=200):
    best_h, best_diff, best_p = 1, float('inf'), 0
    for h in range(1, max_h + 1):
        try:
            m = build_fn(h)
            p = sum(pp.numel() for pp in m.parameters())
            d = abs(p - target)
            if d < best_diff:
                best_diff, best_h, best_p = d, h, p
            del m
            if p > 2 * target:
                break
        except Exception:
            continue
    return best_h, best_p


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
    'param_scale', 'model', 'combo', 'combo_desc',
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
#  加载 KAN + Baseline 结果用于合并画图
# ============================================================
def load_csv_typed(path):
    results = []
    if not path.exists():
        return results
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            r = {
                'param_scale': int(row['param_scale']),
                'model': row['model'],
                'combo': row['combo'],
                'actual_params': int(row['actual_params']),
            }
            for snr in SNR_LIST:
                key = f'ber_{snr_tag(snr)}dB'
                r[key] = float(row[key])
            results.append(r)
    return results


def load_all_results():
    kan = load_csv_typed(SWEEP_DIR / 'kan_sweep_results.csv')
    base = load_csv_typed(SWEEP_DIR / 'baseline_sweep_results.csv')
    return kan + base


# ============================================================
#  样式定义 (6 个模型统一配色)
# ============================================================
ALL_STYLES = {
    'FCNN':       {'color': '#d62728', 'marker': 'v',  'ls': '-'},
    'DNN':        {'color': '#9467bd', 'marker': 'D',  'ls': '-'},
    'BiLSTM':     {'color': '#8c564b', 'marker': 'P',  'ls': '-'},
    'KAN-FCNN':   {'color': '#1f77b4', 'marker': 'o',  'ls': '--'},
    'Hybrid-KAN': {'color': '#ff7f0e', 'marker': 's',  'ls': '--'},
    'ResKAN':     {'color': '#2ca02c', 'marker': '^',  'ls': '--'},
}
MODEL_ORDER = ['FCNN', 'DNN', 'BiLSTM', 'KAN-FCNN', 'Hybrid-KAN', 'ResKAN']
BER_FLOOR = 1e-7


# ============================================================
#  绘图: 基线单参数量级 BER vs SNR
# ============================================================
def plot_baseline_per_scale(results, ps, log):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(SNR_LIST))
    for entry in BASELINE_MODELS:
        name = entry['name']
        st = ALL_STYLES[name]
        for r in results:
            if r['param_scale'] == ps and r['model'] == name and r['combo'] == 'A':
                bers = [max(r[f'ber_{snr_tag(s)}dB'], BER_FLOOR) for s in SNR_LIST]
                ax.semilogy(x, bers, marker=st['marker'], color=st['color'],
                            ls=st['ls'], lw=2, ms=8, label=name)
                break
    ax.set_xticks(x)
    ax.set_xticklabels([snr_display(s) for s in SNR_LIST])
    ax.set_xlabel('SNR')
    ax.set_ylabel('BER')
    ax.set_title(f'Baseline BER vs SNR | Target ~ {ps} params (Combo A)')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.4)
    path = FIG_DIR / f'baseline_scale_{ps}.png'
    fig.savefig(str(path), bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [Plot] {path.name}')


# ============================================================
#  绘图: 综合 2×2 BER vs SNR (全部 6 个模型, Combo A)
# ============================================================
def plot_combined_overview(all_results, log):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    x = np.arange(len(SNR_LIST))

    for idx, ps in enumerate(PARAM_SCALES):
        ax = axes[idx // 2][idx % 2]
        for name in MODEL_ORDER:
            st = ALL_STYLES[name]
            for r in all_results:
                if r['param_scale'] == ps and r['model'] == name and r['combo'] == 'A':
                    bers = [max(r[f'ber_{snr_tag(s)}dB'], BER_FLOOR) for s in SNR_LIST]
                    ax.semilogy(x, bers, marker=st['marker'], color=st['color'],
                                ls=st['ls'], lw=2, ms=7, label=name)
                    break
        ax.set_xticks(x)
        ax.set_xticklabels([snr_display(s) for s in SNR_LIST], fontsize=9)
        ax.set_ylabel('BER')
        ax.set_title(f'Target ~ {ps} params')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which='both', ls='--', alpha=0.4)

    fig.suptitle('All Equalizers — BER vs SNR (Combo A, Default)', fontsize=14, y=1.01)
    fig.tight_layout()
    path = FIG_DIR / 'combined_overview.png'
    fig.savefig(str(path), bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [Plot] {path.name}')


# ============================================================
#  绘图: 综合 Scaling (BER vs Param Count, Combo A)
# ============================================================
def plot_combined_scaling(all_results, log):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in MODEL_ORDER:
        st = ALL_STYLES[name]
        xs, ys = [], []
        for ps in PARAM_SCALES:
            for r in all_results:
                if r['param_scale'] == ps and r['model'] == name and r['combo'] == 'A':
                    xs.append(r['actual_params'])
                    ys.append(max(r[f'ber_{snr_tag(None)}dB'], BER_FLOOR))
                    break
        if xs:
            ax.semilogy(xs, ys, marker=st['marker'], color=st['color'],
                        ls=st['ls'], lw=2, ms=10, label=name)

    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('BER (Noiseless)')
    ax.set_title('Performance Scaling — BER vs Model Size (Combo A)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    path = FIG_DIR / 'combined_scaling.png'
    fig.savefig(str(path), bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [Plot] {path.name}')


# ============================================================
#  绘图: 综合柱状图 (Combo A, 按参数量级分组)
# ============================================================
def plot_combined_bars(all_results, log):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    n = len(MODEL_ORDER)
    bw = 0.12

    for idx, ps in enumerate(PARAM_SCALES):
        ax = axes[idx // 2][idx % 2]
        x = np.arange(1)
        for mi, name in enumerate(MODEL_ORDER):
            st = ALL_STYLES[name]
            val = BER_FLOOR
            for r in all_results:
                if r['param_scale'] == ps and r['model'] == name and r['combo'] == 'A':
                    val = max(r[f'ber_{snr_tag(None)}dB'], BER_FLOOR)
                    break
            ax.bar(mi * bw, val, bw * 0.9, color=st['color'], alpha=0.85, label=name)

        ax.set_xticks([i * bw for i in range(n)])
        ax.set_xticklabels(MODEL_ORDER, fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('BER (Noiseless)')
        ax.set_title(f'Target ~ {ps} params')
        ax.set_yscale('log')
        ax.grid(True, which='both', axis='y', ls='--', alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=8,
              bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('All Equalizers — BER Comparison (Combo A, Noiseless)',
                 fontsize=14, y=1.06)
    fig.tight_layout()
    path = FIG_DIR / 'combined_bars.png'
    fig.savefig(str(path), bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [Plot] {path.name}')


# ============================================================
#  绘图: 基线灵敏度热力图
# ============================================================
def plot_baseline_sensitivity(results, log):
    combo_ids = ['A', 'B', 'C', 'D']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for idx, entry in enumerate(BASELINE_MODELS):
        ax = axes[idx]
        name = entry['name']
        combo_labels = [f"Combo {c['id']}\n({c['desc'][:18]})" for c in entry['combos']]

        data = np.full((len(PARAM_SCALES), len(combo_ids)), np.nan)
        for i, ps in enumerate(PARAM_SCALES):
            for j, cid in enumerate(combo_ids):
                for r in results:
                    if (r['param_scale'] == ps and r['model'] == name
                            and r['combo'] == cid):
                        data[i, j] = r[f'ber_{snr_tag(None)}dB']
                        break

        valid = data[np.isfinite(data) & (data > 0)]
        if len(valid) == 0:
            continue
        vmin = max(valid.min(), BER_FLOOR)
        vmax = valid.max()
        data_plot = np.where(data > 0, data, BER_FLOOR)

        im = ax.imshow(data_plot, cmap='RdYlGn_r', aspect='auto',
                       norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)))
        ax.set_xticks(range(len(combo_ids)))
        ax.set_xticklabels(combo_labels, fontsize=7)
        ax.set_yticks(range(len(PARAM_SCALES)))
        ax.set_yticklabels(PARAM_SCALES)
        ax.set_xlabel('Hyperparameter Combo')
        ax.set_ylabel('Target Parameters')
        ax.set_title(name)

        med = np.nanmedian(data_plot)
        for i in range(data_plot.shape[0]):
            for j in range(data_plot.shape[1]):
                if np.isfinite(data_plot[i, j]):
                    c = 'white' if data_plot[i, j] > med else 'black'
                    ax.text(j, i, f'{data_plot[i, j]:.2e}', ha='center',
                            va='center', fontsize=8, color=c, fontweight='bold')
        plt.colorbar(im, ax=ax, label='BER (Noiseless)', shrink=0.8)

    fig.suptitle('Baseline Hyperparameter Sensitivity — BER Heatmap',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = FIG_DIR / 'baseline_sensitivity.png'
    fig.savefig(str(path), bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [Plot] {path.name}')


# ============================================================
#  绘图: KAN vs Baseline 对比 (每个参数量级, Combo A)
# ============================================================
def plot_kan_vs_baseline(all_results, log):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    x = np.arange(len(SNR_LIST))
    baseline_names = ['FCNN', 'DNN', 'BiLSTM']
    kan_names = ['KAN-FCNN', 'Hybrid-KAN', 'ResKAN']

    for idx, ps in enumerate(PARAM_SCALES):
        ax = axes[idx // 2][idx % 2]
        for name in baseline_names + kan_names:
            st = ALL_STYLES[name]
            for r in all_results:
                if r['param_scale'] == ps and r['model'] == name and r['combo'] == 'A':
                    bers = [max(r[f'ber_{snr_tag(s)}dB'], BER_FLOOR) for s in SNR_LIST]
                    group = 'Baseline' if name in baseline_names else 'KAN'
                    ax.semilogy(x, bers, marker=st['marker'], color=st['color'],
                                ls=st['ls'], lw=2, ms=7,
                                label=f'{name}')
                    break

        ax.set_xticks(x)
        ax.set_xticklabels([snr_display(s) for s in SNR_LIST], fontsize=9)
        ax.set_ylabel('BER')
        ax.set_title(f'Target ~ {ps} params')
        ax.legend(fontsize=7, ncol=2, loc='lower left')
        ax.grid(True, which='both', ls='--', alpha=0.4)

    fig.suptitle('KAN vs Baseline — BER vs SNR (Combo A)',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    path = FIG_DIR / 'combined_kan_vs_baseline.png'
    fig.savefig(str(path), bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [Plot] {path.name}')


# ============================================================
#  绘图: 综合 Pareto 前沿
# ============================================================
def plot_combined_pareto(all_results, log):
    fig, ax = plt.subplots(figsize=(9, 6))

    for name in MODEL_ORDER:
        st = ALL_STYLES[name]
        xs, ys = [], []
        for ps in PARAM_SCALES:
            for r in all_results:
                if r['param_scale'] == ps and r['model'] == name and r['combo'] == 'A':
                    xs.append(r['actual_params'])
                    ys.append(max(r[f'ber_{snr_tag(None)}dB'], BER_FLOOR))
                    break
        if xs:
            ax.semilogy(xs, ys, marker=st['marker'], color=st['color'],
                        ls=st['ls'], lw=2, ms=9, label=name)

    all_pts = []
    for r in all_results:
        if r['combo'] == 'A':
            all_pts.append((r['actual_params'],
                            max(r[f'ber_{snr_tag(None)}dB'], BER_FLOOR)))
    if all_pts:
        all_pts.sort(key=lambda t: t[0])
        px, py = [all_pts[0][0]], [all_pts[0][1]]
        best = all_pts[0][1]
        for xi, yi in all_pts[1:]:
            if yi < best:
                px.append(xi)
                py.append(yi)
                best = yi
        ax.semilogy(px, py, 'k--', lw=2.5, alpha=0.5, label='Pareto frontier')

    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('BER (Noiseless)')
    ax.set_title('Pareto Analysis — All Architectures (Combo A)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    path = FIG_DIR / 'combined_pareto.png'
    fig.savefig(str(path), bbox_inches='tight')
    plt.close(fig)
    log.info(f'  [Plot] {path.name}')


# ============================================================
#  邮件正文
# ============================================================
def format_email_body(all_results):
    lines = [
        '基线 + KAN 均衡器超参数扫描全部完成!',
        '',
        f'参数量级: {PARAM_SCALES}',
        f'模型: {MODEL_ORDER}',
        '',
        '=' * 55,
        '  Noiseless BER 对比 (Combo A, 默认配置)',
        '=' * 55,
    ]
    for ps in PARAM_SCALES:
        lines.append(f'\n--- 参数量级 ~{ps} ---')
        for name in MODEL_ORDER:
            for r in all_results:
                if r['param_scale'] == ps and r['model'] == name and r['combo'] == 'A':
                    lines.append(
                        f"  {name:12s}  actual={r['actual_params']:5d}  "
                        f"BER={r[f'ber_{snr_tag(None)}dB']:.4e}"
                    )
                    break
    lines.append('\n\n详细结果见附件 CSV 和图表。')
    return '\n'.join(lines)


# ============================================================
#  主流程
# ============================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log = logging.getLogger('baseline_sweep')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(LOGS_DIR / f'baseline_sweep_{timestamp}.log', 'w', 'utf-8')
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

    pin = (device == 'cuda')
    train_ds = OpticalDataset(rx_train, symb_train, WINDOW_SIZE, SPS,
                              rx_mean=rx_mean, rx_std=rx_std, label_scale=LABEL_SCALE)
    test_ds = OpticalDataset(rx_test, symb_test, WINDOW_SIZE, SPS,
                             rx_mean=rx_mean, rx_std=rx_std, label_scale=LABEL_SCALE)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=pin)
    val_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=pin)

    all_results = []
    csv_path = SWEEP_DIR / 'baseline_sweep_results.csv'
    sweep_start = time.perf_counter()

    for ps in PARAM_SCALES:
        group_start = time.perf_counter()
        log.info(f'\n{"=" * 60}')
        log.info(f'  PARAMETER SCALE: {ps}')
        log.info(f'{"=" * 60}')

        for entry in BASELINE_MODELS:
            model_key = entry['key']
            model_name = entry['name']

            for combo in entry['combos']:
                cid = combo['id']
                tag = f'{model_key}_{ps}_{cid}'
                ckpt = CKPT_DIR / f'sweep_baseline_{tag}.pth'

                try:
                    best_h, actual_params = tune_by_building(combo['build'], ps)
                    cfg_str = combo['cfg'](best_h)
                    log.info(f'  {model_name:7s} Combo {cid} ({combo["desc"]:22s})  '
                             f'target={ps:5d}  actual={actual_params:5d}  {cfg_str}')

                    model = combo['build'](best_h).to(device)

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
                        'model': model_name,
                        'combo': cid,
                        'combo_desc': combo['desc'],
                        'hidden_config': cfg_str,
                        'actual_params': actual_params,
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
                    log.error(f'    [FAIL] {model_name} combo {cid} @ {ps}: {exc}')

                if device == 'cuda':
                    torch.cuda.empty_cache()

        group_time = time.perf_counter() - group_start
        log.info(f'\n  Scale {ps} done, {group_time:.0f}s')

        save_csv(all_results, csv_path)
        log.info(f'  [CSV] {csv_path}')

        plot_baseline_per_scale(all_results, ps, log)

    total_time = time.perf_counter() - sweep_start
    log.info(f'\n\n{"=" * 60}')
    log.info(f'  基线扫描完成！总耗时 {total_time:.0f}s ({total_time / 60:.1f} min)')
    log.info(f'{"=" * 60}\n')

    # ----- 基线灵敏度热力图 -----
    log.info('生成基线灵敏度图...')
    plot_baseline_sensitivity(all_results, log)

    # ----- 合并 KAN 结果, 生成综合图 -----
    log.info('加载 KAN 结果, 生成综合对比图...')
    combined = load_all_results()
    if combined:
        plot_combined_overview(combined, log)
        plot_combined_scaling(combined, log)
        plot_combined_bars(combined, log)
        plot_kan_vs_baseline(combined, log)
        plot_combined_pareto(combined, log)

    # ----- 发送邮件 -----
    log.info('\n发送邮件通知...')
    try:
        from sendmail import send_notification
        figures = sorted(FIG_DIR.glob('*.png'))
        body = format_email_body(combined if combined else all_results)
        send_notification(
            subject=f'基线+KAN 超参数扫描完成 (耗时 {total_time / 60:.0f} min)',
            body=body,
            attachments=[str(csv_path), str(SWEEP_DIR / 'kan_sweep_results.csv')]
                        + [str(f) for f in figures],
        )
        log.info('[Email] 通知邮件已发送')
    except Exception as exc:
        log.error(f'[Email] 发送失败: {exc}')

    log.info('Done.')


if __name__ == '__main__':
    main()
