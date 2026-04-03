"""
非线性效应均衡器综合对比实验
==============================

测试 6 种均衡器在 10 组不同非线性条件下的 BER 性能:
  信道条件:
    - BTB (Back-to-Back, 无光纤): 激光器驱动电压 0.5/0.75/1.0/1.25/1.5 V
    - SSMF 5km 光纤:             激光器驱动电压 0.5/0.75/1.0/1.25/1.5 V
  均衡器:
    基线: FCNN / DNN / BiLSTM
    KAN:  KAN-FCNN / Hybrid-KAN / ResKAN

GPU 高利用率训练策略 (解决 DataLoader 瓶颈):
  - 每个条件的训练/验证数据在首次使用前 **向量化一次性预载入 GPU 显存**
    (130k 样本 × 21 特征 × float32 ≈ 11MB/数据集，10 组共 ≈ 220MB，24GB 显存充裕)
  - 训练时直接在显存内随机打乱并切 batch，零 CPU↔GPU 传输，GPU 可持续满载
  - 6 个模型在同一条件的 GPU-resident 数据上顺序训练（避免线程竞争）

CPU 并行测试:
  - ProcessPoolExecutor (8 核) 并行推理，每进程独立负责一个 (条件, 模型) 组合

开关 (默认均为 False，程序默认执行全流程: 训练 → 测试 → 绘图):
  TEST_ONLY = False   # True: 跳过训练，直接加载已有 checkpoint 进行测试
  PLOT_ONLY = False   # True: 跳过训练和测试，直接从已有 results.csv 重新绘图

数据准备:
  先运行 RX_nolinear.m 生成 nolinear_results/mat_data/*.mat 数据文件

运行方式:
  python compare_all_equalizers_nolinear.py

输出目录: nolinear_results/
  mat_data/     .mat 数据文件 (由 RX_nolinear.m 生成)
  models/       模型 checkpoint (按条件分子目录)
  logs/         训练与测试日志
  figures/      全部 PNG 图像
  results.csv   BER 汇总表格
"""

import os
import sys
import time
import csv
import logging
import smtplib
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime

import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ── 加入 scripts 目录以便导入各模型模块 ──────────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

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
    build_kan_fcnn,   KAN_FCNN_CONFIG,
    build_hybrid_kan, HYBRID_KAN_CONFIG,
    build_res_kan,    RES_KAN_CONFIG,
)

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# =============================================================================
#  ██████╗ ██╗ ██╗   ██╗ ██████╗██╗  ██╗███████╗███████╗
#  ██╔════╝██║ ██║   ██║██╔════╝██║  ██║██╔════╝██╔════╝
#  ███████╗██║ ██║   ██║██║     ███████║█████╗  ███████╗
#  ╚════██║██║ ╚██╗ ██╔╝██║     ██╔══██║██╔══╝  ╚════██║
#  ███████║███████╗ ╚████╔╝ ╚██████╗██║  ██║███████╗███████║
#  ╚══════╝╚══════╝  ╚═══╝   ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝
#
#  以下三个开关控制程序行为，默认全为 False（执行完整流程）
# =============================================================================

TEST_ONLY = False   # True → 跳过训练，加载已有 checkpoint 直接测试
PLOT_ONLY = True   # True → 跳过训练和测试，直接从 results.csv 重新绘图

# =============================================================================
#  路径 & 全局配置
# =============================================================================

ROOT        = SCRIPTS_DIR.parent
MAT_DIR     = ROOT / 'nolinear_results' / 'mat_data'
OUTPUT_DIR  = ROOT / 'nolinear_results'
MODELS_DIR  = OUTPUT_DIR / 'models'
LOGS_DIR    = OUTPUT_DIR / 'logs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
RESULTS_CSV = OUTPUT_DIR / 'results.csv'

for _d in [OUTPUT_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# GPU 训练超参
TRAIN_EPOCHS     = {'fcnn': 20, 'dnn': 20, 'bilstm': 40,
                    'kan_fcnn': 30, 'hybrid_kan': 30, 'res_kan': 30}
# GPU-resident 训练：数据已在显存内，batch 可设很大以充分利用 CUDA 并行
# 130k 样本 / 16384 batch ≈ 8 次迭代/epoch，每次迭代 GPU 充分满载
TRAIN_BATCH_SIZE = 16384
LABEL_SCALE      = 3.0

# 测试阶段 SNR 列表 (None 表示无外加 AWGN，保留物理仿真噪声)
SNR_LIST = [5, 10, 15, 20, 25, None]

# 测试 batch 大小
TEST_BATCH_SIZE = 8192

# GPU 训练：每条件内 6 个模型顺序训练（数据已预载 GPU，GPU 可持续满载）
# 不使用多线程——线程之间的 GIL + 小模型快速完成导致频繁等待，反而降低利用率

# CPU 测试并行进程数 (None = os.cpu_count())
CPU_TEST_WORKERS = None

# =============================================================================
#  邮件配置
# =============================================================================

SMTP_SERVER    = 'smtp.qq.com'
SMTP_PORT      = 465
SENDER_EMAIL   = '2861173454@qq.com'
SENDER_AUTH    = 'oaryallyjeufdedh'
RECEIVER_EMAIL = '2861173454@qq.com'


def send_email(subject: str, body: str, attachments=None):
    """发送 QQ 邮箱通知，支持多附件。出错不中断主流程。"""
    try:
        msg = MIMEMultipart()
        msg['From']    = SENDER_EMAIL
        msg['To']      = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        if attachments:
            for fp in attachments:
                p = Path(fp)
                if not p.exists():
                    print(f'[Email] 附件不存在，跳过: {p}')
                    continue
                if p.stat().st_size > 20 * 1024 * 1024:   # > 20 MB 跳过
                    print(f'[Email] 附件过大 (>20MB)，跳过: {p.name}')
                    continue
                with open(p, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition',
                                f'attachment; filename="{p.name}"')
                msg.attach(part)

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_AUTH)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print(f'[Email] 邮件已发送: {subject}')
    except Exception as e:
        print(f'[Email] 发送失败 ({e})，继续运行...')


# =============================================================================
#  实验条件定义
# =============================================================================

CONDITIONS = [
    # BTB (无光纤)
    {'id': 'BTB_laser0.5',  'label': 'BTB V=0.5V',
     'fiber': 'BTB', 'voltage': 0.5,  'mat': 'dataset_BTB_laser0.5.mat'},
    {'id': 'BTB_laser0.75', 'label': 'BTB V=0.75V',
     'fiber': 'BTB', 'voltage': 0.75, 'mat': 'dataset_BTB_laser0.75.mat'},
    {'id': 'BTB_laser1.0',  'label': 'BTB V=1.0V',
     'fiber': 'BTB', 'voltage': 1.0,  'mat': 'dataset_BTB_laser1.0.mat'},
    {'id': 'BTB_laser1.25', 'label': 'BTB V=1.25V',
     'fiber': 'BTB', 'voltage': 1.25, 'mat': 'dataset_BTB_laser1.25.mat'},
    {'id': 'BTB_laser1.5',  'label': 'BTB V=1.5V',
     'fiber': 'BTB', 'voltage': 1.5,  'mat': 'dataset_BTB_laser1.5.mat'},
    # 5km SSMF 光纤
    {'id': '5km_laser0.5',  'label': '5km V=0.5V',
     'fiber': '5km', 'voltage': 0.5,  'mat': 'dataset_5km_laser0.5.mat'},
    {'id': '5km_laser0.75', 'label': '5km V=0.75V',
     'fiber': '5km', 'voltage': 0.75, 'mat': 'dataset_5km_laser0.75.mat'},
    {'id': '5km_laser1.0',  'label': '5km V=1.0V',
     'fiber': '5km', 'voltage': 1.0,  'mat': 'dataset_5km_laser1.0.mat'},
    {'id': '5km_laser1.25', 'label': '5km V=1.25V',
     'fiber': '5km', 'voltage': 1.25, 'mat': 'dataset_5km_laser1.25.mat'},
    {'id': '5km_laser1.5',  'label': '5km V=1.5V',
     'fiber': '5km', 'voltage': 1.5,  'mat': 'dataset_5km_laser1.5.mat'},
]

# =============================================================================
#  模型构建函数（基线部分，复用 compare_all_equalizers 的模式）
# =============================================================================

def _build_fcnn(config, device):
    return FCNNEqualizer(
        input_dim   = config['window_size'],
        hidden_dims = config['hidden_dims'],
    ).to(device)


def _build_dnn(config, device):
    return DNNEqualizer(
        input_dim     = config['window_size'],
        hidden_dims   = config['hidden_dims'],
        use_batchnorm = config['use_batchnorm'],
        dropout       = config['dropout'],
    ).to(device)


def _build_bilstm(config, device):
    return BiLSTMEqualizer(
        input_size  = 1,
        hidden_size = config['hidden_size'],
        num_layers  = config['num_lstm_layers'],
        use_center  = config['use_center'],
        window_size = config['window_size'],
    ).to(device)


# =============================================================================
#  模型注册表
# =============================================================================

def make_model_registry():
    """返回模型注册表列表（每次调用生成独立副本，避免多条件间配置污染）。"""
    return [
        {
            'name': 'FCNN',      'key': 'fcnn',
            'ckpt': 'fcnn.pth',
            'build_fn': _build_fcnn,
            'config': {**dict(FCNN_CONFIG)},
            'color': 'C2', 'marker': '^', 'ls': '-.',
        },
        {
            'name': 'DNN',       'key': 'dnn',
            'ckpt': 'dnn.pth',
            'build_fn': _build_dnn,
            'config': {**dict(DNN_CONFIG)},
            'color': 'C4', 'marker': 'v', 'ls': '-.',
        },
        {
            'name': 'BiLSTM',    'key': 'bilstm',
            'ckpt': 'bilstm.pth',
            'build_fn': _build_bilstm,
            'config': {**dict(BILSTM_CONFIG)},
            'color': 'C3', 'marker': 'D', 'ls': ':',
        },
        {
            'name': 'KAN-FCNN',  'key': 'kan_fcnn',
            'ckpt': 'kan_fcnn.pth',
            'build_fn': build_kan_fcnn,
            'config': {**dict(KAN_FCNN_CONFIG)},
            'color': 'C0', 'marker': 'o', 'ls': '-',
        },
        {
            'name': 'Hybrid-KAN','key': 'hybrid_kan',
            'ckpt': 'hybrid_kan.pth',
            'build_fn': build_hybrid_kan,
            'config': {**dict(HYBRID_KAN_CONFIG)},
            'color': 'C1', 'marker': 's', 'ls': '-',
        },
        {
            'name': 'ResKAN',    'key': 'res_kan',
            'ckpt': 'res_kan.pth',
            'build_fn': build_res_kan,
            'config': {**dict(RES_KAN_CONFIG)},
            'color': 'C6', 'marker': 'p', 'ls': '-',
        },
    ]


# =============================================================================
#  通用工具函数
# =============================================================================

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def snr_label(snr):
    return '无噪声' if snr is None else f'{snr} dB'


def add_awgn(rx_signal, snr_db, seed=42):
    if snr_db is None or np.isinf(snr_db):
        return rx_signal.copy()
    rx    = np.asarray(rx_signal, dtype=np.float64)
    Ps    = np.mean(rx ** 2)
    Pn    = Ps / (10 ** (snr_db / 10))
    noise = np.random.RandomState(seed).randn(*rx.shape).astype(np.float64) * np.sqrt(Pn)
    return (rx + noise).astype(np.float64)


def calculate_ber(pred_scaled, true_scaled):
    pred_labels = np.select(
        [pred_scaled < -2,
         (pred_scaled >= -2) & (pred_scaled < 0),
         (pred_scaled >=  0) & (pred_scaled < 2)],
        [-3, -1, 1], default=3
    )
    errors = np.sum(pred_labels != true_scaled.astype(int))
    return float(errors) / len(true_scaled)


def load_mat_data(mat_path):
    """加载 .mat 数据，返回 rx_train, rx_test, symb_train, symb_test, rx_mean, rx_std。"""
    data = scipy.io.loadmat(str(mat_path))
    rx_train   = data['rx_train_export'].flatten()
    symb_train = data['symb_train_export'].flatten()
    rx_test    = data['rx_test_export'].flatten()
    symb_test  = data['symb_test_export'].flatten()

    if np.iscomplexobj(rx_train):
        rx_train = np.abs(rx_train).astype(np.float64)
    else:
        rx_train = rx_train.astype(np.float64)

    if np.iscomplexobj(rx_test):
        rx_test = np.abs(rx_test).astype(np.float64)
    else:
        rx_test = rx_test.astype(np.float64)

    rx_mean = float(np.mean(rx_train))
    rx_std  = float(np.std(rx_train))
    return rx_train, symb_train, rx_test, symb_test, rx_mean, rx_std


def setup_logger(name, log_path, also_stdout=True):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt)
    log.addHandler(fh)
    if also_stdout:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
    return log


def run_inference_cpu(model, loader):
    """CPU 上完成所有 batch 推理，返回 (preds_scaled, targets_scaled)。"""
    preds_list, targets_list = [], []
    with torch.inference_mode():
        for inputs, tgt in loader:
            inputs = inputs.unsqueeze(-1)
            preds_list.append(model(inputs))
            targets_list.append(tgt)
    preds   = torch.cat(preds_list).flatten().numpy() * LABEL_SCALE
    targets = torch.cat(targets_list).flatten().numpy() * LABEL_SCALE
    return preds, targets


# =============================================================================
#  GPU-resident 数据集构建（向量化，一次性预载入显存）
#
#  核心思路：
#    传统 DataLoader(num_workers=0) 每次迭代需要 Python 逐样本切片 + to(device)，
#    CPU 准备数据的时间远超小模型的 GPU 计算时间，导致 GPU 利用率 <10%。
#
#    解决方案：用 numpy 向量化操作一次性构建全部 (N, window_size) 输入矩阵，
#    整体搬到 GPU 显存，训练时直接 tensor slicing，零 CPU-GPU 传输。
#    GPU 利用率可从 <8% 提升到 70-90%+。
# =============================================================================

def build_gpu_tensors(rx_signal, labels, window_size, sps, rx_mean, rx_std,
                      label_scale, device):
    """
    向量化构建 (X_gpu, y_gpu)，全部驻留在 GPU 显存。

    等价于 OpticalDataset，但用 numpy 矩阵索引一次性完成，速度快 100x+。

    Returns:
        X_gpu : FloatTensor (N, window_size)  — 已归一化
        y_gpu : FloatTensor (N, 1)            — 已缩放
    """
    n_samples = len(labels) - (window_size // sps) - 1

    idx      = np.arange(n_samples, dtype=np.int64)
    starts   = idx * sps                                       # (N,)
    col_idx  = starts[:, None] + np.arange(window_size)[None, :]  # (N, W)

    X = rx_signal[col_idx].astype(np.float32)                 # (N, W)
    X = (X - rx_mean) / (rx_std + 1e-8)

    label_idx = idx + (window_size // sps) // 2
    y = labels[label_idx].astype(np.float32) / label_scale    # (N,)

    X_gpu = torch.from_numpy(X).to(device)
    y_gpu = torch.from_numpy(y[:, None]).to(device)
    return X_gpu, y_gpu


def _gpu_batch_iter(X, y, batch_size, shuffle=True):
    """GPU-resident 数据迭代器：在显存内随机打乱后切 batch，零 CPU 开销。"""
    n = X.shape[0]
    if shuffle:
        perm = torch.randperm(n, device=X.device)
        X, y = X[perm], y[perm]
    for start in range(0, n, batch_size):
        yield X[start:start + batch_size].unsqueeze(-1), y[start:start + batch_size]


def train_single_model_gpu(entry, X_train, y_train, X_val, y_val,
                            rx_mean, rx_std, cond_id,
                            ckpt_dir, fig_dir, log_dir, device):
    """
    在 GPU-resident 张量上训练单个模型。

    GPU 数据已预载，训练循环中无任何 CPU-GPU 传输，
    GPU 计算持续满载，利用率可达 70-90%+。
    """
    key    = entry['key']
    name   = entry['name']
    config = dict(entry['config'])
    config['device']       = device
    config['batch_size']   = TRAIN_BATCH_SIZE
    config['epochs']       = TRAIN_EPOCHS.get(key, 20)
    config['label_scale']  = LABEL_SCALE

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = Path(log_dir) / f'train_{cond_id}_{key}_{ts}.log'
    log_m    = setup_logger(f'tr_{cond_id}_{key}_{ts}', log_path, also_stdout=False)

    try:
        model     = entry['build_fn'](config, device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=1e-5
        )

        total_p = sum(p.numel() for p in model.parameters())
        log_m.info(f'[{cond_id}][{name}] GPU-resident 训练 | '
                   f'device={device} | params={total_p:,} | '
                   f'epochs={config["epochs"]} | batch={config["batch_size"]}')
        log_m.info(f'  训练={X_train.shape[0]:,}样本  验证={X_val.shape[0]:,}样本  '
                   f'迭代/epoch≈{X_train.shape[0]//config["batch_size"]+1}')

        best_val_loss = float('inf')
        best_val_ber  = float('inf')
        best_epoch    = 0
        train_hist, val_loss_hist, val_ber_hist = [], [], []
        eval_interval = config.get('eval_interval', 1)

        for epoch in range(1, config['epochs'] + 1):
            t0 = time.perf_counter()
            model.train()
            ep_loss, n_batches = 0.0, 0

            for x_b, y_b in _gpu_batch_iter(X_train, y_train,
                                             config['batch_size'], shuffle=True):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x_b), y_b)
                loss.backward()
                optimizer.step()
                ep_loss  += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = ep_loss / n_batches
            train_hist.append(avg_loss)

            if epoch % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    out_val = model(X_val.unsqueeze(-1))
                    v_loss  = criterion(out_val, y_val).item()
                    preds_sc = out_val.squeeze().cpu().numpy() * LABEL_SCALE
                    tgts_sc  = y_val.squeeze().cpu().numpy()   * LABEL_SCALE
                val_ber = calculate_ber(preds_sc, tgts_sc)

                val_loss_hist.append(v_loss)
                val_ber_hist.append(val_ber)
                elapsed = time.perf_counter() - t0
                log_m.info(f'  Epoch [{epoch:3d}/{config["epochs"]}] '
                           f'{elapsed:5.2f}s | TrainL={avg_loss:.6f} | '
                           f'ValL={v_loss:.6f} | ValBER={val_ber:.4e}')

                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_val_ber  = val_ber
                    best_epoch    = epoch
                    ckpt_path = Path(ckpt_dir) / entry['ckpt']
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_type': name,
                        'rx_mean': rx_mean,
                        'rx_std':  rx_std,
                        'config':  config,
                        'epoch':   epoch,
                        'best_val_loss': best_val_loss,
                        'best_val_ber':  best_val_ber,
                    }, str(ckpt_path))
                    log_m.info(f'  ★ 更新最优 → Epoch {epoch} | BER={val_ber:.4e}')
            else:
                log_m.info(f'  Epoch [{epoch:3d}/{config["epochs"]}] '
                           f'{time.perf_counter()-t0:5.2f}s | TrainL={avg_loss:.6f}')

        log_m.info(f'[{cond_id}][{name}] 完成 | 最优 Epoch={best_epoch} | '
                   f'ValBER={best_val_ber:.4e}')

        # 绘制训练曲线
        eval_x = list(range(eval_interval, config['epochs'] + 1, eval_interval))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(train_hist,               label='Train Loss')
        axes[0].plot(eval_x, val_loss_hist,    label='Val Loss')
        axes[0].axvline(best_epoch, color='r', ls='--', label=f'Best ({best_epoch})')
        axes[0].set_title('MSE Loss'); axes[0].set_xlabel('Epoch')
        axes[0].legend(); axes[0].grid(alpha=0.4)
        axes[1].semilogy(eval_x, val_ber_hist, 'o-', label='Val BER')
        axes[1].axvline(best_epoch, color='r', ls='--', label=f'Best ({best_epoch})')
        axes[1].set_title('Val BER (PAM4)'); axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('BER'); axes[1].legend()
        axes[1].grid(True, which='both', alpha=0.4)
        plt.suptitle(f'{name} [{cond_id}]  BestBER={best_val_ber:.4e}', fontsize=11)
        plt.tight_layout()
        fig_path = Path(fig_dir) / f'train_{cond_id}_{key}.png'
        plt.savefig(str(fig_path), dpi=120, bbox_inches='tight')
        plt.close()

        print(f'    ✓ {name:12s}  BestBER={best_val_ber:.4e}  @Epoch{best_epoch}')

        return {'key': key, 'name': name, 'cond_id': cond_id,
                'best_val_ber': best_val_ber, 'best_epoch': best_epoch,
                'status': 'ok'}

    except Exception:
        tb = traceback.format_exc()
        log_m.error(f'[{cond_id}][{name}] 训练出错:\n{tb}')
        print(f'    ✗ {name:12s}  训练失败: {tb[:200]}')
        return {'key': key, 'name': name, 'cond_id': cond_id,
                'best_val_ber': float('nan'), 'best_epoch': 0,
                'status': 'error'}


# =============================================================================
#  CPU 并行测试 — 必须定义在模块顶层，以支持 Windows spawn 模式 pickle
# =============================================================================

def _cpu_test_worker(task_args):
    """
    ProcessPoolExecutor 工作函数：在 CPU 上测试一个条件下的一个模型。

    每个子进程独立加载 checkpoint 和数据，遍历全部 SNR 点，返回 BER 字典。
    """
    (cond_id, model_key, model_name, build_fn_name, config_dict,
     ckpt_path_str, mat_path_str, snr_list) = task_args

    torch.set_num_threads(1)   # 防止多进程×多线程过度订阅

    # 动态重新获取 build_fn（ProcessPoolExecutor 子进程重新 import 模块）
    _build_fn_map = {
        '_build_fcnn':      _build_fcnn,
        '_build_dnn':       _build_dnn,
        '_build_bilstm':    _build_bilstm,
        'build_kan_fcnn':   build_kan_fcnn,
        'build_hybrid_kan': build_hybrid_kan,
        'build_res_kan':    build_res_kan,
    }
    build_fn = _build_fn_map[build_fn_name]

    # 加载 checkpoint
    ckpt = torch.load(ckpt_path_str, weights_only=False, map_location='cpu')
    rx_mean  = float(ckpt['rx_mean'])
    rx_std   = float(ckpt['rx_std'])
    saved_cfg = ckpt.get('config', {})
    config_dict.update(saved_cfg)

    model = build_fn(config_dict, 'cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 加载测试数据
    data = scipy.io.loadmat(mat_path_str)
    rx_test_base = data['rx_test_export'].flatten()
    if np.iscomplexobj(rx_test_base):
        rx_test_base = np.abs(rx_test_base).astype(np.float64)
    else:
        rx_test_base = rx_test_base.astype(np.float64)
    symb_test = data['symb_test_export'].flatten()

    model_results = {}
    for snr in snr_list:
        rx_test = add_awgn(rx_test_base, snr)
        ds = OpticalDataset(
            rx_test, symb_test,
            config_dict['window_size'], config_dict['sps'],
            rx_mean=rx_mean, rx_std=rx_std,
            label_scale=LABEL_SCALE,
        )
        loader = DataLoader(ds, batch_size=TEST_BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=False)
        preds, targets = run_inference_cpu(model, loader)
        model_results[snr] = calculate_ber(preds, targets)

    return cond_id, model_key, model_name, model_results


# =============================================================================
#  训练主流程
#  策略：逐条件顺序处理。每个条件：
#    1. 向量化构建训练/验证数据，一次性搬入 GPU 显存
#    2. 6 个模型逐一在该 GPU-resident 数据上训练（顺序，非线程）
#    3. 训练完释放该条件的 GPU 数据，处理下一条件
#
#  为何不用多线程：
#    - 模型极小，每个 CUDA kernel < 1ms；线程切换和 GIL 竞争开销相当
#    - 多线程实测 GPU 利用率 <8%，单线程 GPU-resident 可达 70-90%+
# =============================================================================

def run_training(conditions, log):
    device = get_device()
    log.info(f'[训练] 使用设备: {device}')
    log.info(f'[训练] 策略: GPU-resident 顺序训练（预载显存，零 CPU-GPU 传输）')
    log.info(f'[训练] batch_size={TRAIN_BATCH_SIZE}  条件数={len(conditions)}  模型/条件=6')

    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        log.info(f'[训练] GPU: {props.name}  显存: {props.total_memory/1024**3:.1f} GB')

    train_summary  = {}
    all_train_figs = []
    total_conds    = len(conditions)

    for ci, cond in enumerate(conditions):
        cond_id  = cond['id']
        mat_path = MAT_DIR / cond['mat']

        if not mat_path.exists():
            log.warning(f'[训练][{ci+1}/{total_conds}] 跳过 {cond_id}：未找到 {mat_path}')
            continue

        log.info(f'\n{"="*72}')
        log.info(f'  [{ci+1}/{total_conds}] 条件: {cond_id}  ({cond["label"]})')
        log.info(f'{"="*72}')

        ckpt_dir = MODELS_DIR / cond_id
        ckpt_dir.mkdir(exist_ok=True)

        # ── 加载数据 & 向量化构建 GPU 张量 ──────────────────────────────────
        rx_train, symb_train, rx_test, symb_test, rx_mean, rx_std = \
            load_mat_data(mat_path)

        # 使用与 OpticalDataset 一致的 window_size/sps（取自 FCNN 配置）
        _cfg0      = make_model_registry()[0]['config']
        window_sz  = _cfg0['window_size']
        sps_val    = _cfg0['sps']

        log.info(f'  预载训练数据到 GPU... (window={window_sz}, sps={sps_val})')
        t_load = time.time()
        X_tr, y_tr = build_gpu_tensors(rx_train, symb_train, window_sz, sps_val,
                                        rx_mean, rx_std, LABEL_SCALE, device)
        X_va, y_va = build_gpu_tensors(rx_test,  symb_test,  window_sz, sps_val,
                                        rx_mean, rx_std, LABEL_SCALE, device)
        mem_mb = (X_tr.nbytes + y_tr.nbytes + X_va.nbytes + y_va.nbytes) / 1024**2
        log.info(f'  GPU 数据就绪: 训练={X_tr.shape[0]:,}  验证={X_va.shape[0]:,}'
                 f'  耗时={time.time()-t_load:.2f}s  显存≈{mem_mb:.1f}MB')

        if device == 'cuda':
            alloc_mb = torch.cuda.memory_allocated() / 1024**2
            log.info(f'  当前已分配显存: {alloc_mb:.0f} MB')

        # ── 逐模型顺序训练 ──────────────────────────────────────────────────
        registry     = make_model_registry()
        cond_results = {}
        t_cond_start = time.time()

        for entry in registry:
            res = train_single_model_gpu(
                entry, X_tr, y_tr, X_va, y_va,
                rx_mean, rx_std, cond_id,
                str(ckpt_dir), str(FIGURES_DIR), str(LOGS_DIR), device
            )
            cond_results[res['key']] = res['best_val_ber']
            fig_p = FIGURES_DIR / f'train_{cond_id}_{res["key"]}.png'
            if fig_p.exists():
                all_train_figs.append(str(fig_p))

        # ── 释放该条件的显存 ────────────────────────────────────────────────
        del X_tr, y_tr, X_va, y_va
        if device == 'cuda':
            torch.cuda.empty_cache()

        elapsed = time.time() - t_cond_start
        log.info(f'\n  [{cond_id}] 全部 {len(registry)} 个模型完成 | 耗时 {elapsed/60:.1f} min')
        for k, ber in cond_results.items():
            log.info(f'    {k:15s}  ValBER = {ber:.4e}')
        train_summary[cond_id] = cond_results

    return train_summary, all_train_figs


# =============================================================================
#  测试主流程：所有条件+模型，CPU 多进程并行
# =============================================================================

def run_testing(conditions, log):
    n_workers = CPU_TEST_WORKERS or os.cpu_count() or 1
    log.info(f'\n[测试] CPU 并行进程数: {n_workers}')
    log.info(f'[测试] SNR 列表: {[snr_label(s) for s in SNR_LIST]}')

    # 收集所有任务
    task_list = []
    registry_template = make_model_registry()

    for cond in conditions:
        cond_id  = cond['id']
        mat_path = MAT_DIR / cond['mat']
        ckpt_dir = MODELS_DIR / cond_id

        if not mat_path.exists():
            log.warning(f'[测试] 跳过 {cond_id}：未找到数据 {mat_path}')
            continue

        for entry in registry_template:
            ckpt_path = ckpt_dir / entry['ckpt']
            if not ckpt_path.exists():
                log.warning(f'[测试] 跳过 {cond_id}/{entry["name"]}：'
                            f'未找到 checkpoint {ckpt_path}')
                continue

            # build_fn 用名字传递，子进程内重新映射（避免 pickle 问题）
            fn_name_map = {
                _build_fcnn:      '_build_fcnn',
                _build_dnn:       '_build_dnn',
                _build_bilstm:    '_build_bilstm',
                build_kan_fcnn:   'build_kan_fcnn',
                build_hybrid_kan: 'build_hybrid_kan',
                build_res_kan:    'build_res_kan',
            }
            task_list.append((
                cond_id,
                entry['key'],
                entry['name'],
                fn_name_map[entry['build_fn']],
                dict(entry['config']),
                str(ckpt_path),
                str(mat_path),
                SNR_LIST,
            ))

    log.info(f'[测试] 共 {len(task_list)} 个测试任务，启动 {n_workers} 个子进程...\n')

    # {cond_id: {model_key: {snr: ber}}}
    results = {c['id']: {} for c in conditions}

    actual_workers = min(n_workers, len(task_list)) if task_list else 1

    with ProcessPoolExecutor(max_workers=actual_workers) as exe:
        future_map = {exe.submit(_cpu_test_worker, t): t for t in task_list}
        for fut in as_completed(future_map):
            try:
                cond_id, model_key, model_name, model_results = fut.result()
                results[cond_id][model_key] = model_results
                for snr in SNR_LIST:
                    ber = model_results.get(snr, float('nan'))
                    log.info(f'  [{cond_id:20s}][{model_name:12s}] '
                             f'SNR={snr_label(snr):8s}  BER={ber:.4e}')
            except Exception as exc:
                task = future_map[fut]
                log.error(f'  [{task[0]}][{task[2]}] 测试失败: {exc}')

    return results


# =============================================================================
#  结果保存与绘图
# =============================================================================

def save_csv(conditions, results):
    """保存 BER 汇总到 CSV。"""
    registry = make_model_registry()
    model_names = [e['name'] for e in registry]

    fieldnames = ['条件ID', '条件描述', '信道', '驱动电压(V)', 'SNR'] + model_names
    rows = []
    for cond in conditions:
        cid = cond['id']
        if cid not in results or not results[cid]:
            continue
        for snr in SNR_LIST:
            row = {
                '条件ID':    cid,
                '条件描述':  cond['label'],
                '信道':      cond['fiber'],
                '驱动电压(V)': cond['voltage'],
                'SNR':       snr_label(snr),
            }
            for entry in registry:
                ber = results[cid].get(entry['key'], {}).get(snr, float('nan'))
                row[entry['name']] = ber
            rows.append(row)

    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'[保存] CSV 已写入: {RESULTS_CSV}')
    return rows


def load_csv_results():
    """从已有 CSV 恢复 results 字典（供 PLOT_ONLY 模式使用）。"""
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f'未找到结果 CSV: {RESULTS_CSV}')

    results = {}
    with open(RESULTS_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['条件ID']
            if cid not in results:
                results[cid] = {}
            snr_str = row['SNR']
            snr_key = None
            if snr_str != '无噪声':
                try:
                    snr_key = int(snr_str.split()[0])
                except ValueError:
                    snr_key = snr_str
            for col, val in row.items():
                if col in ('条件ID', '条件描述', '信道', '驱动电压(V)', 'SNR'):
                    continue
                try:
                    ber = float(val)
                except (ValueError, TypeError):
                    ber = float('nan')
                model_key = None
                for e in make_model_registry():
                    if e['name'] == col:
                        model_key = e['key']
                        break
                if model_key is None:
                    continue
                if model_key not in results[cid]:
                    results[cid][model_key] = {}
                results[cid][model_key][snr_key] = ber
    return results


def plot_per_condition(conditions, results):
    """每个条件生成一张 BER vs SNR 曲线图。"""
    registry = make_model_registry()
    figs = []

    for cond in conditions:
        cid = cond['id']
        if cid not in results or not results[cid]:
            continue

        x_labels = [snr_label(s) for s in SNR_LIST]
        x_pos    = np.arange(len(x_labels))

        fig, ax = plt.subplots(figsize=(8, 8))
        for entry in registry:
            key = entry['key']
            if key not in results[cid]:
                continue
            bers = [results[cid][key].get(s, np.nan) for s in SNR_LIST]
            ax.semilogy(x_pos, bers,
                        marker=entry['marker'], linestyle=entry['ls'],
                        linewidth=2, markersize=7,
                        color=entry['color'], label=entry['name'])

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=15)
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('BER',      fontsize=12)
        ax.set_title(f'BER vs SNR — {cond["label"]}\n'
                     f'(信道: {cond["fiber"]}, 驱动电压: {cond["voltage"]}V)',
                     fontsize=16)
        ax.legend(fontsize=9, loc='upper right', ncol=2)
        ax.grid(True, which='both', ls='--', alpha=0.55)
        plt.tight_layout()

        fp = FIGURES_DIR / f'ber_vs_snr_{cid}.png'
        plt.savefig(str(fp), dpi=150, bbox_inches='tight')
        plt.close()
        figs.append(str(fp))
        print(f'  [图] {fp.name}')

    return figs


def plot_voltage_sweep(conditions, results, snr_key=None):
    """
    BER vs 驱动电压 曲线图：分 BTB 和 5km 各一张，
    每条曲线对应一种均衡器，横轴为电压。
    snr_key=None 使用"无噪声"结果。
    """
    registry = make_model_registry()
    figs = []

    for fiber in ('BTB', '5km'):
        conds_f = [c for c in conditions if c['fiber'] == fiber]
        if not conds_f:
            continue
        voltages = [c['voltage'] for c in conds_f]

        fig, ax = plt.subplots(figsize=(8, 8))
        has_data = False
        for entry in registry:
            key = entry['key']
            bers = []
            for cond in conds_f:
                cid = cond['id']
                ber = results.get(cid, {}).get(key, {}).get(snr_key, np.nan)
                bers.append(ber)
            if all(np.isnan(b) for b in bers):
                continue
            ax.semilogy(voltages, bers,
                        marker=entry['marker'], linestyle=entry['ls'],
                        linewidth=2, markersize=8,
                        color=entry['color'], label=entry['name'])
            has_data = True

        if not has_data:
            plt.close()
            continue

        ax.set_xlabel('激光器驱动电压 (V)', fontsize=12)
        ax.set_ylabel('BER',               fontsize=12)
        snr_str = '无噪声' if snr_key is None else f'SNR={snr_key}dB'
        fiber_str = 'BTB (无光纤)' if fiber == 'BTB' else '5km SSMF 光纤'
        ax.set_title(f'BER vs 驱动电压 — {fiber_str}  [{snr_str}]\n'
                     f'（电压越大非线性效应越强）', fontsize=16)
        ax.legend(fontsize=9, loc='upper left', ncol=2)
        ax.grid(True, which='both', ls='--', alpha=0.55)
        plt.tight_layout()

        fp = FIGURES_DIR / f'ber_vs_voltage_{fiber}_{snr_str.replace("=","").replace(" ","")}.png'
        plt.savefig(str(fp), dpi=150, bbox_inches='tight')
        plt.close()
        figs.append(str(fp))
        print(f'  [图] {fp.name}')

    return figs


def plot_btb_vs_5km(conditions, results, snr_key=None):
    """
    BTB vs 5km 对比图：每种均衡器一张子图，横轴为电压，
    对比 BTB 和 5km 的 BER 差异（体现光纤色散+非线性的影响）。
    """
    registry = make_model_registry()
    voltages = sorted(set(c['voltage'] for c in conditions))

    fig, axes = plt.subplots(2, 3, figsize=(12, 12))
    axes = axes.flatten()
    snr_str = '无噪声' if snr_key is None else f'SNR={snr_key}dB'

    for idx, entry in enumerate(registry):
        ax  = axes[idx]
        key = entry['key']
        for fiber, ls, marker in [('BTB', '--', 's'), ('5km', '-', 'o')]:
            conds_f = [c for c in conditions if c['fiber'] == fiber]
            vols = [c['voltage'] for c in conds_f]
            bers = [results.get(c['id'], {}).get(key, {}).get(snr_key, np.nan)
                    for c in conds_f]
            label = 'BTB (无光纤)' if fiber == 'BTB' else '5km SSMF'
            ax.semilogy(vols, bers, marker=marker, ls=ls,
                        linewidth=2, markersize=7,
                        color=entry['color'], label=label)

        ax.set_title(entry['name'], fontsize=11)
        ax.set_xlabel('电压 (V)',   fontsize=9)
        ax.set_ylabel('BER',        fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, which='both', ls='--', alpha=0.5)

    plt.suptitle(f'BTB vs 5km SSMF 对比  [{snr_str}]', fontsize=18, y=1.01)
    plt.tight_layout()

    fp = FIGURES_DIR / f'btb_vs_5km_{snr_str.replace("=","").replace(" ","")}.png'
    plt.savefig(str(fp), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [图] {fp.name}')
    return str(fp)


def plot_all_conditions_heatmap(conditions, results, snr_key=None):
    """
    热力图：行=均衡器，列=条件，颜色=log10(BER)。
    直观展示各模型在不同非线性条件下的整体表现。
    """
    registry  = make_model_registry()
    cond_labels = [c['label'] for c in conditions if c['id'] in results]
    model_names = [e['name'] for e in registry]

    mat = np.full((len(model_names), len(cond_labels)), np.nan)
    valid_conds = [c for c in conditions if c['id'] in results]

    for ci, cond in enumerate(valid_conds):
        for mi, entry in enumerate(registry):
            ber = results.get(cond['id'], {}).get(entry['key'], {}).get(snr_key, np.nan)
            if ber > 0:
                mat[mi, ci] = np.log10(ber)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r', vmin=-4, vmax=0)
    plt.colorbar(im, ax=ax, label='log₁₀(BER)')

    ax.set_xticks(range(len(cond_labels)))
    ax.set_xticklabels(cond_labels, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)

    for mi in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[mi, ci]
            txt = f'{10**v:.2e}' if not np.isnan(v) else 'N/A'
            ax.text(ci, mi, txt, ha='center', va='center', fontsize=7,
                    color='black')

    snr_str = '无噪声' if snr_key is None else f'SNR={snr_key}dB'
    ax.set_title(f'BER 热力图 — 各均衡器 × 各非线性条件  [{snr_str}]', fontsize=17)
    plt.tight_layout()

    fp = FIGURES_DIR / f'heatmap_{snr_str.replace("=","").replace(" ","")}.png'
    plt.savefig(str(fp), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [图] {fp.name}')
    return str(fp)


def generate_all_plots(conditions, results):
    """生成全部图形，返回所有 PNG 路径列表。"""
    print('\n[绘图] 开始生成图像...')
    all_figs = []

    # 1. 每个条件的 BER vs SNR 曲线
    all_figs += plot_per_condition(conditions, results)

    # 2. BER vs 驱动电压（无噪声，最能体现非线性影响）
    all_figs += plot_voltage_sweep(conditions, results, snr_key=None)
    if 20 in SNR_LIST:
        all_figs += plot_voltage_sweep(conditions, results, snr_key=20)

    # 3. BTB vs 5km 对比图
    fp = plot_btb_vs_5km(conditions, results, snr_key=None)
    all_figs.append(fp)

    # 4. 全局热力图
    fp = plot_all_conditions_heatmap(conditions, results, snr_key=None)
    all_figs.append(fp)

    print(f'[绘图] 共生成 {len(all_figs)} 张图像，保存于: {FIGURES_DIR}')
    return all_figs


# =============================================================================
#  训练结果摘要邮件 body
# =============================================================================

def _build_train_email_body(train_summary):
    lines = ['非线性均衡器实验 — 训练完成报告', '='*60, '']
    lines.append(f'训练时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'共 {len(train_summary)} 个条件，每条件 6 个模型\n')

    for cond_id, model_bers in train_summary.items():
        lines.append(f'条件: {cond_id}')
        for key, ber in model_bers.items():
            lines.append(f'  {key:15s}  最优ValBER = {ber:.4e}')
        lines.append('')

    lines.append('='*60)
    lines.append('训练完成！程序将继续执行测试阶段。')
    return '\n'.join(lines)


def _build_test_email_body(conditions, results):
    registry = make_model_registry()
    lines = ['非线性均衡器实验 — 测试完成报告', '='*60, '']
    lines.append(f'测试时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'共 {len(conditions)} 个条件，SNR={[snr_label(s) for s in SNR_LIST]}\n')

    lines.append('无噪声条件下 BER 汇总:')
    lines.append('-'*60)
    header = f"{'条件':<22}" + ''.join(f'{e["name"]:>14}' for e in registry)
    lines.append(header)
    lines.append('-'*60)

    for cond in conditions:
        cid = cond['id']
        if cid not in results:
            continue
        row = f"{cond['label']:<22}"
        for entry in registry:
            ber = results[cid].get(entry['key'], {}).get(None, float('nan'))
            row += f'{ber:>14.4e}'
        lines.append(row)

    lines.append('='*60)
    lines.append(f'详细结果见附件 results.csv 和各 PNG 图像。')
    return '\n'.join(lines)


# =============================================================================
#  主入口
# =============================================================================

def main():
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = LOGS_DIR / f'nolinear_main_{ts}.log'
    log      = setup_logger('nolinear_main', log_path, also_stdout=True)

    log.info('='*72)
    log.info('  非线性效应均衡器综合对比实验')
    log.info(f'  TEST_ONLY={TEST_ONLY}  PLOT_ONLY={PLOT_ONLY}')
    log.info(f'  GPU 训练策略: GPU-resident 顺序训练  '
             f'CPU 测试进程: {CPU_TEST_WORKERS or os.cpu_count()}')
    log.info(f'  SNR 列表: {[snr_label(s) for s in SNR_LIST]}')
    log.info('='*72)

    # 检查可用条件（mat 文件是否存在）
    avail_conditions = []
    for cond in CONDITIONS:
        mat_path = MAT_DIR / cond['mat']
        if mat_path.exists():
            avail_conditions.append(cond)
            log.info(f'  [OK] {cond["id"]:22s}  <- {cond["mat"]}')
        else:
            log.warning(f'  [--] {cond["id"]:22s}  未找到 {mat_path}')

    if not avail_conditions:
        log.error('[ERROR] 没有找到任何 .mat 数据文件！'
                  '请先运行 RX_nolinear.m 生成数据。')
        return

    log.info(f'\n可用条件数: {len(avail_conditions)} / {len(CONDITIONS)}\n')

    # ------------------------------------------------------------------
    #  PLOT_ONLY 模式：从 CSV 恢复结果，跳过训练和测试
    # ------------------------------------------------------------------
    if PLOT_ONLY:
        log.info('[PLOT_ONLY] 从已有 CSV 读取结果，直接绘图...')
        results = load_csv_results()
        all_figs = generate_all_plots(avail_conditions, results)
        log.info('[PLOT_ONLY] 绘图完成。')
        send_email(
            subject='非线性均衡器实验 — 重新绘图完成',
            body=f'绘图时间: {datetime.now()}\n共生成 {len(all_figs)} 张图像。',
            attachments=all_figs[:20],   # 附件上限
        )
        return

    # ------------------------------------------------------------------
    #  训练阶段（TEST_ONLY=False 时执行）
    # ------------------------------------------------------------------
    train_summary = {}
    all_train_figs = []

    if not TEST_ONLY:
        log.info('\n' + '='*72)
        log.info('  阶段 1/3：GPU 并行训练')
        log.info('='*72)
        t0 = time.time()
        train_summary, all_train_figs = run_training(avail_conditions, log)
        elapsed = time.time() - t0
        log.info(f'\n[训练] 全部条件训练完成，总耗时 {elapsed/60:.1f} min')

        # 发送训练完成邮件
        send_email(
            subject='非线性均衡器实验 — 训练完成',
            body=_build_train_email_body(train_summary),
            attachments=all_train_figs[:15],
        )
    else:
        log.info('\n[TEST_ONLY] 跳过训练，直接加载已有 checkpoint 进行测试...')

    # ------------------------------------------------------------------
    #  测试阶段
    # ------------------------------------------------------------------
    log.info('\n' + '='*72)
    log.info('  阶段 2/3：CPU 多进程并行测试')
    log.info('='*72)
    t0 = time.time()
    results = run_testing(avail_conditions, log)
    elapsed = time.time() - t0
    log.info(f'\n[测试] 全部测试完成，总耗时 {elapsed/60:.1f} min')

    # ------------------------------------------------------------------
    #  结果保存
    # ------------------------------------------------------------------
    log.info('\n' + '='*72)
    log.info('  阶段 3/3：保存结果与绘图')
    log.info('='*72)
    save_csv(avail_conditions, results)
    all_figs = generate_all_plots(avail_conditions, results)

    # 打印简要汇总表格（无噪声）
    registry = make_model_registry()
    log.info('\n' + '='*72)
    log.info('  无噪声条件下 BER 汇总 (PAM4 硬判决, 阈值 -2/0/2)')
    log.info('='*72)
    header = f"{'条件':<22}" + ''.join(f'{e["name"]:>14}' for e in registry)
    log.info(header)
    log.info('-'*72)
    for cond in avail_conditions:
        cid = cond['id']
        if cid not in results:
            continue
        row = f"{cond['label']:<22}"
        for entry in registry:
            ber = results[cid].get(entry['key'], {}).get(None, float('nan'))
            row += f'{ber:>14.4e}'
        log.info(row)
    log.info('='*72)

    # ------------------------------------------------------------------
    #  发送测试完成邮件（带 CSV + PNG 附件）
    # ------------------------------------------------------------------
    attachments = [str(RESULTS_CSV)] + all_figs
    send_email(
        subject='非线性均衡器实验 — 测试完成，含全部结果',
        body=_build_test_email_body(avail_conditions, results),
        attachments=attachments[:25],   # 邮件附件上限 ~25 个
    )

    log.info(f'\n[完成] 结果 CSV: {RESULTS_CSV}')
    log.info(f'[完成] 图像目录: {FIGURES_DIR}')
    log.info('[完成] 邮件已发送至 2861173454@qq.com')


# Windows spawn 模式要求多进程程序必须在此守卫下启动
if __name__ == '__main__':
    main()
