"""
KAN-Former vs Transformer TEQ 均衡器对比测试
在 SNR = 0, 5, 10, 15, 20 dB 及无噪声条件下并行评估两个模型的 BER。

模型文件:
  - teq_model.pth        (LightweightTransformerEQ, 由 train_teq.py 生成)
  - kan_teq_model.pth    (KANFormerEQ,              由 train_kan_teq.py 生成)
"""

import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader

# 解决 matplotlib 中文乱码：优先使用微软雅黑，回退到黑体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 修复负号显示为方块的问题

# 从训练脚本导入模型定义和数据集
from train_teq import (
    LightweightTransformerEQ,
    OpticalDataset,
    CONFIG as TEQ_CONFIG,
)
from train_kan_teq import (
    KANFormerEQ,
    FastKANLinear,
    KANTransformerEncoderLayer,
    CONFIG as KAN_CONFIG,
)

# ================= 全局配置 =================
# None 表示无噪声；数字为 dB 值
SNR_LIST   = [ 0, 5, 10, 15, 20,None]
MAX_WORKERS = 4          # 并行线程数（GPU 环境推荐 ≤4）
LABEL_SCALE = 3.0        # 与训练时保持一致
BATCH_SIZE  = 1024


# ================= 工具函数 =================
def snr_label(snr):
    return "无噪声" if snr is None else f"{snr} dB"


def add_awgn(rx_signal, snr_db):
    """按指定 SNR(dB) 向接收信号叠加 AWGN 白噪声。"""
    if snr_db is None or np.isinf(snr_db):
        return rx_signal.copy()
    rx   = np.asarray(rx_signal, dtype=np.float64)
    Ps   = np.mean(rx ** 2)
    Pn   = Ps / (10 ** (snr_db / 10))
    sigma = np.sqrt(Pn)
    noise = np.random.RandomState(seed=42).randn(*rx.shape).astype(np.float64) * sigma
    return (rx + noise).astype(np.float64)


def calculate_ber(pred_scaled, true_scaled):
    """
    PAM4 硬判决（判决门限 -2, 0, 2），返回 BER 及预测符号序列。
    pred_scaled / true_scaled 均已还原到 {-3, -1, 1, 3} 量纲。
    """
    pred_labels = np.zeros_like(pred_scaled)
    pred_labels[pred_scaled <  -2] = -3
    pred_labels[(pred_scaled >= -2) & (pred_scaled < 0)] = -1
    pred_labels[(pred_scaled >=  0) & (pred_scaled < 2)] =  1
    pred_labels[pred_scaled >=  2] =  3
    errors = np.sum(pred_labels != true_scaled)
    ber    = errors / len(true_scaled)
    return ber, pred_labels


def _run_inference(model, loader, device):
    """通用推理循环，返回 (preds, targets) numpy 数组（已还原量纲）。"""
    preds_list, targets_list = [], []
    model.eval()
    with torch.no_grad():
        for inputs, tgt in loader:
            inputs = inputs.unsqueeze(-1).to(device)
            out    = model(inputs).cpu().numpy()
            preds_list.append(out)
            targets_list.append(tgt.numpy())
    preds   = np.concatenate(preds_list).flatten()   * LABEL_SCALE
    targets = np.concatenate(targets_list).flatten() * LABEL_SCALE
    return preds, targets


# ================= 单 SNR 测试函数 =================
def run_teq_snr(snr_db, rx_test_base, symb_test, ckpt_path, config, rx_mean, rx_std):
    """Transformer TEQ 在指定 SNR 下的 BER 测试。"""
    rx_test = add_awgn(rx_test_base, snr_db)
    dataset = OpticalDataset(
        rx_test, symb_test,
        config['window_size'], config['sps'],
        rx_mean=rx_mean, rx_std=rx_std,
        label_scale=LABEL_SCALE,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = config['device']
    model  = LightweightTransformerEQ(
        input_dim         = config['window_size'],
        d_model           = config['d_model'],
        nhead             = config['nhead'],
        num_layers        = config['num_layers'],
        dim_feedforward   = config['dim_feedforward'],
        num_passes        = config['num_passes'],
        use_weight_sharing= config['use_weight_sharing'],
        use_center_token  = config['use_center_token'],
        use_sinusoidal_pe = config['use_sinusoidal_pe'],
    ).to(device)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    preds, targets = _run_inference(model, loader, device)
    ber, _         = calculate_ber(preds, targets)
    return snr_db, ber


def run_kan_snr(snr_db, rx_test_base, symb_test, ckpt_path, config, rx_mean, rx_std):
    """KAN-Former 在指定 SNR 下的 BER 测试。"""
    rx_test = add_awgn(rx_test_base, snr_db)
    # OpticalDataset 在 train_kan_teq 中与 train_teq 接口一致，直接复用 train_teq 导入版本
    dataset = OpticalDataset(
        rx_test, symb_test,
        config['window_size'], config['sps'],
        rx_mean=rx_mean, rx_std=rx_std,
        label_scale=LABEL_SCALE,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = config['device']
    model  = KANFormerEQ(
        input_dim         = config['window_size'],
        d_model           = config['d_model'],
        nhead             = config['nhead'],
        num_layers        = config['num_layers'],
        dim_feedforward   = config['dim_feedforward'],
        num_passes        = config['num_passes'],
        use_weight_sharing= config['use_weight_sharing'],
        use_center_token  = config['use_center_token'],
        use_sinusoidal_pe = config['use_sinusoidal_pe'],
        grid_size         = config['kan_grid_size'],
    ).to(device)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    preds, targets = _run_inference(model, loader, device)
    ber, _         = calculate_ber(preds, targets)
    return snr_db, ber


# ================= 主测试流程 =================
def test():
    print("=" * 68)
    print("   KAN-Former  vs  Transformer TEQ  —  BER vs SNR 对比测试")
    print("=" * 68)

    # ---------- 加载测试数据 ----------
    data = scipy.io.loadmat('dataset_for_python.mat')
    rx_test_base = data['rx_test_export'].flatten()
    if np.iscomplexobj(rx_test_base):
        print("[INFO] 检测到复数信号，取绝对值作为 IM/DD 包络。")
        rx_test_base = np.abs(rx_test_base).astype(np.float64)
    else:
        rx_test_base = rx_test_base.astype(np.float64)
    symb_test = data['symb_test_export'].flatten()

    print(f"[INFO] 测试样本数 (接收信号点数): {len(rx_test_base):,}")
    print(f"[INFO] 测试符号数:               {len(symb_test):,}")

    # ---------- 加载 checkpoint 中的归一化参数 ----------
    teq_ckpt = torch.load('teq_model.pth',     weights_only=False,
                          map_location='cpu')
    kan_ckpt = torch.load('kan_teq_model.pth', weights_only=False,
                          map_location='cpu')

    teq_mean = float(teq_ckpt['rx_mean'])
    teq_std  = float(teq_ckpt['rx_std'])
    kan_mean = float(kan_ckpt['rx_mean'])
    kan_std  = float(kan_ckpt['rx_std'])

    print(f"\n[INFO] TEQ     归一化参数: mean={teq_mean:.4f}, std={teq_std:.4f}")
    print(f"[INFO] KAN-Former 归一化参数: mean={kan_mean:.4f}, std={kan_std:.4f}")
    print(f"[INFO] 测试 SNR 条件: {[snr_label(s) for s in SNR_LIST]}")
    print(f"[INFO] 并行线程数: {MAX_WORKERS}\n")

    teq_config = dict(TEQ_CONFIG)
    kan_config = dict(KAN_CONFIG)

    # ---------- 并行提交所有 SNR 测试任务 ----------
    teq_results = {}
    kan_results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {}

        for snr in SNR_LIST:
            ft = executor.submit(
                run_teq_snr, snr, rx_test_base, symb_test,
                'teq_model.pth', teq_config, teq_mean, teq_std,
            )
            future_map[ft] = ('TEQ', snr)

            fk = executor.submit(
                run_kan_snr, snr, rx_test_base, symb_test,
                'kan_teq_model.pth', kan_config, kan_mean, kan_std,
            )
            future_map[fk] = ('KAN', snr)

        for future in as_completed(future_map):
            model_tag, snr = future_map[future]
            try:
                _, ber = future.result()
                (teq_results if model_tag == 'TEQ' else kan_results)[snr] = ber
                print(f"  [{model_tag:3s}] SNR={snr_label(snr):8s}  →  BER = {ber:.4e}")
            except Exception as exc:
                print(f"  [{model_tag:3s}] SNR={snr_label(snr):8s}  →  错误: {exc}")

    # ---------- 汇总打印 ----------
    print("\n" + "=" * 68)
    print("           BER 对比汇总（PAM4 硬判决，门限 -2 / 0 / 2）")
    print("=" * 68)
    header = f"{'SNR':^12} | {'Transformer TEQ':^20} | {'KAN-Former':^20} | {'改善幅度':^12}"
    print(header)
    print("-" * 68)
    for snr in SNR_LIST:
        tb  = teq_results.get(snr, float('nan'))
        kb  = kan_results.get(snr, float('nan'))
        if not np.isnan(tb) and not np.isnan(kb) and kb > 0:
            improve = (tb - kb) / tb * 100
            imp_str = f"{improve:+.1f}%"
        else:
            imp_str = "  N/A"
        print(f"{snr_label(snr):^12} | {tb:^20.4e} | {kb:^20.4e} | {imp_str:^12}")
    print("=" * 68)
    print("  改善幅度 = (TEQ_BER - KAN_BER) / TEQ_BER × 100%，正值表示 KAN 更优。")

    # ---------- 绘图：BER vs SNR 半对数坐标 ----------
    teq_bers = [teq_results.get(s, np.nan) for s in SNR_LIST]
    kan_bers = [kan_results.get(s, np.nan) for s in SNR_LIST]
    x_labels = [snr_label(s) for s in SNR_LIST]
    x_pos    = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(x_pos, teq_bers, 's--', linewidth=2, markersize=8,
                color='C1', label='Transformer TEQ (基准)')
    ax.semilogy(x_pos, kan_bers, 'o-',  linewidth=2, markersize=8,
                color='C0', label='KAN-Former (创新)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER',      fontsize=12)
    ax.set_title('KAN-Former vs Transformer TEQ — BER vs SNR', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_fig = 'kan_ber_comparison.png'
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[INFO] BER 对比图已保存: {out_fig}")


if __name__ == '__main__':
    test()
