import torch
import scipy.io
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from train_teq import LightweightTransformerEQ, CONFIG, OpticalDataset
from torch.utils.data import DataLoader


def calculate_ber(pred, true_labels):
    # PAM4 硬判决
    # 标准电平: -3, -1, 1, 3
    # 判决门限: -2, 0, 2
    pred_labels = np.zeros_like(pred)
    pred_labels[pred < -2] = -3
    pred_labels[(pred >= -2) & (pred < 0)] = -1
    pred_labels[(pred >= 0) & (pred < 2)] = 1
    pred_labels[pred >= 2] = 3

    errors = np.sum(pred_labels != true_labels)
    total = len(true_labels)
    return errors / total, pred_labels


def calculate_evm(pred, true_labels):
    # Error Vector Magnitude
    # EVM_RMS = sqrt( sum(|y_pred - y_true|^2) / sum(|y_true|^2) )
    error_power = np.sum((pred - true_labels)**2)
    ref_power = np.sum(true_labels**2)
    evm = np.sqrt(error_power / ref_power)
    return evm * 100  # percentage


def add_awgn(rx_signal, snr_db):
    """
    在接收信号上添加 AWGN，达到指定 SNR (dB)。
    SNR = 10*log10(Ps/Pn)，Ps 为信号功率，Pn 为噪声功率。
    """
    if snr_db is None or np.isinf(snr_db):
        return rx_signal.copy()
    rx = np.asarray(rx_signal, dtype=np.float64)
    Ps = np.mean(rx ** 2)  # 信号功率
    # Pn = Ps / 10^(SNR/10)
    Pn = Ps / (10 ** (snr_db / 10))
    sigma = np.sqrt(Pn)
    noise = np.random.randn(*rx.shape).astype(np.float64) * sigma
    return (rx + noise).astype(np.float64)


def run_single_snr_test(snr_db, rx_test_base, symb_test, checkpoint_path, config, rx_mean, rx_std):
    """
    单 SNR 条件下的测试，供多进程调用。
    返回 (snr_db, ber, evm, mse_normalized, mse_raw, n_samples)
    """
    device = config['device']
    label_scale = 3.0
    np.random.seed(42)  # 保证可复现，但每个进程不同

    # 加噪
    rx_test = add_awgn(rx_test_base, snr_db)

    # 构建数据集和 DataLoader
    test_dataset = OpticalDataset(
        rx_test, symb_test, config['window_size'], config['sps'],
        rx_mean=rx_mean, rx_std=rx_std, label_scale=label_scale
    )
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # 加载模型
    model = LightweightTransformerEQ(
        input_dim=config['window_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        num_passes=config['num_passes'],
        use_weight_sharing=config['use_weight_sharing'],
        use_center_token=config['use_center_token'],
        use_sinusoidal_pe=config['use_sinusoidal_pe'],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.unsqueeze(-1).to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    all_preds_raw = all_preds * label_scale
    all_targets_raw = all_targets * label_scale

    ber, _ = calculate_ber(all_preds_raw, all_targets_raw)
    evm = calculate_evm(all_preds_raw, all_targets_raw)
    mse_normalized = np.mean((all_preds - all_targets) ** 2)
    mse_raw = np.mean((all_preds_raw - all_targets_raw) ** 2)

    return snr_db, ber, evm, mse_normalized, mse_raw, len(all_preds)


# 多 SNR 测试: 自由添加/删除 SNR，None 表示无噪声，数字为 dB 值
SNR_LIST = [None, 0, 5, 10, 15, 20, 25]
# 最大并行进程数 (自适应为 min(本值, SNR个数)；若 GPU 显存不足可改为 1)
MAX_WORKERS = 6


def _snr_to_str(snr):
    """将 SNR 值转为显示字符串"""
    return "无噪声" if snr is None else f"{snr} dB"


def test():
    import matplotlib.pyplot as plt

    n_snr = len(SNR_LIST)
    n_workers = min(MAX_WORKERS, n_snr)
    snr_display = ", ".join(_snr_to_str(s) for s in SNR_LIST)

    # 1. 加载数据和 checkpoint 统计量
    data = scipy.io.loadmat('dataset_for_python.mat')
    rx_test_base = data['rx_test_export'].flatten()
    if np.iscomplexobj(rx_test_base):
        rx_test_base = np.abs(rx_test_base).astype(np.float64)
    symb_test = data['symb_test_export'].flatten()

    checkpoint = torch.load('teq_model.pth', weights_only=False)
    rx_mean = float(checkpoint['rx_mean'])
    rx_std = float(checkpoint['rx_std'])
    print(f"加载训练集统计量: rx_mean={rx_mean:.4f}, rx_std={rx_std:.4f}")
    print(f"SNR 条件 ({n_snr} 个): {snr_display}")
    print(f"并行进程数: {n_workers}")

    config = dict(CONFIG)
    checkpoint_path = 'teq_model.pth'

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                run_single_snr_test,
                snr_db, rx_test_base, symb_test,
                checkpoint_path, config, rx_mean, rx_std
            ): snr_db
            for snr_db in SNR_LIST
        }
        for future in as_completed(futures):
            snr_label = futures[future]
            try:
                row = future.result()
                results.append(row)
                snr_str = _snr_to_str(row[0])
                print(f"  完成: SNR={snr_str}, BER={row[1]:.2e}, EVM={row[2]:.2f}%")
            except Exception as e:
                print(f"  SNR={snr_label} 测试失败: {e}")

    # 按 SNR 排序 (无噪声排最前)
    results.sort(key=lambda r: (-np.inf if r[0] is None else r[0]))

    # 汇总表格
    print("\n" + "=" * 70)
    print("多 SNR 测试汇总")
    print("=" * 70)
    print(f"{'SNR':^12} | {'样本数':^8} | {'MSE(归一化)':^12} | {'MSE(原始)':^12} | {'BER':^12} | {'EVM(%)':^8}")
    print("-" * 70)
    for snr_db, ber, evm, mse_n, mse_r, n in results:
        snr_str = _snr_to_str(snr_db)
        print(f"{snr_str:^12} | {n:^8} | {mse_n:^12.6f} | {mse_r:^12.6f} | {ber:^12.2e} | {evm:^8.2f}")
    print("=" * 70)

    # BER / EVM 随 SNR 曲线
    snr_labels = [_snr_to_str(r[0]).replace(" dB", "") for r in results]  # 图例用简短标签
    bers = [r[1] for r in results]
    evms = [r[2] for r in results]
    x_pos = np.arange(len(snr_labels))

    rot = 45 if n_snr > 6 else 0  # SNR 多时旋转标签防重叠
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.semilogy(x_pos, bers, 'o-', linewidth=2, markersize=8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(snr_labels, rotation=rot, ha='right' if rot else 'center')
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('BER')
    ax1.set_title('BER vs SNR')
    ax1.grid(True, which='both')

    ax2.plot(x_pos, evms, 's-', color='C1', linewidth=2, markersize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(snr_labels, rotation=rot, ha='right' if rot else 'center')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('EVM (%)')
    ax2.set_title('EVM vs SNR')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('snr_comparison.png', dpi=150, bbox_inches='tight')
    print("\n曲线图已保存: snr_comparison.png")
    plt.show()

if __name__ == '__main__':
    test()