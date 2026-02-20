import torch
import scipy.io
import numpy as np
from train_pit import LightweightTransformerEQ, CONFIG, OpticalDataset
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
    return evm * 100 # percentage

def test():
    device = CONFIG['device']
    label_scale = 3.0

    # 1. 加载数据
    data = scipy.io.loadmat('dataset_for_python.mat')
    rx_test = data['rx_test_export'].flatten()
    if np.iscomplexobj(rx_test):
        rx_test = np.abs(rx_test)
    symb_test = data['symb_test_export'].flatten()

    # 2. 加载模型和训练时的归一化参数
    checkpoint = torch.load('pit_model.pth', weights_only=False)
    rx_mean = checkpoint['rx_mean']
    rx_std = checkpoint['rx_std']
    print(f"加载训练集统计量: rx_mean={rx_mean:.4f}, rx_std={rx_std:.4f}")

    model = LightweightTransformerEQ(
        input_dim=CONFIG['window_size'],
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dim_feedforward=CONFIG['dim_feedforward'],
        num_passes=CONFIG['num_passes'],
        use_weight_sharing=CONFIG['use_weight_sharing'],
        use_center_token=CONFIG['use_center_token'],
        use_sinusoidal_pe=CONFIG['use_sinusoidal_pe'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. 准备数据 (复用训练集的全局统计量, 标签同样归一化)
    test_dataset = OpticalDataset(rx_test, symb_test, CONFIG['window_size'], CONFIG['sps'],
                                  rx_mean=rx_mean, rx_std=rx_std, label_scale=label_scale)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

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

    # 4. 反归一化回原始 PAM4 电平再算指标
    all_preds_raw = all_preds * label_scale
    all_targets_raw = all_targets * label_scale

    ber, decisions = calculate_ber(all_preds_raw, all_targets_raw)
    evm = calculate_evm(all_preds_raw, all_targets_raw)
    mse_normalized = np.mean((all_preds - all_targets) ** 2)
    mse_raw = np.mean((all_preds_raw - all_targets_raw) ** 2)
    
    print(f"\n========== 测试结果 ==========")
    print(f"测试样本数: {len(all_preds)}")
    print(f"MSE (归一化标签): {mse_normalized:.6f}")
    print(f"MSE (原始标签):   {mse_raw:.6f}")
    print(f"BER (Bit Error Rate): {ber:.2e}")
    print(f"EVM (Error Vector Magnitude): {evm:.2f}%")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(all_preds_raw, bins=100)
    plt.title('Received Histogram (After Transformer)')
    plt.subplot(1, 2, 2)
    plt.scatter(all_targets_raw[:500], all_preds_raw[:500], alpha=0.5)
    plt.xlabel('Transmitted Symbol')
    plt.ylabel('Recovered Symbol')
    plt.title('Scatter Plot')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    test()