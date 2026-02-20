import torch
import scipy.io
import numpy as np
import os
from torch.utils.data import DataLoader
from train_teq import LightweightTransformerEQ, OpticalDataset, SinusoidalPE, count_parameters
from train_kan import KANTransformerEQ

LABEL_SCALE = 3.0

# ================= 待对比的模型列表 =================
MODELS_TO_TEST = [
    {'name': 'Transformer-EQ',     'checkpoint': 'teq_model.pth'},
    {'name': 'KAN-Transformer-EQ', 'checkpoint': 'kan_model.pth'},
]

# ================= 指标计算 =================
def calculate_ber(pred, true_labels):
    thresholds = [-2, 0, 2]
    levels = [-3, -1, 1, 3]
    pred_labels = np.zeros_like(pred)
    pred_labels[pred < thresholds[0]] = levels[0]
    pred_labels[(pred >= thresholds[0]) & (pred < thresholds[1])] = levels[1]
    pred_labels[(pred >= thresholds[1]) & (pred < thresholds[2])] = levels[2]
    pred_labels[pred >= thresholds[2]] = levels[3]
    errors = np.sum(pred_labels != true_labels)
    return errors / len(true_labels), pred_labels


def calculate_evm(pred, true_labels):
    error_power = np.sum((pred - true_labels) ** 2)
    ref_power = np.sum(true_labels ** 2)
    return np.sqrt(error_power / ref_power) * 100


# ================= 模型工厂 =================
def create_model_from_checkpoint(checkpoint, device):
    """根据 checkpoint 中保存的 model_type 和 config 自动重建模型。"""
    config = checkpoint['config']
    model_type = checkpoint.get('model_type', 'TEQ')

    if model_type == 'TEQ':
        model = LightweightTransformerEQ(
            input_dim=config['window_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config.get('num_layers', 1),
            dim_feedforward=config['dim_feedforward'],
            num_passes=config['num_passes'],
            use_weight_sharing=config['use_weight_sharing'],
            use_center_token=config['use_center_token'],
            use_sinusoidal_pe=config['use_sinusoidal_pe'],
        )
    elif model_type == 'KAN':
        model = KANTransformerEQ(
            input_dim=config['window_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            grid_size=config['grid_size'],
            spline_order=config['spline_order'],
            num_passes=config['num_passes'],
            use_weight_sharing=config['use_weight_sharing'],
            use_center_token=config['use_center_token'],
            use_sinusoidal_pe=config['use_sinusoidal_pe'],
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device), config


def get_param_count(model):
    return sum(p.numel() for p in model.parameters())


# ================= 单模型推理 =================
def run_inference(model, test_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.unsqueeze(-1).to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
    return np.concatenate(all_preds).flatten(), np.concatenate(all_targets).flatten()


# ================= 主测试流程 =================
def test():
    import matplotlib.pyplot as plt

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}\n")

    # 1. 发现可用模型
    available = []
    for m in MODELS_TO_TEST:
        if os.path.exists(m['checkpoint']):
            available.append(m)
        else:
            print(f"[跳过] {m['name']}: 未找到 {m['checkpoint']}")
    if not available:
        print("错误: 没有找到任何模型 checkpoint，请先训练。")
        return

    # 2. 加载测试数据
    data = scipy.io.loadmat('dataset_for_python.mat')
    rx_test = data['rx_test_export'].flatten()
    if np.iscomplexobj(rx_test):
        rx_test = np.abs(rx_test)
    symb_test = data['symb_test_export'].flatten()

    # 3. 逐模型测试
    results = []
    model_preds = {}
    model_targets = {}

    for m in available:
        print(f"{'='*50}")
        print(f"测试模型: {m['name']}")
        print(f"{'='*50}")

        ckpt = torch.load(m['checkpoint'], weights_only=False)
        rx_mean = float(ckpt['rx_mean'])
        rx_std = float(ckpt['rx_std'])
        print(f"  归一化参数: mean={rx_mean:.4f}, std={rx_std:.4f}")

        model, config = create_model_from_checkpoint(ckpt, device)
        n_params = get_param_count(model)
        print(f"  参数量: {n_params:,}")

        test_dataset = OpticalDataset(
            rx_test, symb_test, config['window_size'], config['sps'],
            rx_mean=rx_mean, rx_std=rx_std, label_scale=LABEL_SCALE
        )
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        preds_norm, targets_norm = run_inference(model, test_loader, device)

        preds_raw = preds_norm * LABEL_SCALE
        targets_raw = targets_norm * LABEL_SCALE

        ber, decisions = calculate_ber(preds_raw, targets_raw)
        evm = calculate_evm(preds_raw, targets_raw)
        mse_norm = float(np.mean((preds_norm - targets_norm) ** 2))
        mse_raw = float(np.mean((preds_raw - targets_raw) ** 2))

        results.append({
            'name': m['name'], 'params': n_params,
            'mse_norm': mse_norm, 'mse_raw': mse_raw,
            'ber': ber, 'evm': evm, 'n_samples': len(preds_raw),
        })
        model_preds[m['name']] = preds_raw
        model_targets[m['name']] = targets_raw

        print(f"  BER: {ber:.2e}  |  EVM: {evm:.2f}%  |  MSE(归一化): {mse_norm:.6f}")

    # 4. 对比结果汇总表
    print(f"\n{'='*80}")
    print("模型对比结果汇总")
    print(f"{'='*80}")
    header = f"{'模型':^22} | {'参数量':^8} | {'MSE(归一化)':^12} | {'MSE(原始)':^12} | {'BER':^12} | {'EVM(%)':^8}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r['name']:^22} | {r['params']:^8,} | {r['mse_norm']:^12.6f} | "
              f"{r['mse_raw']:^12.6f} | {r['ber']:^12.2e} | {r['evm']:^8.2f}")
    print(f"{'='*80}")

    if len(results) == 2:
        r0, r1 = results[0], results[1]
        ber_change = (r1['ber'] - r0['ber']) / (r0['ber'] + 1e-15) * 100
        evm_change = (r1['evm'] - r0['evm']) / (r0['evm'] + 1e-15) * 100
        param_change = (r1['params'] - r0['params']) / r0['params'] * 100
        print(f"\n{r1['name']} 相对于 {r0['name']}:")
        print(f"  参数量变化: {param_change:+.1f}%")
        print(f"  BER  变化: {ber_change:+.1f}%  ({'改善' if ber_change < 0 else '恶化'})")
        print(f"  EVM  变化: {evm_change:+.1f}%  ({'改善' if evm_change < 0 else '恶化'})")

    # 5. 对比可视化
    n_models = len(results)
    if n_models == 0:
        return

    fig = plt.figure(figsize=(16, 10))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    names = [r['name'] for r in results]

    # (a) 参数量 & BER 柱状图
    ax1 = fig.add_subplot(2, 3, 1)
    x = np.arange(n_models)
    bars = ax1.bar(x, [r['params'] for r in results], color=colors[:n_models], width=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel('Parameter Count')
    ax1.set_title('(a) Parameters')
    for bar, r in zip(bars, results):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{r["params"]:,}', ha='center', va='bottom', fontsize=8)

    ax2 = fig.add_subplot(2, 3, 2)
    bars = ax2.bar(x, [r['ber'] for r in results], color=colors[:n_models], width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel('BER')
    ax2.set_title('(b) Bit Error Rate')
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-3, -3))

    ax3 = fig.add_subplot(2, 3, 3)
    bars = ax3.bar(x, [r['evm'] for r in results], color=colors[:n_models], width=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, fontsize=8)
    ax3.set_ylabel('EVM (%)')
    ax3.set_title('(c) Error Vector Magnitude')

    # (d)(e) 各模型的直方图
    for i, name in enumerate(names):
        ax = fig.add_subplot(2, n_models, n_models + 1 + i)
        ax.hist(model_preds[name], bins=100, color=colors[i], alpha=0.8, density=True)
        for lvl in [-3, -1, 1, 3]:
            ax.axvline(lvl, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(f'({"defgh"[i]}) {name}\nHistogram', fontsize=9)
        ax.set_xlabel('Predicted Level')
        ax.set_ylabel('Density')

    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存: comparison_results.png")
    plt.show()


if __name__ == '__main__':
    test()
