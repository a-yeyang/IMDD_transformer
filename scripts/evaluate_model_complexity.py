# -*- coding: utf-8 -*-
"""
评估各均衡器模型的硬件相关复杂度：
- 模型参数数量
- 乘法器数量（每样本前向推理的乘法运算次数，即 MAC 相关）
- 查找表（LUT）相关数量（需查表实现的非线性单元数量）
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent

# --------------- 从各训练脚本复用的模型与配置 ---------------
def get_dnn_model():
    from train_dnn import DNNEqualizer, CONFIG
    return DNNEqualizer(
        input_dim=CONFIG['window_size'],
        hidden_dims=CONFIG['hidden_dims'],
        use_batchnorm=CONFIG['use_batchnorm'],
        dropout=CONFIG['dropout'],
    ), 'DNN'

def get_fcnn_model():
    from train_fcnn import FCNNEqualizer, CONFIG
    return FCNNEqualizer(
        input_dim=CONFIG['window_size'],
        hidden_dims=CONFIG['hidden_dims'],
    ), 'FCNN'

def get_bilstm_model():
    from train_bilstm import BiLSTMEqualizer, CONFIG
    return BiLSTMEqualizer(
        input_size=1,
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_lstm_layers'],
        use_center=CONFIG['use_center'],
        window_size=CONFIG['window_size'],
    ), 'BiLSTM'

def get_teq_model():
    from train_teq import LightweightTransformerEQ, CONFIG
    return LightweightTransformerEQ(
        input_dim=CONFIG['window_size'],
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dim_feedforward=CONFIG['dim_feedforward'],
        num_passes=CONFIG['num_passes'],
        use_weight_sharing=CONFIG['use_weight_sharing'],
        use_center_token=CONFIG['use_center_token'],
        use_sinusoidal_pe=CONFIG['use_sinusoidal_pe'],
    ), 'Transformer TEQ'

def get_kan_teq_model():
    from train_kan_teq import KANFormerEQ, CONFIG
    return KANFormerEQ(
        input_dim=CONFIG['window_size'],
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dim_feedforward=CONFIG['dim_feedforward'],
        num_passes=CONFIG['num_passes'],
        use_weight_sharing=CONFIG['use_weight_sharing'],
        use_center_token=CONFIG['use_center_token'],
        use_sinusoidal_pe=CONFIG['use_sinusoidal_pe'],
        grid_size=CONFIG['kan_grid_size'],
    ), 'KAN-Former TEQ'


def count_params(model):
    """总参数量（标量个数）"""
    return sum(p.numel() for p in model.parameters())


def count_weight_params(model):
    """仅权重参数量（不含 bias），用于估算乘法次数下界"""
    n = 0
    for name, p in model.named_parameters():
        if 'bias' not in name:
            n += p.numel()
    return n


def count_multipliers_and_luts(model, seq_len=21):
    """
    估算单样本前向推理的乘法次数与查找表相关数量。
    - 乘法器数量：权值参与的乘法次数（矩阵乘 in*out + BN/LN 的 scale）。
    - 查找表数量：需查表实现的非线性单元数（sigmoid/tanh/sin/softmax），供 FPGA 估算。
    """
    mults = 0
    luts = 0

    def add_linear(in_f, out_f):
        nonlocal mults
        mults += in_f * out_f

    def add_batchnorm1d(c):
        nonlocal mults
        mults += 2 * c  # scale and shift

    def add_layernorm(d):
        nonlocal mults
        mults += 2 * d

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            add_linear(mod.in_features, mod.out_features)
        elif isinstance(mod, nn.BatchNorm1d):
            add_batchnorm1d(mod.num_features)
        elif isinstance(mod, nn.LayerNorm):
            add_layernorm(mod.normalized_shape[0])
        elif isinstance(mod, nn.LSTM):
            d, h = mod.input_size, mod.hidden_size
            n_layers = mod.num_layers
            # 每层每方向: 4 门，每门 (in+h)*h 乘法
            mult_per_dir = 4 * (d + h) * h
            mults += n_layers * (2 if mod.bidirectional else 1) * mult_per_dir
            # 每时间步每方向: 4 sigmoid + 1 tanh
            n_dir = 2 if mod.bidirectional else 1
            luts += n_layers * n_dir * seq_len * 5
        elif isinstance(mod, nn.MultiheadAttention):
            embed_dim = mod.embed_dim
            num_heads = mod.num_heads
            head_dim = embed_dim // num_heads
            # Q,K,V,O 四个投影
            mults += 4 * embed_dim * embed_dim
            # Q@K^T: (seq, head_dim) @ (head_dim, seq) -> seq*seq 每 head，共 num_heads
            mults += num_heads * seq_len * seq_len * head_dim * 2  # QK^T + attn@V
            luts += num_heads * seq_len * seq_len  # softmax 按元素/头
        elif type(mod).__name__ == 'FastKANLinear':
            in_f, out_f = mod.in_features, mod.out_features
            g = getattr(mod, 'grid_size', 3)
            add_linear(in_f, out_f)
            mults += out_f * in_f * g  # spline 基与权
            luts += in_f + out_f  # SiLU
            luts += in_f * g      # sin(i*x)

    return mults, luts


def run_one(get_model_fn):
    try:
        model, label = get_model_fn()
    except Exception as e:
        return {'name': get_model_fn.__name__.replace('get_', '').replace('_model', ''), 'error': str(e)}
    model.eval()
    seq_len = 21
    n_params = count_params(model)
    n_weight = count_weight_params(model)
    mults, luts = count_multipliers_and_luts(model, seq_len=seq_len)
    return {
        'name': label,
        'params': n_params,
        'weight_params': n_weight,
        'multipliers': mults,
        'luts': luts,
    }


def main():
    import sys
    import os
    sys.path.insert(0, str(ROOT / 'scripts'))
    os.chdir(str(ROOT / 'scripts'))

    results = []
    for get_fn in [get_dnn_model, get_fcnn_model, get_bilstm_model, get_teq_model, get_kan_teq_model]:
        r = run_one(get_fn)
        if 'error' in r:
            print(f"{r['name']}: Error - {r['error']}")
            continue
        results.append(r)

    # 打印表格
    print("\n" + "=" * 80)
    print("  光纤通信均衡器 — 模型复杂度评估（参数数量 / 乘法器数量 / 查找表相关数量）")
    print("=" * 80)
    print(f"{'模型':<22} {'参数量':>12} {'权重参数量':>12} {'乘法运算数(次/样本)':>22} {'查找表相关数':>14}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<22} {r['params']:>12,} {r['weight_params']:>12,} {r['multipliers']:>22,} {r['luts']:>14,}")
    print("=" * 80)
    print("\n说明：")
    print("  - 参数量：模型总标量参数（含 bias）。")
    print("  - 权重参数量：仅权重，不含 bias，与乘法次数同量级。")
    print("  - 乘法运算数：单样本前向推理中权值参与的乘法次数（近似 MAC 量级）。")
    print("  - 查找表相关数：需查表实现的非线性单元数（sigmoid/tanh/sin/softmax 等），供 FPGA 估算 LUT 用量。")
    print()
    return results


if __name__ == '__main__':
    import os
    main()
