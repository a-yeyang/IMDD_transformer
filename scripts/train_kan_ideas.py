"""
KAN 均衡器变体 — 训练脚本

本脚本探索八种将 KAN 与各类神经网络结构结合的方案：

  方案 0: VanillaKAN     — 纯 KAN 两层网络，不含任何辅助模块 (基准)
  方案 1: KAN-FCNN       — 纯 KAN 前馈网络 (KANLinear 替代 Linear+ReLU)
  方案 2: Hybrid-KAN     — FCNN 前端线性特征提取 + KAN 后端非线性映射
  方案 3: ResKAN         — FCNN 主路径 + KAN 残差校正分支
  方案 4: ConvKAN        — 1D 卷积前端局部 ISI 特征提取 + KAN 后端非线性回归
  方案 5: KAN-Attention  — 注意力池化 KAN，动态加权输入窗口各采样点
  方案 6: MultiScale-KAN — 多尺度并行 KAN，不同 B-样条分辨率自适应融合
  方案 7: GatedKAN       — 门控双路径 KAN，自适应选择线性/非线性处理

参数量对齐机制 (MATCH_PARAMS 开关):
  打开后，各模型自动微调隐层维度使总参数量对齐到 FCNN 基准。
  用户只需修改 BASE 中的基本参数 (window_size, fcnn_hidden_dims,
  grid_size, spline_order)，各模型的内部维度会自动跟随调整。

运行方式:
  python train_kan_ideas.py                    # 顺序训练全部 8 个模型
  python train_kan_ideas.py vanilla_kan        # 仅训练 VanillaKAN
  python train_kan_ideas.py conv_kan kan_attn  # 训练指定的多个模型
  可用 key: vanilla_kan / kan_fcnn / hybrid_kan / res_kan / conv_kan /
            kan_attn / multiscale_kan / gated_kan
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from pathlib import Path
from datetime import datetime

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / 'models'
IMAGES_DIR = ROOT / 'images'
LOGS_DIR   = ROOT / 'logs'
MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# ================= KAN 线性层 =================
class KANLinear(nn.Module):
    """
    KAN (Kolmogorov-Arnold Network) 线性层。

    每条边 (i->j) 上放置一个可学习单变量函数：
        phi_{j,i}(x_i) = w_b_{j,i} * SiLU(x_i) + sum_l c_{j,i,l} * B_l^k(x_i)

    参数:
        in_features   – 输入维度
        out_features  – 输出维度
        grid_size     – B-样条均匀网格区间数 G
        spline_order  – B-样条阶数 k (3 = 三次)
        grid_range    – 网格覆盖范围 [a, b]

    内部张量:
        grid          – (in, G+2k+1) 含两端扩展的节点向量
        spline_weight – (out, in, G+k) 样条系数
        base_weight   – (out, in) SiLU 基权重
    """

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 grid_range=(-2.0, 2.0)):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self.num_bases     = grid_size + spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - h * spline_order,
            grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer('grid', grid.unsqueeze(0).expand(in_features, -1).contiguous())

        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, self.num_bases) * 0.1
        )
        self.base_weight = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(out_features, in_features), a=5**0.5)
        )

    def _bspline_bases(self, x):
        """
        向量化 de Boor 递推，计算 k 阶 B-样条基。
        x:       (..., in_features)
        Returns: (..., in_features, num_bases)
        """
        x_expand = x.unsqueeze(-1)

        bases = ((x_expand >= self.grid[..., :-1]) &
                 (x_expand <  self.grid[..., 1:])).to(x.dtype)

        for p in range(1, self.spline_order + 1):
            left_num  = x_expand - self.grid[..., :-(p + 1)]
            left_den  = (self.grid[..., p:-1] - self.grid[..., :-(p + 1)]).clamp(min=1e-8)
            right_num = self.grid[..., (p + 1):] - x_expand
            right_den = (self.grid[..., (p + 1):] - self.grid[..., 1:(-p)]).clamp(min=1e-8)

            bases = (left_num / left_den) * bases[..., :-1] \
                  + (right_num / right_den) * bases[..., 1:]

        return bases

    def forward(self, x):
        """x: (..., in_features) -> (..., out_features)"""
        base_out = F.linear(F.silu(x), self.base_weight)

        bases = self._bspline_bases(x)
        batch_shape = x.shape[:-1]
        flat_bases  = bases.reshape(-1, self.in_features * self.num_bases)
        flat_weight = self.spline_weight.reshape(self.out_features, -1)
        spline_out  = F.linear(flat_bases, flat_weight).reshape(*batch_shape, self.out_features)

        return base_out + spline_out


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# ╔═══════════════════════════════════════════════════════════╗
# ║                  参数量对齐开关 & 基准配置                    ║
# ╚═══════════════════════════════════════════════════════════╝
#
# MATCH_PARAMS = True  时：
#   1. 以 FCNN(BASE['fcnn_hidden_dims']) 的参数量为对齐目标
#   2. 各模型自动搜索 hidden_dims 使总参数量 ≈ 目标
#   3. 修改 BASE 中任何参数后，所有模型自动重新微调
#
# MATCH_PARAMS = False 时：
#   使用各模型手动指定的默认隐层维度

MATCH_PARAMS = True

BASE = {
    'window_size':      21,
    'sps':              2,
    'fcnn_hidden_dims': [64, 32],   # ← FCNN 基准隐层，决定目标参数量
    'grid_size':        5,          # ← B-样条网格区间数
    'spline_order':     3,          # ← 样条阶数
    'grid_range':       (-2.0, 2.0),
    'batch_size':       256,
    'lr':               0.001,
    'label_scale':      3.0,
    'eval_interval':    1,
}


# ================= 参数量计算工具 =================

def _fcnn_param_count(input_dim, hidden_dims):
    """FCNN (Linear+bias 逐层) 的精确参数量"""
    total, d = 0, input_dim
    for h in hidden_dims:
        total += d * h + h          # Linear(d, h) + bias
        d = h
    total += d + 1                  # 输出层 Linear(d, 1) + bias
    return total


def _kan_factor():
    """每条 KANLinear 连接的参数数 = grid_size + spline_order + 1"""
    return BASE['grid_size'] + BASE['spline_order'] + 1


# ================= 自动微调搜索 =================

def _tune_kan_fcnn(target, W, F):
    """
    搜索 KAN-FCNN 的 [h1, h2] 使参数量最接近 target。

    参数公式:
      KANLinear(W, h1)  : W·h1·F
      LayerNorm(h1)     : 2·h1
      KANLinear(h1, h2) : h1·h2·F
      LayerNorm(h2)     : 2·h2
      KANLinear(h2, 1)  : h2·F
      ──────────────────────────
      Total = F·(W·h1 + h1·h2 + h2) + 2·(h1 + h2)
    """
    best, best_diff = [16, 8], float('inf')
    for h1 in range(2, 128):
        for h2 in range(1, h1 + 1):
            t = F * (W * h1 + h1 * h2 + h2) + 2 * (h1 + h2)
            d = abs(t - target)
            if d < best_diff:
                best_diff = d
                best = [h1, h2]
    return best


def _tune_hybrid_kan(target, W, F):
    """
    搜索 Hybrid-KAN 的 [h1, h2] 使参数量最接近 target。

    参数公式:
      Linear(W, h1)+bias  : W·h1 + h1
      Linear(h1, h2)+bias : h1·h2 + h2
      KANLinear(h2, 1)    : h2·F
      ──────────────────────────
      Total = h1·(W+1) + h2·(h1+1) + h2·F
            = h1·(W+1) + h2·(h1 + 1 + F)
    """
    best, best_diff = [64, 32], float('inf')
    for h1 in range(4, 256):
        for h2 in range(2, h1 + 1):
            t = h1 * (W + 1) + h2 * (h1 + 1 + F)
            d = abs(t - target)
            if d < best_diff:
                best_diff = d
                best = [h1, h2]
    return best


def _tune_res_kan(target, W, F):
    """
    搜索 ResKAN 的 (fcnn_dims=[fh1, fh2], kan_hidden) 使参数量最接近 target。

    参数公式:
      FCNN 主路:
        Linear(W, fh1)+b  : W·fh1 + fh1
        Linear(fh1, fh2)+b: fh1·fh2 + fh2
        Linear(fh2, 1)+b  : fh2 + 1
      KAN 残差:
        KANLinear(W, kh)  : W·kh·F
        KANLinear(kh, 1)  : kh·F
      alpha: 1
      ──────────────────────────
      Total = fh1·(W+1) + fh2·(fh1+2) + 1 + kh·(W+1)·F + 1
    """
    best, best_diff = ([48, 24], 4), float('inf')
    for fh1 in range(4, 128):
        for fh2 in range(2, fh1 + 1):
            fcnn_p = fh1 * (W + 1) + fh2 * (fh1 + 2) + 1
            remaining = target - fcnn_p - 1   # -1 for alpha
            if remaining < F * (W + 1) * 2:   # 至少 kh=2
                continue
            kh_opt = remaining / (F * (W + 1))
            for kh in (max(2, int(kh_opt)), max(2, int(kh_opt) + 1)):
                total = fcnn_p + kh * (W + 1) * F + 1
                d = abs(total - target)
                if d < best_diff:
                    best_diff = d
                    best = ([fh1, fh2], kh)
    return best


def _tune_vanilla_kan(target, W, F):
    """
    搜索 VanillaKAN 的 kan_hidden 使参数量最接近 target。

    参数公式:
      KANLinear(W, h)  : W·h·F
      KANLinear(h, 1)  : h·F
      ──────────────────────────
      Total = (W + 1)·h·F
    """
    best, best_diff = 8, float('inf')
    for h in range(2, 128):
        t = (W + 1) * h * F
        d = abs(t - target)
        if d < best_diff:
            best_diff = d
            best = h
    return best


def _tune_conv_kan(target, W, F, ks=5, po=4):
    """
    搜索 ConvKAN 的 (n_filters, kan_hidden) 使参数量最接近 target。

    参数公式:
      Conv1d(1, nf, ks)+b       : nf·ks + nf
      Conv1d(nf, nf, ks)+b      : nf²·ks + nf
      KANLinear(nf·po, kh)      : nf·po·kh·F
      LayerNorm(kh)             : 2·kh
      KANLinear(kh, 1)          : kh·F
      ──────────────────────────
      Total = nf·(ks+1) + nf²·ks + nf + kh·(nf·po·F + F + 2)
    """
    best, best_diff = (8, 8), float('inf')
    for nf in range(4, 32):
        for kh in range(2, 32):
            t = nf * (ks + 1) + nf * nf * ks + nf + kh * (nf * po * F + F + 2)
            d = abs(t - target)
            if d < best_diff:
                best_diff = d
                best = (nf, kh)
    return best


def _tune_kan_attention(target, F):
    """
    搜索 KANAttention 的 d_model 使参数量最接近 target。

    参数公式:
      Linear(1, d)+b            : 2·d
      KANLinear(d, 1) [attn]    : d·F
      KANLinear(d, d) [feat]    : d²·F
      KANLinear(d, 1) [out]     : d·F
      ──────────────────────────
      Total = d·(2 + 2·F) + d²·F
    """
    best, best_diff = 16, float('inf')
    for d in range(4, 64):
        t = d * (2 + 2 * F) + d * d * F
        dd = abs(t - target)
        if dd < best_diff:
            best_diff = dd
            best = d
    return best


def _tune_multiscale_kan(target, W, k, grid_sizes=(3, 5, 8)):
    """
    搜索 MultiScaleKAN 的 hidden 使参数量最接近 target。

    参数公式 (每个分支 i, F_i = gs_i + k + 1):
      KANLinear(W, h)           : W·h·F_i
      LayerNorm(h)              : 2·h
      KANLinear(h, 1)           : h·F_i
      ──────────────────────────
      Total = Σ_i [h·(W+1)·F_i + 2·h] + n_branches
    """
    n_b = len(grid_sizes)
    best, best_diff = 8, float('inf')
    for h in range(2, 64):
        t = sum(h * (W + 1) * (gs + k + 1) + 2 * h
                for gs in grid_sizes) + n_b
        d = abs(t - target)
        if d < best_diff:
            best_diff = d
            best = h
    return best


def _tune_gated_kan(target, W, F):
    """
    搜索 GatedKAN 的 hidden_dim 使参数量最接近 target。

    参数公式:
      signal KANLinear(W, h)    : W·h·F
      LayerNorm(h)              : 2·h
      gate KANLinear(W, h)      : W·h·F
      bypass Linear(W, h)+b     : W·h + h
      out KANLinear(h, 1)       : h·F
      ──────────────────────────
      Total = W·h·(2·F + 1) + h·(F + 3)
    """
    best, best_diff = 8, float('inf')
    for h in range(2, 64):
        t = W * h * (2 * F + 1) + h * (F + 3)
        d = abs(t - target)
        if d < best_diff:
            best_diff = d
            best = h
    return best


# ================= 配置生成器 =================

def _build_configs():
    """
    根据 MATCH_PARAMS 开关和 BASE 基准，生成八个模型的训练配置。
    MATCH_PARAMS=True 时自动微调隐层维度；False 时使用手动默认值。
    """
    shared = {
        'window_size':  BASE['window_size'],
        'sps':          BASE['sps'],
        'grid_size':    BASE['grid_size'],
        'spline_order': BASE['spline_order'],
        'grid_range':   BASE['grid_range'],
        'batch_size':   BASE['batch_size'],
        'lr':           BASE['lr'],
        'label_scale':  BASE['label_scale'],
        'eval_interval': BASE['eval_interval'],
        'device':       get_device(),
    }

    W = BASE['window_size']
    F = _kan_factor()
    k = BASE['spline_order']

    if MATCH_PARAMS:
        target = _fcnn_param_count(W, BASE['fcnn_hidden_dims'])

        vanilla_hidden             = _tune_vanilla_kan(target, W, F)
        kan_fcnn_dims              = _tune_kan_fcnn(target, W, F)
        hybrid_dims                = _tune_hybrid_kan(target, W, F)
        res_fcnn_dims, res_kan_hid = _tune_res_kan(target, W, F)
        conv_nf, conv_kh           = _tune_conv_kan(target, W, F)
        attn_d                     = _tune_kan_attention(target, F)
        ms_hidden                  = _tune_multiscale_kan(target, W, k)
        gated_hidden               = _tune_gated_kan(target, W, F)
    else:
        vanilla_hidden = 18
        kan_fcnn_dims  = [16, 8]
        hybrid_dims    = [64, 32]
        res_fcnn_dims  = [64, 32]
        res_kan_hid    = 8
        conv_nf, conv_kh = 8, 8
        attn_d         = 16
        ms_hidden      = 8
        gated_hidden   = 8

    vanilla_cfg  = {**shared, 'vanilla_hidden': vanilla_hidden, 'epochs': 25}
    kan_fcnn_cfg = {**shared, 'hidden_dims': kan_fcnn_dims, 'epochs': 30}
    hybrid_cfg   = {**shared, 'linear_dims': hybrid_dims,   'epochs': 20}
    res_cfg      = {**shared, 'fcnn_dims': res_fcnn_dims,
                               'kan_hidden': res_kan_hid,   'epochs': 20}
    conv_kan_cfg = {**shared, 'n_filters': conv_nf, 'kernel_size': 5,
                               'pool_out': 4, 'kan_hidden': conv_kh,
                               'epochs': 25}
    attn_cfg     = {**shared, 'd_model': attn_d,             'epochs': 25}
    ms_cfg       = {**shared, 'ms_hidden': ms_hidden,
                               'ms_grid_sizes': (3, 5, 8),   'epochs': 25}
    gated_cfg    = {**shared, 'gated_hidden': gated_hidden,  'epochs': 25}

    return (vanilla_cfg, kan_fcnn_cfg, hybrid_cfg, res_cfg,
            conv_kan_cfg, attn_cfg, ms_cfg, gated_cfg)


(VANILLA_KAN_CONFIG,
 KAN_FCNN_CONFIG, HYBRID_KAN_CONFIG, RES_KAN_CONFIG,
 CONV_KAN_CONFIG, KAN_ATTN_CONFIG,
 MULTISCALE_KAN_CONFIG, GATED_KAN_CONFIG) = _build_configs()


# ================= 数据集定义 =================
class OpticalDataset(Dataset):
    def __init__(self, rx_signal, labels, window_size, sps,
                 rx_mean=None, rx_std=None, label_scale=3.0):
        self.rx = rx_signal
        self.labels = labels
        self.w = window_size
        self.sps = sps
        self.label_scale = label_scale
        self.n_samples = len(labels) - (window_size // sps) - 1
        self.rx_mean = rx_mean if rx_mean is not None else np.mean(rx_signal)
        self.rx_std  = rx_std  if rx_std  is not None else np.std(rx_signal)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start_sample = idx * self.sps
        end_sample   = start_sample + self.w
        x_seq = self.rx[start_sample:end_sample]
        x_seq = (x_seq - self.rx_mean) / (self.rx_std + 1e-8)
        label_idx = idx + (self.w // self.sps) // 2
        y = self.labels[label_idx] / self.label_scale
        return torch.FloatTensor(x_seq.real), torch.FloatTensor([y])


# ============================================================
#  方案 0:  VanillaKAN — 纯 KAN 基准
# ============================================================
class VanillaKANEqualizer(nn.Module):
    """
    纯 KAN 均衡器 — 仅由两层 KANLinear 构成，不含任何辅助模块。

    架构: KANLinear(W, h) → KANLinear(h, 1)

    作为 KAN 算法在 IM/DD 均衡中的纯净基准:
      - 无 LayerNorm / BatchNorm (测试 KAN 裸性能)
      - 无 Linear / ReLU / CNN / 注意力等辅助结构
      - 完全依赖 KAN 的 B-样条 + SiLU 基函数进行非线性逼近
    用于与所有混合方案对比，量化各辅助结构带来的增益。
    """

    def __init__(self, input_dim, kan_hidden=18,
                 grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)):
        super().__init__()
        self.net = nn.Sequential(
            KANLinear(input_dim, kan_hidden, grid_size, spline_order,
                      grid_range),
            KANLinear(kan_hidden, 1, grid_size, spline_order, grid_range),
        )

    def forward(self, src):
        x = src.squeeze(-1)
        return self.net(x)


def build_vanilla_kan(config, device):
    return VanillaKANEqualizer(
        input_dim    = config['window_size'],
        kan_hidden   = config['vanilla_hidden'],
        grid_size    = config['grid_size'],
        spline_order = config['spline_order'],
        grid_range   = tuple(config.get('grid_range', (-2.0, 2.0))),
    ).to(device)


# ============================================================
#  方案 1:  KAN-FCNN — 纯 KAN 前馈网络
# ============================================================
class KANFCNNEqualizer(nn.Module):
    """
    用 KANLinear (B-样条边激活) 完全替代 FCNN 的 Linear+ReLU。
    KAN 自带非线性 (SiLU 基 + 样条)，无需额外激活函数。
    层间加 LayerNorm 使中间激活落在样条网格范围内。
    """

    def __init__(self, input_dim, hidden_dims=None,
                 grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 8]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(KANLinear(in_dim, h, grid_size, spline_order, grid_range))
            layers.append(nn.LayerNorm(h))
            in_dim = h
        layers.append(KANLinear(in_dim, 1, grid_size, spline_order, grid_range))
        self.net = nn.Sequential(*layers)

    def forward(self, src):
        x = src.squeeze(-1)
        return self.net(x)


def build_kan_fcnn(config, device):
    return KANFCNNEqualizer(
        input_dim    = config['window_size'],
        hidden_dims  = config['hidden_dims'],
        grid_size    = config['grid_size'],
        spline_order = config['spline_order'],
        grid_range   = tuple(config.get('grid_range', (-2.0, 2.0))),
    ).to(device)


# ============================================================
#  方案 2:  Hybrid-KAN — FCNN 前端 + KAN 后端
# ============================================================
class HybridKANEqualizer(nn.Module):
    """
    混合架构：
      前端 — 标准 Linear+ReLU 做高效线性降维 (与 FCNN 相同)
      后端 — KANLinear 做非线性符号回归

    核心假设: FCNN 的 Linear+ReLU 擅长线性特征提取 (撤色散)，
    但在最终的非线性映射步骤 (逆方检波) 上可能不够精确。
    用 KAN 替换最后一层可以更好地逼近该非线性函数。
    """

    def __init__(self, input_dim, linear_dims=None,
                 grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)):
        super().__init__()
        if linear_dims is None:
            linear_dims = [64, 32]

        layers = []
        in_dim = input_dim
        for h in linear_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.linear_front = nn.Sequential(*layers)

        self.kan_out = KANLinear(
            in_dim, 1, grid_size, spline_order, grid_range
        )

    def forward(self, src):
        x = src.squeeze(-1)
        x = self.linear_front(x)
        return self.kan_out(x)


def build_hybrid_kan(config, device):
    return HybridKANEqualizer(
        input_dim    = config['window_size'],
        linear_dims  = config['linear_dims'],
        grid_size    = config['grid_size'],
        spline_order = config['spline_order'],
        grid_range   = tuple(config.get('grid_range', (-2.0, 2.0))),
    ).to(device)


# ============================================================
#  方案 3:  ResKAN — FCNN + KAN 残差校正
# ============================================================
class ResKANEqualizer(nn.Module):
    """
    残差增强架构：
      主路径 — 标准 FCNN (保证基线性能)
      残差路径 — 轻量 KAN 分支 (学习 FCNN 遗漏的非线性校正)
      输出 = FCNN(x) + α · KAN(x)，α 为可学习标量

    优势: 即使 KAN 分支学不到有用信息 (α→0)，
    性能也不会比 FCNN 差。
    """

    def __init__(self, input_dim, fcnn_dims=None, kan_hidden=8,
                 grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)):
        super().__init__()
        if fcnn_dims is None:
            fcnn_dims = [64, 32]

        fcnn_layers = []
        in_dim = input_dim
        for h in fcnn_dims:
            fcnn_layers.append(nn.Linear(in_dim, h))
            fcnn_layers.append(nn.ReLU())
            in_dim = h
        fcnn_layers.append(nn.Linear(in_dim, 1))
        self.fcnn = nn.Sequential(*fcnn_layers)

        self.kan_branch = nn.Sequential(
            KANLinear(input_dim, kan_hidden, grid_size, spline_order, grid_range),
            KANLinear(kan_hidden, 1, grid_size, spline_order, grid_range),
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, src):
        x = src.squeeze(-1)
        main_out = self.fcnn(x)
        kan_res  = self.kan_branch(x)
        return main_out + self.alpha * kan_res


def build_res_kan(config, device):
    return ResKANEqualizer(
        input_dim    = config['window_size'],
        fcnn_dims    = config['fcnn_dims'],
        kan_hidden   = config['kan_hidden'],
        grid_size    = config['grid_size'],
        spline_order = config['spline_order'],
        grid_range   = tuple(config.get('grid_range', (-2.0, 2.0))),
    ).to(device)


# ============================================================
#  方案 4:  ConvKAN — 1D 卷积 + KAN 管道
# ============================================================
class ConvKANEqualizer(nn.Module):
    """
    卷积 + KAN 管道均衡器。

    前端 — 两层 1D 卷积提取局部 ISI 特征（平移不变性）
    池化 — 自适应平均池化降维
    后端 — KAN 层进行非线性符号回归

    设计动机 (受 Kolmogorov-Arnold Convolutions [arXiv:2407.01092]
    和 OFC 2025 CNN 均衡器研究启发):
      ISI 本质上是信号与信道脉冲响应的卷积，Conv1D 直接建模其逆过程
      比全连接层更高效且具有平移不变性；KAN 后端处理残余非线性效应
      (方检波、SOA 非线性、限幅失真等)。
    """

    def __init__(self, input_dim, n_filters=8, kernel_size=5, pool_out=4,
                 kan_hidden=8, grid_size=5, spline_order=3,
                 grid_range=(-2.0, 2.0)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_out),
        )
        flat_dim = n_filters * pool_out
        self.kan_head = nn.Sequential(
            KANLinear(flat_dim, kan_hidden, grid_size, spline_order,
                      grid_range),
            nn.LayerNorm(kan_hidden),
            KANLinear(kan_hidden, 1, grid_size, spline_order, grid_range),
        )

    def forward(self, src):
        x = src.squeeze(-1)              # (B, W)
        x = x.unsqueeze(1)               # (B, 1, W) for Conv1d
        x = self.conv(x)                 # (B, nf, pool_out)
        x = x.flatten(1)                 # (B, nf * pool_out)
        return self.kan_head(x)


def build_conv_kan(config, device):
    return ConvKANEqualizer(
        input_dim    = config['window_size'],
        n_filters    = config['n_filters'],
        kernel_size  = config.get('kernel_size', 5),
        pool_out     = config.get('pool_out', 4),
        kan_hidden   = config['kan_hidden'],
        grid_size    = config['grid_size'],
        spline_order = config['spline_order'],
        grid_range   = tuple(config.get('grid_range', (-2.0, 2.0))),
    ).to(device)


# ============================================================
#  方案 5:  KAN-Attention — 注意力池化 KAN
# ============================================================
class KANAttentionEqualizer(nn.Module):
    """
    注意力增强 KAN 均衡器。

    将输入窗口的每个采样点嵌入到 d_model 维空间，
    用 KAN 计算各位置的注意力权重 (动态重要性)，
    再用 KAN 对特征进行非线性变换后加权池化。

    设计动机 (受 KArAt [OpenReview 2024] 和 IEEE OFC 2024
    注意力 CNN 均衡器 [200 Gbit/s/λ PAM-4] 启发):
      IM/DD 系统中 ISI 的影响随位置变化 — 窗口中心附近的采样点
      包含最多的目标符号信息，而边缘采样点主要提供 ISI 上下文。
      注意力机制使均衡器能动态聚焦于信息量最大的采样点，
      KAN 则在注意力计算和特征变换中提供更精确的非线性逼近。
    """

    def __init__(self, input_dim, d_model=16,
                 grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.attn_kan = KANLinear(d_model, 1, grid_size, spline_order,
                                  grid_range)
        self.feat_kan = KANLinear(d_model, d_model, grid_size, spline_order,
                                  grid_range)
        self.out_kan  = KANLinear(d_model, 1, grid_size, spline_order,
                                  grid_range)

    def forward(self, src):
        x = src if src.dim() == 3 else src.unsqueeze(-1)  # (B, W, 1)
        x = self.embed(x)                                  # (B, W, d)
        attn = self.attn_kan(x).squeeze(-1)                # (B, W)
        attn = F.softmax(attn, dim=-1).unsqueeze(-1)       # (B, W, 1)
        feat = self.feat_kan(x)                            # (B, W, d)
        pooled = (attn * feat).sum(dim=1)                  # (B, d)
        return self.out_kan(pooled)                        # (B, 1)


def build_kan_attention(config, device):
    return KANAttentionEqualizer(
        input_dim    = config['window_size'],
        d_model      = config['d_model'],
        grid_size    = config['grid_size'],
        spline_order = config['spline_order'],
        grid_range   = tuple(config.get('grid_range', (-2.0, 2.0))),
    ).to(device)


# ============================================================
#  方案 6:  MultiScale-KAN — 多尺度并行 KAN
# ============================================================
class MultiScaleKANEqualizer(nn.Module):
    """
    多尺度并行 KAN 均衡器。

    多个并行 KAN 分支使用不同的 B-样条网格分辨率 (grid_size)，
    分别捕获粗粒度和细粒度的非线性特征，
    输出通过可学习 softmax 权重自适应融合。

    设计动机 (受 BSRBF-KAN 双基函数思想和多分辨率分析启发):
      B-样条的网格密度决定了可表示函数的频率范围 ——
      粗网格 (grid=3) 捕获低频趋势 (缓变非线性如色散),
      中网格 (grid=5) 平衡精度与泛化,
      细网格 (grid=8) 刻画高频细节 (尖锐非线性如限幅/SOA 饱和)。
      多尺度融合避免了手动选择最优 grid_size 的困难，
      并在不同信道条件下自适应调整各尺度的贡献权重。
    """

    def __init__(self, input_dim, hidden=8,
                 grid_sizes=(3, 5, 8), spline_order=3,
                 grid_range=(-2.0, 2.0)):
        super().__init__()
        self.n_branches = len(grid_sizes)
        self.branches = nn.ModuleList()
        for gs in grid_sizes:
            branch = nn.Sequential(
                KANLinear(input_dim, hidden, gs, spline_order, grid_range),
                nn.LayerNorm(hidden),
                KANLinear(hidden, 1, gs, spline_order, grid_range),
            )
            self.branches.append(branch)
        self.fusion_weights = nn.Parameter(
            torch.ones(self.n_branches) / self.n_branches
        )

    def forward(self, src):
        x = src.squeeze(-1)
        outputs = torch.stack(
            [branch(x) for branch in self.branches], dim=0
        )                                                       # (n, B, 1)
        w = F.softmax(self.fusion_weights, dim=0).view(-1, 1, 1) # (n, 1, 1)
        return (w * outputs).sum(dim=0)                         # (B, 1)


def build_multiscale_kan(config, device):
    return MultiScaleKANEqualizer(
        input_dim    = config['window_size'],
        hidden       = config['ms_hidden'],
        grid_sizes   = tuple(config.get('ms_grid_sizes', (3, 5, 8))),
        spline_order = config['spline_order'],
        grid_range   = tuple(config.get('grid_range', (-2.0, 2.0))),
    ).to(device)


# ============================================================
#  方案 7:  GatedKAN — 门控双路径 KAN
# ============================================================
class GatedKANEqualizer(nn.Module):
    """
    门控 KAN 均衡器。

    双路径架构:
      信号路径 — KAN 层计算候选非线性输出
      门控路径 — KAN 层 + Sigmoid 生成逐元素门控信号
      线性旁路 — 标准 Linear 提供稳定基线
    输出 = gate ⊙ signal + (1 - gate) ⊙ bypass

    设计动机 (受 LSTM/GRU 门控机制和 KAN 2.0 自适应激活选择启发):
      光通道非线性在特征空间中并非均匀分布 — 某些特征维度
      受方检波非线性影响显著，另一些则以线性 ISI 为主。
      门控机制允许网络自适应选择: 在非线性失真明显的维度
      使用 KAN 信号路径的精确逼近，在线性区域使用简单的
      线性旁路。这种「按需非线性」策略在训练稳定性和
      泛化性上优于全局统一的非线性变换。
    """

    def __init__(self, input_dim, hidden_dim=8,
                 grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)):
        super().__init__()
        self.signal_kan = nn.Sequential(
            KANLinear(input_dim, hidden_dim, grid_size, spline_order,
                      grid_range),
            nn.LayerNorm(hidden_dim),
        )
        self.gate_kan = nn.Sequential(
            KANLinear(input_dim, hidden_dim, grid_size, spline_order,
                      grid_range),
            nn.Sigmoid(),
        )
        self.bypass = nn.Linear(input_dim, hidden_dim)
        self.out_kan = KANLinear(hidden_dim, 1, grid_size, spline_order,
                                 grid_range)

    def forward(self, src):
        x = src.squeeze(-1)
        signal = self.signal_kan(x)
        gate   = self.gate_kan(x)
        bypass = self.bypass(x)
        fused  = gate * signal + (1.0 - gate) * bypass
        return self.out_kan(fused)


def build_gated_kan(config, device):
    return GatedKANEqualizer(
        input_dim    = config['window_size'],
        hidden_dim   = config['gated_hidden'],
        grid_size    = config['grid_size'],
        spline_order = config['spline_order'],
        grid_range   = tuple(config.get('grid_range', (-2.0, 2.0))),
    ).to(device)


# ================= 模型注册表 =================
MODEL_TABLE = {
    'vanilla_kan': {
        'display': 'VanillaKAN',
        'build_fn': build_vanilla_kan,
        'config': VANILLA_KAN_CONFIG,
        'ckpt': 'vanilla_kan_model.pth',
    },
    'kan_fcnn': {
        'display': 'KAN-FCNN',
        'build_fn': build_kan_fcnn,
        'config': KAN_FCNN_CONFIG,
        'ckpt': 'kan_fcnn_model.pth',
    },
    'hybrid_kan': {
        'display': 'Hybrid-KAN',
        'build_fn': build_hybrid_kan,
        'config': HYBRID_KAN_CONFIG,
        'ckpt': 'hybrid_kan_model.pth',
    },
    'res_kan': {
        'display': 'ResKAN',
        'build_fn': build_res_kan,
        'config': RES_KAN_CONFIG,
        'ckpt': 'res_kan_model.pth',
    },
    'conv_kan': {
        'display': 'ConvKAN',
        'build_fn': build_conv_kan,
        'config': CONV_KAN_CONFIG,
        'ckpt': 'conv_kan_model.pth',
    },
    'kan_attn': {
        'display': 'KAN-Attention',
        'build_fn': build_kan_attention,
        'config': KAN_ATTN_CONFIG,
        'ckpt': 'kan_attn_model.pth',
    },
    'multiscale_kan': {
        'display': 'MultiScale-KAN',
        'build_fn': build_multiscale_kan,
        'config': MULTISCALE_KAN_CONFIG,
        'ckpt': 'multiscale_kan_model.pth',
    },
    'gated_kan': {
        'display': 'GatedKAN',
        'build_fn': build_gated_kan,
        'config': GATED_KAN_CONFIG,
        'ckpt': 'gated_kan_model.pth',
    },
}


# ================= 参数量统计 =================
def count_parameters(model, log):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  总参数量:     {total:,}")
    log.info(f"  可训练参数量: {trainable:,}")
    log.info("  各层参数明细:")
    for name, p in model.named_parameters():
        log.info(f"    {name}: {p.numel():,}  {list(p.shape)}")
    return total


# ================= 评估 =================
def evaluate(model, loader, device, criterion, label_scale):
    model.eval()
    total_loss = 0.0
    preds_list, targets_list = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.unsqueeze(-1).to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    model.train()
    avg_loss = total_loss / len(loader)

    preds   = np.concatenate(preds_list).flatten()   * label_scale
    targets = np.concatenate(targets_list).flatten() * label_scale

    thresholds = [-2, 0, 2]
    pred_labels = np.select(
        [preds < thresholds[0], preds < thresholds[1], preds < thresholds[2]],
        [-3, -1, 1], default=3
    )
    true_labels = np.select(
        [targets < thresholds[0], targets < thresholds[1], targets < thresholds[2]],
        [-3, -1, 1], default=3
    )
    ber = float(np.mean(pred_labels != true_labels))
    return avg_loss, ber


# ================= 数据加载 =================
def load_data():
    data = scipy.io.loadmat(str(ROOT / 'dataset_for_python.mat'))

    rx_train = data['rx_train_export'].flatten()
    if np.iscomplexobj(rx_train):
        rx_train = np.abs(rx_train)
    symb_train = data['symb_train_export'].flatten()

    rx_test = data['rx_test_export'].flatten()
    if np.iscomplexobj(rx_test):
        rx_test = np.abs(rx_test)
    symb_test = data['symb_test_export'].flatten()

    rx_mean = float(np.mean(rx_train))
    rx_std  = float(np.std(rx_train))

    return {
        'rx_train': rx_train, 'symb_train': symb_train,
        'rx_test': rx_test,   'symb_test': symb_test,
        'rx_mean': rx_mean,   'rx_std': rx_std,
    }


# ================= 单模型训练 =================
def train_single(key, entry, data_dict):
    display_name = entry['display']
    build_fn     = entry['build_fn']
    config       = entry['config']
    ckpt_name    = entry['ckpt']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = LOGS_DIR / f'{key}_train_{timestamp}.log'
    log = logging.getLogger(f'{key}_trainer')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(logging.FileHandler(log_path, mode='w', encoding='utf-8'))
    log.addHandler(logging.StreamHandler())
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for h in log.handlers:
        h.setFormatter(formatter)

    log.info("=" * 60)
    log.info(f"  {display_name} Equalizer — 训练开始")
    log.info("=" * 60)
    dev = config['device']
    log.info(f"运行设备: {dev}")
    log.info(f"配置参数: {config}")

    if MATCH_PARAMS:
        target = _fcnn_param_count(BASE['window_size'], BASE['fcnn_hidden_dims'])
        log.info(f"[参数对齐] 目标参数量: {target:,} (FCNN {BASE['fcnn_hidden_dims']})")

    rx_mean, rx_std = data_dict['rx_mean'], data_dict['rx_std']
    log.info(f"归一化统计: mean={rx_mean:.4f}, std={rx_std:.4f}")
    log.info(f"训练信号点数: {len(data_dict['rx_train']):,}  |  "
             f"训练符号数: {len(data_dict['symb_train']):,}")
    log.info(f"测试信号点数: {len(data_dict['rx_test']):,}   |  "
             f"测试符号数: {len(data_dict['symb_test']):,}")

    train_dataset = OpticalDataset(
        data_dict['rx_train'], data_dict['symb_train'],
        config['window_size'], config['sps'],
        rx_mean=rx_mean, rx_std=rx_std, label_scale=config['label_scale']
    )
    test_dataset = OpticalDataset(
        data_dict['rx_test'], data_dict['symb_test'],
        config['window_size'], config['sps'],
        rx_mean=rx_mean, rx_std=rx_std, label_scale=config['label_scale']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'],
                              shuffle=False, num_workers=0)

    model = build_fn(config, dev)
    log.info(f"\n===== {display_name} 参数量统计 =====")
    count_parameters(model, log)
    log.info("=" * 40 + "\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-5
    )

    train_loss_history = []
    val_loss_history   = []
    val_ber_history    = []
    best_val_loss = float('inf')
    best_val_ber  = float('inf')
    best_epoch    = 0

    model.train()
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.perf_counter()
        epoch_loss  = 0.0

        for inputs, targets in train_loader:
            inputs  = inputs.unsqueeze(-1).to(dev)
            targets = targets.to(dev)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_time     = time.perf_counter() - epoch_start
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % config['eval_interval'] == 0:
            val_loss, val_ber = evaluate(
                model, test_loader, dev, criterion, config['label_scale']
            )
            val_loss_history.append(val_loss)
            val_ber_history.append(val_ber)

            log.info(
                f"Epoch [{epoch:3d}/{config['epochs']}] "
                f"Time: {epoch_time:5.1f}s | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val BER: {val_ber:.4e} | "
                f"LR: {current_lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ber  = val_ber
                best_epoch    = epoch
                save_path = MODELS_DIR / ckpt_name
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_type': display_name,
                    'rx_mean': rx_mean,
                    'rx_std':  rx_std,
                    'config':  config,
                    'epoch':   epoch,
                    'best_val_loss': best_val_loss,
                    'best_val_ber':  best_val_ber,
                }, str(save_path))
                log.info(f"  ★ 最优模型已更新 → Epoch {epoch}, "
                         f"Val Loss={val_loss:.6f}, Val BER={val_ber:.4e}")
        else:
            log.info(
                f"Epoch [{epoch:3d}/{config['epochs']}] "
                f"Time: {epoch_time:5.1f}s | "
                f"Train Loss: {avg_train_loss:.6f} | LR: {current_lr:.2e}"
            )

    log.info("\n" + "=" * 60)
    log.info(f"训练完成！最优 Epoch: {best_epoch}")
    log.info(f"最优验证损失: {best_val_loss:.6f}")
    log.info(f"最优验证 BER:  {best_val_ber:.4e}")
    log.info(f"模型已保存至:  {MODELS_DIR / ckpt_name}")
    log.info("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_loss_history, label='Train Loss')
    eval_epochs = list(range(config['eval_interval'],
                             config['epochs'] + 1, config['eval_interval']))
    axes[0].plot(eval_epochs, val_loss_history, label='Val Loss')
    axes[0].axvline(x=best_epoch, color='r', linestyle='--',
                    label=f'Best Epoch ({best_epoch})')
    axes[0].set_title('MSE Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].semilogy(eval_epochs, val_ber_history, 'o-', color='C2', label='Val BER')
    axes[1].axvline(x=best_epoch, color='r', linestyle='--',
                    label=f'Best Epoch ({best_epoch})')
    axes[1].set_title('Validation BER (PAM4 硬判决)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('BER')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.4)

    plt.suptitle(f'{display_name} — Training Summary (Best BER={best_val_ber:.4e})',
                 fontsize=12)
    plt.tight_layout()
    fig_path = IMAGES_DIR / f'{key}_training_loss.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"训练曲线已保存: {fig_path}")

    return best_val_ber


# ================= 入口 =================
def train_all(keys=None):
    if keys is None:
        keys = list(MODEL_TABLE.keys())

    # 打印参数对齐信息
    target = _fcnn_param_count(BASE['window_size'], BASE['fcnn_hidden_dims'])
    print("\n" + "=" * 62)
    if MATCH_PARAMS:
        print(f"  [参数对齐: ON]  目标 = {target:,} (FCNN {BASE['fcnn_hidden_dims']})")
    else:
        print(f"  [参数对齐: OFF]  FCNN 参数量 = {target:,} (仅供参考)")
    print("=" * 62)

    for k in keys:
        entry = MODEL_TABLE[k]
        cfg = entry['config']
        tmp = entry['build_fn'](cfg, 'cpu')
        p = sum(pp.numel() for pp in tmp.parameters())
        dims_info = (cfg.get('hidden_dims') or
                     cfg.get('linear_dims') or
                     cfg.get('fcnn_dims') or [])
        extra_parts = []
        for ek in ('vanilla_hidden', 'kan_hidden', 'n_filters', 'd_model',
                    'ms_hidden', 'gated_hidden'):
            if ek in cfg:
                extra_parts.append(f"{ek}={cfg[ek]}")
        extra = (", " + ", ".join(extra_parts)) if extra_parts else ""
        print(f"  {entry['display']:15s}  dims={dims_info}{extra}"
              f"  →  {p:,} 参数 (Δ={p - target:+d})")
        del tmp

    print("=" * 62)

    data_dict = load_data()
    print(f"\n数据已加载，将训练: {[MODEL_TABLE[k]['display'] for k in keys]}\n")

    results = {}
    for k in keys:
        entry = MODEL_TABLE[k]
        best_ber = train_single(k, entry, data_dict)
        results[entry['display']] = best_ber
        print()

    print("\n" + "=" * 50)
    print("         全部训练完成 — 最优 BER 汇总")
    print("=" * 50)
    for name, ber in results.items():
        print(f"  {name:15s}  BER = {ber:.4e}")
    print("=" * 50)


if __name__ == '__main__':
    requested = sys.argv[1:] if len(sys.argv) > 1 else None
    if requested:
        valid = [k for k in requested if k in MODEL_TABLE]
        if not valid:
            print(f"可用模型: {list(MODEL_TABLE.keys())}")
            sys.exit(1)
        train_all(valid)
    else:
        train_all()
