"""
CMA 和 Volterra 均衡器（Python 实现）

用于 PAM4 IM/DD 光通信系统，与 compare_all_equalizers.py 配合使用。

CMAEqualizer:
    3 级 CMA 自适应均衡，从 MATLAB CMA.m 精确移植。
    内部使用"反转滤波器" H_rev（H_rev[k] = H[n_taps-1-k]）与前向滑窗直接点积。
    均衡后对 sps 个候选相位做相位选择，再用线性回归将输出对齐到
    PAM4 电平 {-3,-1,1,3}，使判决门限落在 ±2/0 的标准位置。

VolterraEqualizer:
    二阶 Volterra 级数均衡器，批量正规方程 + 岭正则化最小二乘训练。
    与 NN 均衡器使用完全相同的滑窗格式（window_size=21, sps=2）：
        窗口    : rx[i·sps : i·sps + window_size]
        标签偏移: i + window_size // (sps·2)  = i + 5 (默认参数)
    训练目标为 PAM4 电平 {-3,-1,1,3}（无需归一化），
    输出在 PAM4 尺度，判决门限 ±2/0 直接适用。
    特征数（window_size=21, order=2）: 1 + 21 + 231 = 253
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ===========================================================
#  CMA 均衡器
# ===========================================================
class CMAEqualizer:
    """
    3 级 CMA FIR 均衡器（实值信号版本，适配 IM/DD PAM4）。

    MATLAB 对应推导
    ---------------
    MATLAB 原始量:
        col_rev[k] = I_sig[i + n_taps - 1 - k]   (反转的滑窗列)
        y          = H_xx^T · col_rev
        H_xx      += mu · eps · y · col_rev

    等价 Python 量（令 H_rev[k] = H_xx[n_taps-1-k]，col_fwd = I_sig[i:i+n_taps]）:
        y          = H_rev^T · col_fwd
        H_rev     += mu · eps · y · col_fwd

    对于奇数 n_taps，中心抽头索引相同（center = n_taps//2），
    因此 H_rev 的初始化与 H_xx 完全对称。

    均衡输出（完整信号）:
        output = windows @ H_rev         (高效矩阵向量乘法)
    """

    def __init__(self, n_taps=25, sps=2,
                 mu1=1e-3, mu2=6e-4, mu3=1e-4,
                 n_adapt=None):
        """
        Parameters
        ----------
        n_taps   : 奇数，FIR 滤波器抽头数（默认 25，同 MATLAB 默认值）
        sps      : 每符号采样数（默认 2）
        mu1/2/3  : 三级 CMA 步长（同 MATLAB 默认值）
        n_adapt  : 用于 CMA 自适应的最大样本数；None = 使用全部训练数据
        """
        assert n_taps % 2 == 1, "n_taps 必须为奇数"
        self.n_taps  = n_taps
        self.sps     = sps
        self.mus     = [mu1, mu2, mu3]
        self.n_adapt = n_adapt

        self.H_rev   = None    # 已训练的反转滤波器系数 (n_taps,)
        self.scale_a = 1.0     # 线性映射: y_pam4 = a · y_cma + b
        self.scale_b = 0.0
        self.phase   = 0       # 最优下采样相位 (0 .. sps-1)

    # -------------------------------------------------------
    def _preprocess(self, rx):
        """
        两端各填充 n_taps//2 个样本 + RMS 归一化。
        返回 I_sig，长度 = len(rx) + n_taps - 1。
        与 MATLAB: PAM_rx=[x(1:L1);x;x(end-L1+1:end)]; I=PAM_rx/sqrt(mean(PAM_rx^2)) 等价。
        """
        L      = self.n_taps // 2
        padded = np.concatenate([rx[:L], rx, rx[-L:]])
        rms    = np.sqrt(max(float(np.mean(padded ** 2)), 1e-12))
        return padded / rms

    # -------------------------------------------------------
    def fit(self, rx_signal, symb_labels):
        """
        在训练信号上运行 3 级 CMA 自适应，再拟合线性尺度对齐到 PAM4 电平。

        Parameters
        ----------
        rx_signal   : (N_rx,) float, 接收信号（sps 倍过采样）
        symb_labels : (N_sym,) int/float, 真实 PAM4 电平 ∈ {-3,-1,1,3}
        """
        n_taps  = self.n_taps
        I_sig   = self._preprocess(rx_signal.astype(np.float64))

        # 前向滑窗矩阵: windows[i, j] = I_sig[i+j]，形状 (N_rx, n_taps)
        windows = sliding_window_view(I_sig, n_taps)
        tmp_len = len(windows)
        n_use   = min(self.n_adapt or tmp_len, tmp_len)

        # 初始化 H_rev：中心抽头为 1（等价于纯零延迟初始化，因两端填充已补偿延迟）
        center = n_taps // 2
        H_rev  = np.zeros(n_taps, dtype=np.float64)
        H_rev[center] = 1.0

        # ── 3 级顺序 CMA 自适应 ──
        for mu in self.mus:
            for i in range(n_use):
                w    = windows[i]             # (n_taps,) 前向窗口
                y    = float(H_rev @ w)
                eps  = 1.0 - y * y
                H_rev += (mu * eps * y) * w   # 原地更新，避免每步重新分配

        self.H_rev = H_rev

        # ── 用冻结的 H_rev 计算全训练集输出 ──
        output = windows @ H_rev              # (N_rx,) 矩阵向量乘法（快）

        # ── 相位选择 + 线性尺度拟合 ──
        best_mse = np.inf
        for ph in range(self.sps):
            sym_out = output[ph::self.sps]
            n_align = min(len(sym_out), len(symb_labels))
            A       = np.column_stack([sym_out[:n_align], np.ones(n_align)])
            a, b    = np.linalg.lstsq(
                          A, symb_labels[:n_align].astype(np.float64), rcond=None)[0]
            mse     = float(np.mean((a * sym_out[:n_align] + b
                                     - symb_labels[:n_align]) ** 2))
            if mse < best_mse:
                best_mse         = mse
                self.phase       = ph
                self.scale_a     = float(a)
                self.scale_b     = float(b)

        return self

    # -------------------------------------------------------
    def predict_aligned(self, rx_signal, symb_labels):
        """
        对测试信号应用冻结滤波器 + 线性尺度变换，返回已对齐的 (预测, 真实) PAM4 电平。

        两个返回值长度一致，可直接传入 calculate_ber()。

        Parameters
        ----------
        rx_signal   : (N_rx,) 接收信号（可含加性噪声）
        symb_labels : (N_sym,) 真实 PAM4 电平

        Returns
        -------
        pred : (n,) float   预测 PAM4 值（接近 {-3,-1,1,3}，含浮点噪声）
        true : (n,) float   对应的真实 PAM4 电平
        """
        I_sig   = self._preprocess(rx_signal.astype(np.float64))
        windows = sliding_window_view(I_sig, self.n_taps)
        output  = windows @ self.H_rev
        sym_out = output[self.phase::self.sps]
        pred    = self.scale_a * sym_out + self.scale_b
        n       = min(len(pred), len(symb_labels))
        return pred[:n], symb_labels[:n].astype(np.float64)


# ===========================================================
#  Volterra 均衡器
# ===========================================================
class VolterraEqualizer:
    """
    二阶 Volterra 级数均衡器（批量最小二乘 + 岭正则化）。

    与 NN 均衡器使用完全相同的滑窗格式（OpticalDataset）：
        窗口    : rx_norm[i·sps : i·sps + window_size]，i = 0,1,...
        标签偏移: label_idx = i + window_size // (sps·2)  （默认 5）

    训练目标为 PAM4 原始电平 {-3,-1,1,3}（不进行 label_scale 归一化），
    因此输出已在 PAM4 尺度，判决门限 ±2/0 直接适用。

    特征向量（window_size=21, order=2）：
        [1, x_0,...,x_20,  x_0x_0, x_0x_1,..., x_20x_20]   共 253 维

    内存高效策略：
        不存储完整特征矩阵，而是逐批累积正规方程 (A^T A, A^T b)，
        再一次性求解 (A^T A + λI) w = A^T b。
    """

    def __init__(self, window_size=21, sps=2, order=2, lambda_reg=1e-6):
        """
        Parameters
        ----------
        window_size : 滑窗宽度（与 NN 均衡器一致，默认 21）
        sps         : 每符号采样数（默认 2）
        order       : Volterra 阶数，1=线性，2=含二阶交叉积（默认 2）
        lambda_reg  : 岭正则化系数（防止过拟合，默认 1e-6）
        """
        self.window_size  = window_size
        self.sps          = sps
        self.order        = order
        self.lambda_reg   = lambda_reg

        W                 = window_size
        n_quad            = W * (W + 1) // 2 if order >= 2 else 0
        self.n_feat       = 1 + W + n_quad
        self.label_offset = W // (sps * 2)          # = 5 for W=21, sps=2
        self._ti, self._tj = np.triu_indices(W)     # 上三角索引（含对角线）

        self.coeffs  = None
        self.rx_mean = 0.0
        self.rx_std  = 1.0

    # -------------------------------------------------------
    def _normalize(self, rx):
        """用训练集统计量做零均值、单位方差归一化。"""
        return (rx - self.rx_mean) / (self.rx_std + 1e-8)

    # -------------------------------------------------------
    def _make_windows(self, rx_norm):
        """
        构建滑窗矩阵，与 OpticalDataset.__getitem__ 对齐。
        返回形状 (N_sym, window_size)，N_sym = (len(rx_norm) - W) // sps。
        """
        W, sps = self.window_size, self.sps
        n      = (len(rx_norm) - W) // sps
        starts = np.arange(n) * sps
        return rx_norm[starts[:, None] + np.arange(W)]   # fancy index → copy

    # -------------------------------------------------------
    def _features(self, windows):
        """
        构建 Volterra 特征矩阵。
        windows: (N, W) float64
        返回  : (N, n_feat) = [偏置 | 线性项 | 二阶交叉积]
        """
        W_f   = windows.astype(np.float64)
        N     = len(W_f)
        parts = [np.ones((N, 1), dtype=np.float64), W_f]
        if self.order >= 2:
            parts.append(W_f[:, self._ti] * W_f[:, self._tj])
        return np.hstack(parts)

    # -------------------------------------------------------
    def fit(self, rx_signal, symb_labels):
        """
        用正规方程（批量累积，节省内存）训练 Volterra 均衡器。

        Parameters
        ----------
        rx_signal   : (N_rx,) 接收信号
        symb_labels : (N_sym,) PAM4 电平 ∈ {-3,-1,1,3}
        """
        self.rx_mean = float(np.mean(rx_signal))
        self.rx_std  = float(np.std(rx_signal))

        rx_norm = self._normalize(rx_signal.astype(np.float64))
        windows = self._make_windows(rx_norm)               # (N, W)
        N       = len(windows)

        sym_idx = np.arange(N) + self.label_offset
        valid   = sym_idx < len(symb_labels)
        windows = windows[valid]
        labels  = symb_labels[sym_idx[valid]].astype(np.float64)  # {-3,-1,1,3}

        # 批量累积正规方程 A^T A 和 A^T b
        n_f = self.n_feat
        ATA = np.zeros((n_f, n_f), dtype=np.float64)
        ATb = np.zeros(n_f,        dtype=np.float64)
        bs  = 8192                                          # 批大小（内存与速度权衡）

        for s in range(0, len(windows), bs):
            e    = min(s + bs, len(windows))
            A    = self._features(windows[s:e])             # (batch, n_f)
            ATA += A.T @ A
            ATb += A.T @ labels[s:e]

        # 岭正则化求解
        ATA.flat[::n_f + 1] += self.lambda_reg             # 对角线 += lambda
        self.coeffs = np.linalg.solve(ATA, ATb)
        return self

    # -------------------------------------------------------
    def predict_aligned(self, rx_signal, symb_labels):
        """
        对测试信号预测 PAM4 电平，返回已对齐的 (预测, 真实)。

        Parameters
        ----------
        rx_signal   : (N_rx,) 接收信号（可含加性噪声）
        symb_labels : (N_sym,) 真实 PAM4 电平

        Returns
        -------
        pred : (n,) float   预测 PAM4 值
        true : (n,) float   对应真实 PAM4 电平
        """
        rx_norm = self._normalize(rx_signal.astype(np.float64))
        windows = self._make_windows(rx_norm)
        N       = len(windows)

        sym_idx = np.arange(N) + self.label_offset
        valid   = sym_idx < len(symb_labels)
        feats   = self._features(windows[valid])
        pred    = feats @ self.coeffs
        true    = symb_labels[sym_idx[valid]].astype(np.float64)
        return pred, true
