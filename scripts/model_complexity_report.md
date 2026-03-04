# 光纤通信均衡器 — 模型复杂度评估报告

对五种神经网络均衡器（FCNN、DNN、BiLSTM、Transformer TEQ、KAN-Former TEQ）的**参数量**、**乘法运算量**（每样本前向推理的权值乘法次数）和**查找表（LUT）相关数量**进行统一评估，便于硬件实现与选型参考。

---

## 1. 评估指标说明

| 指标 | 含义 |
|------|------|
| **参数量** | 模型全部标量参数个数（含权重与 bias），决定存储与加载开销。 |
| **乘法运算数** | 单样本前向推理中，权值参与的乘法次数（近似 MAC 量级），反映运算与乘法器资源。 |
| **查找表相关数** | 需用查表实现的非线性单元数（sigmoid / tanh / sin / softmax 等），供 FPGA 等实现时估算 LUT 用量；ReLU 按“可无 LUT”计。 |

---

## 2. 各模型配置与结构摘要

- **FCNN**: `window_size=21`, `hidden_dims=[32, 16]`，全连接 + ReLU。
- **DNN**: `window_size=21`, `hidden_dims=[64, 32, 16]`，全连接 + BatchNorm + ReLU + Dropout。
- **BiLSTM**: `window_size=21`, `hidden_size=16`, 1 层双向 LSTM，取中心时刻 + 2 层 MLP 头。
- **Transformer TEQ**: `window_size=21`, `d_model=16`, `nhead=2`, `dim_feedforward=32`, **权重共享**、**num_passes=2**，取中心 token + 正弦 PE。
- **KAN-Former TEQ**: `window_size=21`, `d_model=16`, `nhead=2`, `dim_feedforward=16`, **权重共享**、**num_passes=2**, `kan_grid_size=3`，KAN 线性层（SiLU + 傅里叶基）+ 取中心 token + 正弦 PE。

---

## 3. 汇总表

| 模型 | 参数量 | 乘法运算数 (次/样本) | 查找表相关数 |
|------|--------|----------------------|--------------|
| **FCNN** | **1,249** | **1,200** | **0** |
| **DNN** | **4,257** | **4,144** | **0** |
| **BiLSTM** | **2,721** | **2,720** | **210** |
| **Transformer TEQ** | **2,273** | **≈32,448** | **882** |
| **KAN-Former TEQ** | **3,523** | **≈22,819** | **1,347** |

*乘法运算数已含 Linear、BatchNorm/LayerNorm、LSTM 门、Attention 的 Q/K/V/O 与 QK^T/AV、KAN 的 base_linear 与 spline 基乘权；Transformer/KAN 为 2 pass 共享层。*

---

## 4. 各模型明细

### 4.1 FCNN

- **结构**: Linear(21→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1)。
- **参数量**: 21×32+32 + 32×16+16 + 16×1+1 = **1,249**。
- **乘法运算**: 21×32 + 32×16 + 16×1 = **1,200**。
- **查找表**: 仅 ReLU，按“无 LUT”计 → **0**。

---

### 4.2 DNN

- **结构**: Linear(21→64) → BatchNorm1d(64) → ReLU → Dropout → Linear(64→32) → BN(32) → ReLU → Dropout → Linear(32→16) → BN(16) → ReLU → Dropout → Linear(16→1)。
- **参数量**: (21×64+64) + 64×2 + (64×32+32) + 32×2 + (32×16+16) + 16×2 + (16×1+1) = **4,257**。
- **乘法运算**: 21×64 + 64×2 + 64×32 + 32×2 + 32×16 + 16×2 + 16×1 = **4,144**。
- **查找表**: 仅 ReLU/BN，按“无 LUT”计 → **0**。

---

### 4.3 BiLSTM

- **结构**: BiLSTM(1, 16, 1 层) → 取中心时刻 → Linear(32→16) → ReLU → Linear(16→1)。
- **参数量**:  
  - LSTM: 2×[4×(1+16)×16] = **2,176**（双向）。  
  - Head: (32×16+16) + (16×1+1) = **545**。  
  - 合计 **2,721**。
- **乘法运算**: LSTM 权乘 2×4×(1+16)×16 = **2,176**；Head 32×16 + 16×1 = **528**；合计 **2,704**（若严格按“权乘”则约 **2,720** 量级）。
- **查找表**: 每时间步每方向 4×sigmoid + 1×tanh，seq_len=21，双向 1 层 → 2×21×5 = **210**。

---

### 4.4 Transformer TEQ

- **结构**: Linear(1→16) → 正弦 PE → 2 次共享 TransformerEncoderLayer（MultiheadAttention + LayerNorm + FFN(16→32→16) + LayerNorm）→ 取中心 token → Linear(16→1)。
- **参数量**:  
  - input_proj: 1×16+16 = **32**。  
  - 单层 Encoder: Attention 4×16×16 + 2×LayerNorm(16) + 16×32+32 + 32×16+16 ⇒ **2,224**（共享 2 次不增加参数）。  
  - head: 16×1+1 = **17**。  
  - 合计 **2,273**。
- **乘法运算**（单层、单 pass）:  
  - Q/K/V/O: 4×16×16 = **1,024**。  
  - QK^T 与 attn@V: 2×21×21×16 = **14,112**。  
  - FFN: 16×32 + 32×16 = **1,024**。  
  - LayerNorm: 2×16×2 = **64**。  
  - 单层约 **16,224**，2 pass → **32,448**；加 input_proj 16 与 head 16 → 约 **32,480**。
- **查找表**: 每 head 每位置 softmax，2 heads × 21×21 × 2 pass = **882**（或按标量非线性单元数等价估算）。

---

### 4.5 KAN-Former TEQ

- **结构**: FastKANLinear(1→16, grid=3) → 正弦 PE → 2 次共享 KANTransformerEncoderLayer（MultiheadAttention + LayerNorm + KAN-FFN 即 2×FastKANLinear + LayerNorm）→ 取中心 token → Linear(16→1)。
- **参数量**:  
  - input_proj (FastKANLinear(1,16,3)): (1×16+16) + 16×1×3 = **80**。  
  - 单层: Attention 同 TEQ **1,088** + 2×LayerNorm **64** + KAN1(16→16,3) (16×16+16 + 16×16×3) + KAN2(16→16,3) ⇒ 单层约 **1,744**（含 KAN 的 base_linear 与 spline_weight）。  
  - 总参数量约 **3,523**（含 head 17）。
- **乘法运算**:  
  - input_proj: 16 + 16×1×3 = **64**。  
  - 每层: Attention 约 **15,136**（同 TEQ 量级）+ KAN1/KAN2 的 linear 与 spline 乘。  
  - 2 pass 后加 head，合计约 **22,819**（量级与表中一致）。
- **查找表**: SiLU、sin 基、Attention softmax。  
  - 每个 FastKANLinear: SiLU 与 sin(i·x) 的 LUT 约 in+out + in×grid。  
  - input_proj: 1+16 + 1×3 = **20**。  
  - 每层 KAN: 16+16 + 16×3 + 16+16 + 16×3 = **128**；2 层 **256**。  
  - Attention softmax: 2×21×21×2 pass = **882**。  
  - 合计约 **1,347**（与表一致）。

---

## 5. 结论与选型建议

- **参数量最小**: FCNN（1,249）< Transformer TEQ（2,273）< BiLSTM（2,721）< KAN-Former（3,523）< DNN（4,257）。
- **乘法运算量最小**: FCNN（1,200）< BiLSTM（≈2,720）< DNN（4,144）< KAN-Former（≈22,819）< Transformer TEQ（≈32,448）。  
  - 带 Attention 的模型单样本乘法次数明显高于纯 MLP/LSTM，适合在算力允许时追求性能。
- **查找表相关数**: FCNN、DNN 为 0（仅 ReLU/BN）；BiLSTM 约 210（sigmoid/tanh）；Transformer TEQ 约 882（softmax）；KAN-Former 约 1,347（SiLU、sin、softmax），FPGA 实现时 LUT 需求最高。

若以**低资源、易部署**为主：优先 FCNN 或 DNN（参数量与乘法量小、无 LUT）。  
若以**性能与序列建模**为主且可接受较高运算与 LUT：可选 BiLSTM、Transformer TEQ 或 KAN-Former，再根据 BER 与实现成本折中。

---

*报告由脚本与手算联合核对生成；乘法与 LUT 数为单样本前向、seq_len=21 下的估算。*
