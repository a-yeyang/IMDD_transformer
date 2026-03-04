# 均衡器结构对比流程图（论文用）

以下为 Mermaid 流程图源码。可复制到 [Mermaid Live Editor](https://mermaid.live) 或 Typora / VS Code 等支持 Mermaid 的编辑器中渲染，并导出为 PNG/SVG 插入论文。

---

## (a) 左图：Transformer TEQ

```mermaid
flowchart TB
    subgraph TEQ["Transformer TEQ"]
        A["Input x (B×L×1)"]
        B["Linear(1→d) Input Projection"]
        C["+ Sinusoidal Position Encoding"]
        D["Shared Encoder Layer × 2<br/>• Multi-Head Self-Attention<br/>• Add & LayerNorm<br/>• FFN: Linear→ReLU→Linear<br/>• Add & LayerNorm"]
        E["Center Token z_{L/2} (B×d)"]
        F["Linear(d→1) Output Head"]
        G["Output ŷ (B×1)"]
    end
    A --> B --> C --> D --> E --> F --> G
```

---

## (b) 右图：KAN-Former TEQ

```mermaid
flowchart TB
    subgraph KAN["KAN-Former TEQ"]
        A2["Input x (B×L×1)"]
        B2["FastKANLinear(1→d)<br/>SiLU + Linear + Σₖ wₖ sin(k·x)"]
        C2["+ Sinusoidal Position Encoding"]
        D2["Shared KAN-Encoder Layer × 2<br/>• Multi-Head Self-Attention<br/>• Add & LayerNorm<br/>• KAN-FFN: FastKANLinear→FastKANLinear<br/>• Add & LayerNorm"]
        E2["Center Token z_{L/2} (B×d)"]
        F2["Linear(d→1) Output Head"]
        G2["Output ŷ (B×1)"]
    end
    A2 --> B2 --> C2 --> D2 --> E2 --> F2 --> G2
```

---

## 并排对比（单图，可选）

若需在一张图内左右并排，可使用下方代码（部分渲染器支持 subgraph 左右排列）：

```mermaid
flowchart LR
    subgraph Left["(a) Transformer TEQ"]
        direction TB
        L1[Input] --> L2[Linear Proj] --> L3[+ PE] --> L4[Encoder×2<br/>MHA+FFN] --> L5[Center] --> L6[Head] --> L7[Output]
    end
    subgraph Right["(b) KAN-Former TEQ"]
        direction TB
        R1[Input] --> R2[FastKANLinear Proj] --> R3[+ PE] --> R4[KAN-Encoder×2<br/>MHA+KAN-FFN] --> R5[Center] --> R6[Head] --> R7[Output]
    end
```

---

**说明**：论文中建议分别渲染 (a)(b) 两张图，在正文中并排插入，图注为 “(a) Transformer TEQ 结构；(b) KAN-Former TEQ 结构”。
