# -*- coding: utf-8 -*-
"""
绘制 Transformer 均衡器 与 KAN-Former 均衡器 的程序流程图（一左一右，论文结构对比用）
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# 论文用图：高分辨率、无衬线字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def draw_box(ax, x, y, w, h, text, facecolor='#E8F4FD', edgecolor='#1f77b4', fontsize=8):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color='#333'):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))


def draw_flowchart_transformer(ax, x0=0.5, y_top=1.0, dy=0.12, box_w=0.9, box_h=0.08):
    """左侧：Transformer TEQ 流程图"""
    y = y_top
    # 输入
    draw_box(ax, x0, y, box_w, box_h, r'Input $\mathbf{x}$ $(B \times L \times 1)$', facecolor='#f0f0f0')
    y -= dy
    draw_arrow(ax, x0, y + box_h/2 + 0.01, x0, y + box_h/2)
    # Linear 投影
    draw_box(ax, x0, y, box_w, box_h, r'Linear$(1 \to d)$  Input Projection', facecolor='#E8F4FD')
    y -= dy
    draw_arrow(ax, x0, y + box_h/2 + 0.01, x0, y + box_h/2)
    # 位置编码
    draw_box(ax, x0, y, box_w, box_h, r'+ Sinusoidal Position Encoding', facecolor='#E8F4FD')
    y -= dy
    draw_arrow(ax, x0, y + box_h/2 + 0.01, x0, y + box_h/2)
    # Encoder × 2
    draw_box(ax, x0, y, box_w, box_h*1.4,
             'Shared Encoder Layer × 2\n'
             '• Multi-Head Self-Attention (MHA)\n'
             '• Add & Layer Norm (LN)\n'
             '• FFN: Linear→ReLU→Linear\n'
             '• Add & Layer Norm (LN)',
             facecolor='#B8D4E8')
    y -= dy + box_h*0.6
    draw_arrow(ax, x0, y + box_h/2 + 0.01, x0, y + box_h/2)
    # 取中心 token
    draw_box(ax, x0, y, box_w, box_h, r'Center Token $\mathbf{z}_{L/2}$ $(B \times d)$', facecolor='#E8F4FD')
    y -= dy
    draw_arrow(ax, x0, y + box_h/2 + 0.01, x0, y + box_h/2)
    # 输出头
    draw_box(ax, x0, y, box_w, box_h, r'Linear$(d \to 1)$  Output Head', facecolor='#E8F4FD')
    y -= dy
    draw_arrow(ax, x0, y + box_h/2 + 0.01, x0, y + box_h/2)
    draw_box(ax, x0, y, box_w, box_h, r'Output $\hat{y}$ $(B \times 1)$', facecolor='#d4edda')
    ax.set_xlim(0, 1)
    ax.set_ylim(y - 0.15, y_top + 0.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Transformer TEQ', fontsize=11, fontweight='bold')


def draw_flowchart_kan_former(ax, x0=0.5, y_top=1.0, dy=0.12, box_w=0.9, box_h=0.08):
    """右侧：KAN-Former TEQ 流程图"""
    y = y_top
    draw_box(ax, x0, y, box_w, box_h, r'Input $\mathbf{x}$ $(B \times L \times 1)$', facecolor='#f0f0f0')
    y -= dy
    draw_arrow(ax, x0, y + box_h/2 + 0.01, x0, y + box_h/2)
    # KAN 输入投影
    draw_box(ax, x0, y, box_w, box_h*1.15,
             'FastKANLinear$(1 \to d)$\n'
             'SiLU + Linear + Fourier basis $\\sum_k w_k \\sin(k\\cdot x)$',
             facecolor='#FFE8CC', edgecolor='#e65100')
    y -= dy + box_h*0.3
    draw_arrow(ax, x0, y + box_h/2 + 0.01, x0, y + box_h/2)
    draw_box(ax, x0, y, box_w, box_h, r'+ Sinusoidal Position Encoding', facecolor='#FFE8CC', edgecolor='#e65100')
    y -= dy
    draw_arrow(ax, x0, y + box_h/2 + 0.01, x0, y + box_h/2)
    # KAN-Encoder × 2
    draw_box(ax, x0, y, box_w, box_h*1.5,
             'Shared KAN-Encoder Layer × 2\n'
             '• Multi-Head Self-Attention (MHA)\n'
             '• Add & Layer Norm (LN)\n'
             '• KAN-FFN: FastKANLinear → FastKANLinear\n'
             '  (SiLU + Fourier spline per layer)\n'
             '• Add & Layer Norm (LN)',
             facecolor='#FFCC80', edgecolor='#e65100')
    y -= dy + box_h*0.7
    draw_arrow(ax, x0, y + box_h/2 + 0.01, x0, y + box_h/2)
    draw_box(ax, x0, y, box_w, box_h, r'Center Token $\mathbf{z}_{L/2}$ $(B \times d)$', facecolor='#FFE8CC', edgecolor='#e65100')
    y -= dy
    draw_arrow(ax, x0, y + box_h/2 + 0.01, x0, y + box_h/2)
    draw_box(ax, x0, y, box_w, box_h, r'Linear$(d \to 1)$  Output Head', facecolor='#FFE8CC', edgecolor='#e65100')
    y -= dy
    draw_arrow(ax, x0, y + box_h/2 + 0.01, x0, y + box_h/2)
    draw_box(ax, x0, y, box_w, box_h, r'Output $\hat{y}$ $(B \times 1)$', facecolor='#d4edda')
    ax.set_xlim(0, 1)
    ax.set_ylim(y - 0.15, y_top + 0.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) KAN-Former TEQ', fontsize=11, fontweight='bold')


if __name__ == '__main__':
    from pathlib import Path
    ROOT = Path(__file__).parent.parent
    IMAGES_DIR = ROOT / 'images'
    IMAGES_DIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    draw_flowchart_transformer(ax1)
    draw_flowchart_kan_former(ax2)
    plt.suptitle('Equalizer Architecture Comparison', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = IMAGES_DIR / 'equalizer_flowcharts.png'
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存: {out_path}')
