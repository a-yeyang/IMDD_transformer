import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置参数 =================
CONFIG = {
    'window_size': 21,       # 输入滑窗大小 (符号数 * SPS)
    'sps': 2,                # 每个符号的采样点数 (需与Matlab一致)
    'd_model': 64,           # Transformer特征维度
    'nhead': 4,              # 注意力头数
    'num_layers': 2,         # Encoder层数
    'batch_size': 256,
    'epochs': 50,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ================= 数据集定义 =================
class OpticalDataset(Dataset):
    def __init__(self, rx_signal, labels, window_size, sps):
        """
        rx_signal: 接收到的序列 (复数或实数)
        labels: 发射的符号 (PAM4: -3, -1, 1, 3)
        """
        self.rx = rx_signal
        self.labels = labels
        self.w = window_size
        self.sps = sps
        
        # 能够生成的样本总数
        # 我们预测窗口中心的符号
        self.n_samples = len(labels) - (window_size // sps) - 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 计算对应的采样点索引
        # 假设 labels[i] 对应 rx[i*sps + delay]
        # 这里做一个简化的对齐假设，实际中通常需要先做互相关同步
        
        # 简单的滑动窗口：取 idx*sps 到 idx*sps + window_size
        start_sample = idx * self.sps
        end_sample = start_sample + self.w
        
        # 输入序列
        x_seq = self.rx[start_sample:end_sample]
        
        # 归一化 (简单功率归一化)
        x_seq = x_seq / (np.std(x_seq) + 1e-8)
        
        # 对应的标签 (位于窗口中心的符号)
        # 窗口对应的符号索引大约是 start_symbol + window_symbol_len / 2
        label_idx = idx + (self.w // self.sps) // 2
        y = self.labels[label_idx]
        
        # 将PAM4标签归一化到网络友好范围 (例如 -3, -1, 1, 3 -> 0, 1, 2, 3)
        # 或者直接做回归预测
        return torch.FloatTensor(x_seq.real), torch.FloatTensor([y])

# ================= 模型定义：Physics-Aware Transformer =================
class PhysicsInformedTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super().__init__()
        
        # 输入投影: 将时域采样点映射到高维特征
        self.input_proj = nn.Linear(1, d_model) # 如果是复数输入，这里改为2
        
        # 位置编码 (可学习)
        self.pos_encoder = nn.Parameter(torch.randn(1, input_dim, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出头 (回归预测PAM4电平)
        self.head = nn.Sequential(
            nn.Linear(d_model * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 输出一个实数值
        )

    def forward(self, src):
        # src: [Batch, Window, 1]
        x = self.input_proj(src) # -> [Batch, Window, d_model]
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Transformer处理
        x = self.transformer_encoder(x)
        
        # Flatten并输出
        x = x.reshape(x.size(0), -1)
        out = self.head(x)
        return out

# ================= 训练流程 =================
def train():
    print(f"Running on {CONFIG['device']}")
    
    # 1. 加载Matlab数据
    data = scipy.io.loadmat('dataset_for_python.mat')
    rx_train = data['rx_train_export'].flatten()
    # 确保是实数 (如果是PAM4 IM/DD)
    if np.iscomplexobj(rx_train):
        print("检测到复数信号，取模用于PAM4处理 (如果是Coherent PAM4请保留实部虚部)")
        rx_train = np.abs(rx_train) # 简化的IM/DD假设
        
    symb_train = data['symb_train_export'].flatten()
    
    # 2. 构建Dataset
    train_dataset = OpticalDataset(rx_train, symb_train, CONFIG['window_size'], CONFIG['sps'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    
    # 3. 初始化模型
    model = PhysicsInformedTransformer(CONFIG['window_size'], CONFIG['d_model'], CONFIG['nhead'], CONFIG['num_layers']).to(CONFIG['device'])
    criterion = nn.MSELoss() # PAM4作为回归问题处理
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # 4. 训练循环
    loss_history = []
    model.train()
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.unsqueeze(-1).to(CONFIG['device']) # [B, W, 1]
            targets = targets.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {avg_loss:.4f}")
        
    # 保存模型
    torch.save(model.state_dict(), 'pit_model.pth')
    print("模型已保存。")
    
    # 绘制Loss曲线
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()

if __name__ == '__main__':
    train()