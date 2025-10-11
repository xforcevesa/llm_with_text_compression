import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import math

# ==============================================================================
# 0. 配置与超参数 (Configuration)
# ==============================================================================
class Config:
    VOCAB_SIZE = 50
    SEQ_LENGTH = 256 # 使用一个较长的序列来测试长距离依赖
    D_MODEL = 64
    N_HEAD = 4
    N_LAYERS = 3
    D_FF = 128
    DROPOUT = 0.1
    BATCH_SIZE = 64
    NUM_EPOCHS = 15
    LR = 0.001

# ==============================================================================
# 1. ROSA 算法实现 (ROSA Algorithm Implementation)
# ==============================================================================
def rosa_numpy(x):
    """论文中ROSA算法的Numpy实现，用于在CPU上进行符号计算"""
    n = len(x)
    y = np.full(n, -1, dtype=int)
    s = 2 * n + 1
    b = [{} for _ in range(s)]
    c = np.full(s, -1, dtype=int)
    d = np.zeros(s, dtype=int)
    e = np.full(s, -1, dtype=int)
    g, z = 0, 1
    b[0] = {}

    for i, t_int in enumerate(x):
        t = int(t_int)
        r = z
        z += 1
        b[r] = {}
        d[r] = d[g] + 1
        p = g
        while p != -1 and t not in b[p]:
            b[p][t] = r
            p = c[p]
        
        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z
                z += 1
                b[u] = b[q].copy()
                d[u] = d[p] + 1
                c[u] = c[q]
                e[u] = e[q]
                while p != -1 and b[p].get(t) == q:
                    b[p][t] = u
                    p = c[p]
                c[q] = u
                c[r] = u
        
        v_g = r
        a = -1
        while v_g != -1:
            if d[v_g] > 0 and e[v_g] >= 0:
                if e[v_g] + 1 < n:
                    a = x[e[v_g] + 1]
                break
            v_g = c[v_g]
        
        y[i] = a
        v_g = g
        while v_g != -1 and e[v_g] < i:
            e[v_g] = i
            v_g = c[v_g]
        g = r
        
    # Clamp negative values to 0 to avoid embedding index error
    y[y < 0] = 0
    return torch.tensor(y, dtype=torch.long)

# ==============================================================================
# 2. 数据加载与处理 (Data Loading & Processing)
# ==============================================================================
class CopyTaskDataset(Dataset):
    """生成长距离复制任务的数据集"""
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 序列结构: [start, ...A..., key_pattern, target, ...B..., key_pattern, end]
        start_token = Config.VOCAB_SIZE - 1
        end_token = Config.VOCAB_SIZE - 2
        
        pattern_len = random.randint(4, 8)
        key_pattern = [random.randint(1, Config.VOCAB_SIZE - 3) for _ in range(pattern_len)]
        
        target_token = random.randint(1, Config.VOCAB_SIZE - 3)
        
        len_a = random.randint(10, 30)
        part_a = [random.randint(1, Config.VOCAB_SIZE - 3) for _ in range(len_a)]
        
        # 关键：中间部分的长度，用于测试长距离依赖
        len_b = Config.SEQ_LENGTH - (1 + len_a + pattern_len + 1 + pattern_len + 1) - 20
        part_b = [0] * len_b # 用0填充，作为无关内容

        sequence = [start_token] + part_a + key_pattern + [target_token] + part_b + key_pattern + [end_token]
        # 截断或填充到固定长度
        sequence = sequence[:Config.SEQ_LENGTH]
        sequence += [0] * (Config.SEQ_LENGTH - len(sequence))
        
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(target_token, dtype=torch.long)
        
        return x, y

# ==============================================================================
# 3. 模型架构 (Model Architecture)
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class BaselineTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.D_MODEL)
        self.pos_encoder = PositionalEncoding(Config.D_MODEL)
        encoder_layers = nn.TransformerEncoderLayer(Config.D_MODEL, Config.N_HEAD, Config.D_FF, Config.DROPOUT, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, Config.N_LAYERS)
        self.decoder = nn.Linear(Config.D_MODEL, Config.VOCAB_SIZE)

    def forward(self, src):
        x = self.embedding(src)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        # 使用最后一个token的输出来进行预测
        final_token_output = output[:, -1, :]
        return self.decoder(final_token_output)

class ROSATransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, Config.D_MODEL)
        # 为ROSA的输出创建一个独立的嵌入层
        self.rosa_embedding = nn.Embedding(Config.VOCAB_SIZE, Config.D_MODEL)
        self.pos_encoder = PositionalEncoding(Config.D_MODEL)
        encoder_layers = nn.TransformerEncoderLayer(Config.D_MODEL, Config.N_HEAD, Config.D_FF, Config.DROPOUT, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, Config.N_LAYERS)
        self.decoder = nn.Linear(Config.D_MODEL, Config.VOCAB_SIZE)

    def forward(self, src):
        # 1. 在CPU上对批次中的每个序列独立进行ROSA符号计算
        with torch.no_grad():
            # 使用列表推导来处理批次中的每个序列
            rosa_outputs = [rosa_numpy(s.cpu().numpy()) for s in src]
            # 将结果堆叠成一个批次张量
            rosa_output_batch = torch.stack(rosa_outputs).to(src.device)

        # 2. 获取词嵌入和ROSA嵌入
        word_emb = self.embedding(src)
        # rosa_output_batch 的维度现在是 (batch_size, seq_len-1)，与src相同
        rosa_emb = self.rosa_embedding(rosa_output_batch)

        # 3. 特征注入：将两者相加 (现在维度完全匹配)
        combined_emb = word_emb + rosa_emb

        # 4. 正常通过Transformer处理
        x = self.pos_encoder(combined_emb)
        output = self.transformer_encoder(x)
        final_token_output = output[:, -1, :]
        return self.decoder(final_token_output)

# ==============================================================================
# 4. 训练与验证 (Training & Validation)
# ==============================================================================
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            output = model(x)
            preds = torch.argmax(output, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

# ==============================================================================
# 5. 主执行逻辑 (Main Execution Logic)
# ==============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("--- 创建长距离复制任务数据集 ---")
    train_dataset = CopyTaskDataset(5000)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dataset = CopyTaskDataset(500)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    print("--- 初始化模型：基准Transformer vs ROSA增强Transformer ---")
    baseline_model = BaselineTransformer()
    rosa_model = ROSATransformer()

    optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=Config.LR)
    optimizer_rosa = optim.Adam(rosa_model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()

    print("\n--- 开始对比训练 ---")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch:02d}/{Config.NUM_EPOCHS} ---")
        
        loss_base = train(baseline_model, train_loader, optimizer_baseline, criterion)
        acc_base = validate(baseline_model, val_loader)
        print(f"  - Baseline Model: Train Loss: {loss_base:.4f} | Val Acc: {acc_base:.2f}%")

        loss_rosa = train(rosa_model, train_loader, optimizer_rosa, criterion)
        acc_rosa = validate(rosa_model, val_loader)
        print(f"  - ROSA Model    : Train Loss: {loss_rosa:.4f} | Val Acc: {acc_rosa:.2f}%")

    print("\n--- 训练完成 ---")
    final_acc_base = validate(baseline_model, val_loader)
    final_acc_rosa = validate(rosa_model, val_loader)

    print(f"\n最终准确率对比:")
    print(f"  - 基准Transformer: {final_acc_base:.2f}%")
    print(f"  - ROSA增强Transformer: {final_acc_rosa:.2f}%")
    
    if final_acc_rosa > final_acc_base + 5:
        print("\n结论：实验成功！ROSA增强模型在需要长距离精确记忆的任务上，性能显著优于基准模型。" )
    else:
        print("\n结论：实验未达到预期效果。ROSA的优势未能在此次实验中明确体现。" )

if __name__ == '__main__':
    main()
