import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class Seq2SeqCompression(nn.Module):
    """
    基于LSTM的序列到序列文本压缩模型
    编码器将输入序列压缩为隐藏状态，解码器从该状态还原序列
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 start_token, end_token, max_length, teacher_forcing_ratio=0.5):
        super(Seq2SeqCompression, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.start_token = start_token
        self.end_token = end_token
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # 共享嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 编码器：多层LSTM
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # 解码器：多层LSTM
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # 隐藏状态转换层（每层一个线性变换）
        self.hidden_transform = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def encode(self, input_seq):
        """编码器：将输入序列压缩为隐藏状态"""
        # 嵌入层转换
        embedded = self.embedding(input_seq)  # shape: (batch_size, seq_len, embedding_dim)
        
        # LSTM编码
        encoder_outputs, (hidden, cell) = self.encoder(embedded)
        
        # 对每层的隐藏状态进行线性变换[6](@ref)
        transformed_hidden = []
        transformed_cell = []
        for i in range(self.num_layers):
            transformed_hidden.append(self.hidden_transform[i](hidden[i]))
            transformed_cell.append(self.hidden_transform[i](cell[i]))
        
        # 堆叠回原来的形状
        hidden = torch.stack(transformed_hidden)
        cell = torch.stack(transformed_cell)
        
        return encoder_outputs, (hidden, cell)
    
    def decode(self, decoder_input, hidden_state, cell_state, target_seq=None):
        """解码器：从隐藏状态还原序列"""
        batch_size = decoder_input.size(0)
        seq_length = self.max_length if target_seq is None else target_seq.size(1)
        
        # 存储解码器输出
        outputs = torch.zeros(batch_size, seq_length, self.vocab_size)
        
        # 初始输入是开始令牌
        decoder_input = decoder_input.unsqueeze(1)  # 添加序列维度
        
        for t in range(seq_length):
            # 解码一步[6](@ref)
            decoder_output, (hidden_state, cell_state) = self.decoder(
                decoder_input, (hidden_state, cell_state)
            )
            
            # 通过输出层得到词汇表分布
            output = self.output_layer(decoder_output.squeeze(1))
            outputs[:, t, :] = output

            top_token = None
            
            # 决定下一个输入：教师强制或自己的预测[6](@ref)
            if target_seq is not None and random.random() < self.teacher_forcing_ratio:
                # 教师强制：使用真实的下一个令牌
                decoder_input = target_seq[:, t].unsqueeze(1)
            else:
                # 使用自己的预测
                top_token = output.argmax(1)
                decoder_input = top_token.unsqueeze(1)
            
            # 嵌入转换
            decoder_input = self.embedding(decoder_input)
            
            # 如果预测到结束令牌，提前终止[3](@ref)
            if top_token is None or (top_token == self.end_token).all():
                break
                
        return outputs, (hidden_state, cell_state)
    
    def forward(self, input_seq, target_seq=None):
        """前向传播"""
        batch_size = input_seq.size(0)
        
        # 编码阶段
        _, (hidden, cell) = self.encode(input_seq)
        
        # 解码器初始输入是开始令牌[6](@ref)
        decoder_input = torch.full((batch_size,), self.start_token, dtype=torch.long)
        decoder_input = self.embedding(decoder_input)
        
        # 解码阶段
        outputs, _ = self.decode(decoder_input, hidden, cell, target_seq)
        
        return outputs

class RandomSequenceDataset(Dataset):
    """随机整数序列数据集，用于训练和测试"""
    def __init__(self, num_samples, seq_length, vocab_size, start_token, end_token):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机序列（避开特殊令牌）
        seq = torch.randint(2, self.vocab_size-2, (self.seq_length,))
        return seq, seq  # 输入和目标相同（自编码器）[3](@ref)

def train_model(model, dataloader, criterion, optimizer, device):
    """训练模型"""
    model.train()
    total_loss = 0
    
    for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(input_seq, target_seq)
        
        # 计算损失（忽略填充位置）
        loss = criterion(
            outputs.view(-1, model.vocab_size),
            target_seq.view(-1)
        )
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸[6](@ref)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    """评估模型准确率"""
    model.eval()
    total_tokens = 0
    correct_tokens = 0
    
    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # 推理（不使用教师强制）
            outputs = model(input_seq)
            
            # 获取预测结果
            predictions = outputs.argmax(dim=-1)
            
            # 计算准确率
            mask = (target_seq != 0)  # 忽略填充位置
            total_tokens += mask.sum().item()
            correct_tokens += ((predictions == target_seq) & mask).sum().item()
    
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return accuracy

def main():
    # 超参数设置（可根据需要调整）
    VOCAB_SIZE = 1000       # 词汇表大小（包括特殊令牌）
    EMBEDDING_DIM = 128    # 嵌入维度
    HIDDEN_DIM = 256       # LSTM隐藏层维度
    NUM_LAYERS = 2         # LSTM层数
    START_TOKEN = 0        # 开始令牌
    END_TOKEN = 1          # 结束令牌
    MAX_LENGTH = 20        # 序列最大长度
    BATCH_SIZE = 32        # 批大小
    NUM_EPOCHS = 50        # 训练轮数
    LEARNING_RATE = 0.001  # 学习率
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = Seq2SeqCompression(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        start_token=START_TOKEN,
        end_token=END_TOKEN,
        max_length=MAX_LENGTH
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数: {total_params:,}")
    
    # 创建数据集和数据加载器
    train_dataset = RandomSequenceDataset(
        num_samples=10000, 
        seq_length=MAX_LENGTH, 
        vocab_size=VOCAB_SIZE,
        start_token=START_TOKEN,
        end_token=END_TOKEN
    )
    
    test_dataset = RandomSequenceDataset(
        num_samples=1000, 
        seq_length=MAX_LENGTH, 
        vocab_size=VOCAB_SIZE,
        start_token=START_TOKEN,
        end_token=END_TOKEN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充位置的损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    print("开始训练...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        
        # 每10轮评估一次
        if (epoch + 1) % 1 == 0:
            accuracy = evaluate_model(model, test_loader, device)
            print(f"轮次 [{epoch+1}/{NUM_EPOCHS}], 训练损失: {train_loss:.4f}, "
                  f"测试准确率: {accuracy*100:.2f}%")
    
    # 最终评估
    final_accuracy = evaluate_model(model, test_loader, device)
    print(f"最终测试准确率: {final_accuracy*100:.2f}%")
    
    # 演示压缩和还原过程
    print("\n--- 演示压缩和还原过程 ---")
    model.eval()
    with torch.no_grad():
        # 创建随机测试序列
        test_input = torch.randint(2, VOCAB_SIZE-2, (1, MAX_LENGTH))
        print(f"原始序列: {test_input.squeeze().tolist()}")
        
        # 编码（压缩）
        _, (hidden, cell) = model.encode(test_input.to(device))
        print("序列已压缩为LSTM隐藏状态")
        
        # 解码（还原）
        decoder_input = torch.full((1,), START_TOKEN, dtype=torch.long)
        decoder_input = model.embedding(decoder_input)
        outputs, _ = model.decode(decoder_input.to(device), hidden, cell)
        
        # 获取预测结果
        predictions = outputs.argmax(dim=-1)
        print(f"还原序列: {predictions.squeeze().tolist()}")
        
        # 计算匹配度
        match_count = (test_input.to(device) == predictions).sum().item()
        match_rate = match_count / MAX_LENGTH
        print(f"令牌匹配率: {match_rate*100:.2f}%")

if __name__ == "__main__":
    embedding_dim = 32
    hidden_dim = 16
    num_layers = 2
    lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
    x = torch.randn(10, 16, embedding_dim)
    y, _ = lstm(x)
    print(x.shape, y.shape)

    main()
