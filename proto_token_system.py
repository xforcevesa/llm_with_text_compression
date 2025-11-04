import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import random
import math

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ProtoTokenEncoder(nn.Module):
    """
    自回归Transformer编码器，将输入序列压缩为proto-tokens
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_length, dropout=0.1):
        super(ProtoTokenEncoder, self).__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # 自回归Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出投影层，将隐藏状态映射为proto-tokens
        self.output_projection = nn.Linear(d_model, d_model)
        
        # 因果掩码确保自回归性质
        self.register_buffer("causal_mask", torch.triu(
            torch.ones(max_length, max_length) * float('-inf'), diagonal=1
        ))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_tokens):
        """
        输入: input_tokens [batch_size, seq_len]
        输出: proto_tokens [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = input_tokens.shape
        
        # 生成位置编码
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        
        # 词嵌入 + 位置编码
        token_embeds = self.token_embedding(input_tokens) * math.sqrt(self.d_model)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)
        
        # 应用因果Transformer
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        x = self.transformer(x, mask=causal_mask)
        
        # 投影到proto-token空间
        proto_tokens = self.output_projection(x)
        
        return proto_tokens

class ProtoTokenSystem(nn.Module):
    """
    完整的Proto-token系统：编码器 + 冻结LLM解码器
    """
    def __init__(self, model_name, max_length=256, d_model=512, nhead=8, num_layers=6):
        self.max_length = max_length
        
        # 加载冻结LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.frozen_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)
        
        # 冻结模型参数
        for param in self.frozen_model.parameters():
            param.requires_grad = False
        self.frozen_model.eval()
        
        # 获取模型维度
        if hasattr(self.frozen_model.config, 'hidden_size'):
            model_d_model = self.frozen_model.config.hidden_size
        else:
            model_d_model = d_model  # 默认值
        
        # 初始化编码器
        self.encoder = ProtoTokenEncoder(
            vocab_size=len(self.tokenizer),
            d_model=model_d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_length=max_length
        ).to(device)
        
        # 初始化可学习的M tokens（每个位置一个）
        self.M_tokens = nn.Parameter(torch.randn(max_length, model_d_model, device=device))
        
        self.vocab_size = len(self.tokenizer)
        
    def decode_with_proto_tokens(self, proto_token_e, seq_length):
        """
        使用proto-token e和M tokens通过冻结LLM解码
        
        参数:
            proto_token_e: [batch_size, d_model] 序列的压缩表示
            seq_length: 要生成的序列长度
            
        返回:
            logits: [batch_size, seq_length, vocab_size]
        """
        batch_size = proto_token_e.shape[0]
        
        # 构建输入序列: [e, M1, M2, ..., M_{seq_length-1}]
        # 复制M tokens以适应batch size和序列长度
        M_seq = self.M_tokens[:seq_length-1].unsqueeze(0).expand(batch_size, seq_length-1, -1)
        
        # 拼接proto-token e和M序列
        input_embeddings = torch.cat([
            proto_token_e.unsqueeze(1),  # [batch_size, 1, d_model]
            M_seq  # [batch_size, seq_length-1, d_model]
        ], dim=1)  # [batch_size, seq_length, d_model]
        
        # 通过冻结LLM前向传播（无attention mask，完全可见）
        outputs = self.frozen_model(
            inputs_embeds=input_embeddings,
            attention_mask=None  # 无mask，所有位置相互可见
        )
        
        return outputs.logits
    
    def forward(self, input_tokens, target_length=None):
        """
        完整的前向传播：编码输入序列，然后解码重建
        
        参数:
            input_tokens: [batch_size, seq_len] 输入token序列
            target_length: 要重建的序列长度（默认为输入长度）
            
        返回:
            all_logits: 每个位置m的重建logits列表
            all_proto_tokens: 所有proto-tokens
        """
        batch_size, seq_len = input_tokens.shape
        
        if target_length is None:
            target_length = seq_len
        
        # 1. 通过编码器获取所有proto-tokens
        all_proto_tokens = self.encoder(input_tokens)  # [batch_size, seq_len, d_model]
        
        all_logits = []
        
        # 2. 对每个位置m，使用E_m重建前m个token
        for m in range(1, min(seq_len, target_length) + 1):
            # 获取第m个proto-token（对应前m个token的压缩）
            proto_token_e_m = all_proto_tokens[:, m-1, :]  # [batch_size, d_model]
            
            # 使用冻结LLM解码，重建长度为m的序列
            logits_m = self.decode_with_proto_tokens(proto_token_e_m, m)
            all_logits.append(logits_m)
        
        return all_logits, all_proto_tokens
    
    def reconstruct_sequence(self, input_tokens):
        """重建整个序列（用于测试）"""
        self.encoder.eval()
        
        with torch.no_grad():
            all_logits, _ = self.forward(input_tokens)
            
            if not all_logits:
                return None
                
            # 取最后一个logits（对应完整序列重建）
            final_logits = all_logits[-1]  # [batch_size, seq_len, vocab_size]
            predicted_tokens = torch.argmax(final_logits, dim=-1)
            
            return predicted_tokens

def train_proto_token_system(system: ProtoTokenSystem):
    """训练完整的proto-token系统"""
    
    # 优化器（只训练编码器和M tokens）
    optimizer = AdamW([
        {'params': system.encoder.parameters()},
        {'params': [system.M_tokens]}
    ], lr=1e-4, weight_decay=0.01)
    
    # 训练参数
    batch_size = 8
    num_batches = 10000
    print_interval = 100
    
    # 词汇表大小
    vocab_size = system.vocab_size
    
    for batch_idx in range(num_batches):
        system.encoder.train()
        optimizer.zero_grad()
        
        # 生成随机训练数据
        seq_length = system.max_length
        input_tokens = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        
        # 前向传播
        all_logits, _ = system(input_tokens)
        
        # 计算多任务损失：每个位置m的重建损失
        total_loss = 0
        num_losses = 0
        
        for m, logits_m in enumerate(all_logits, 1):
            # logits_m形状: [batch_size, m, vocab_size]
            # 目标: 前m个token
            targets_m = input_tokens[:, :m]
            
            # 计算交叉熵损失
            loss_m = F.cross_entropy(
                logits_m.reshape(-1, vocab_size),
                targets_m.reshape(-1)
            )
            total_loss += loss_m
            num_losses += 1
        
        if num_losses > 0:
            average_loss = total_loss / num_losses
            
            # 反向传播
            average_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(system.encoder.parameters()) + [system.M_tokens], 
                max_norm=1.0
            )
            optimizer.step()
        
        # 打印训练信息
        if batch_idx % print_interval == 0:
            if num_losses > 0:
                print(f"Batch {batch_idx}, Loss: {average_loss.item():.4f}")
            else:
                print(f"Batch {batch_idx}, No loss computed")
            
            # 测试重建准确率
            if batch_idx % 500 == 0:
                test_reconstruction(system, vocab_size)
    return system

def test_reconstruction(system, vocab_size, test_texts=None):
    """测试重建准确率"""
    system.encoder.eval()
    
    if test_texts is None:
        # 生成测试序列
        test_sequences = []
        for _ in range(3):  # 测试3个序列
            seq_length = random.randint(15, 30)
            seq = torch.randint(0, vocab_size, (1, seq_length), device=device)
            test_sequences.append(seq)
    else:
        # 使用提供的文本测试
        test_sequences = []
        for text in test_texts:
            tokens = system.tokenizer.encode(text, return_tensors="pt").to(device)
            test_sequences.append(tokens)
    
    print("\n" + "="*50)
    print("重建测试结果:")
    print("="*50)
    
    total_accuracy = 0
    num_tests = 0
    
    for i, test_tokens in enumerate(test_sequences):
        with torch.no_grad():
            # 重建序列
            reconstructed = system.reconstruct_sequence(test_tokens)
            
            if reconstructed is not None:
                # 计算准确率
                seq_len = min(test_tokens.shape[1], reconstructed.shape[1])
                correct = (test_tokens[0, :seq_len] == reconstructed[0, :seq_len]).sum().item()
                accuracy = correct / seq_len * 100
                
                total_accuracy += accuracy
                num_tests += 1
                
                # 解码文本（如果可能）
                try:
                    original_text = system.tokenizer.decode(test_tokens[0], skip_special_tokens=True)
                    reconstructed_text = system.tokenizer.decode(reconstructed[0], skip_special_tokens=True)
                    
                    print(f"\n测试序列 {i+1}:")
                    print(f"原始: {original_text}")
                    print(f"重建: {reconstructed_text}")
                    print(f"准确率: {accuracy:.2f}%")
                except:
                    print(f"\n测试序列 {i+1}: 准确率: {accuracy:.2f}%")
    
    if num_tests > 0:
        avg_accuracy = total_accuracy / num_tests
        print(f"\n平均重建准确率: {avg_accuracy:.2f}%")
    
    print("="*50)
    system.encoder.train()

def interactive_test(system):
    """交互式测试"""
    system.encoder.eval()
    
    print("\n交互式测试模式（输入'quit'退出）")
    
    while True:
        text = input("\n请输入要测试的文本: ").strip()
        
        if text.lower() == 'quit':
            break
            
        if not text:
            continue
            
        try:
            # 编码输入文本
            input_tokens = system.tokenizer.encode(text, return_tensors="pt").to(device)
            
            # 重建
            with torch.no_grad():
                reconstructed = system.reconstruct_sequence(input_tokens)
                
                if reconstructed is not None:
                    # 计算准确率
                    seq_len = min(input_tokens.shape[1], reconstructed.shape[1])
                    correct = (input_tokens[0, :seq_len] == reconstructed[0, :seq_len]).sum().item()
                    accuracy = correct / seq_len * 100
                    
                    reconstructed_text = system.tokenizer.decode(reconstructed[0], skip_special_tokens=True)
                    
                    print(f"\n原始文本: {text}")
                    print(f"重建文本: {reconstructed_text}")
                    print(f"重建准确率: {accuracy:.2f}%")
                else:
                    print("重建失败")
                    
        except Exception as e:
            print(f"错误: {e}")

if __name__ == "__main__":
    # 训练系统
    print("开始训练Proto-token系统...")

    # 初始化系统
    model_name = "Qwen/Qwen3-0.6B"  # 可替换为其他模型
    system = ProtoTokenSystem(
        model_name=model_name,
        max_length=128,  # 最大序列长度
        d_model=512,
        nhead=8,
        num_layers=4
    )
    
    system = train_proto_token_system(system=system)
    
    # 最终测试
    print("\n训练完成，进行最终测试...")
    
    # 测试样例
    test_texts = [
        "This is a test of the proto-token system.",
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a longer sequence to test reconstruction accuracy."
    ]
    
    test_reconstruction(system, system.vocab_size, test_texts)
    
    # 交互式测试
    # interactive_test(system)
