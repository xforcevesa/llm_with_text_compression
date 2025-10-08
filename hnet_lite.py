import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor  # (batch_size, seq_len, 2)
    boundary_mask: torch.Tensor  # (batch_size, seq_len)
    selected_probs: torch.Tensor  # (batch_size, seq_len, 1)

class RoutingModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        # 初始化权重为单位矩阵
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(d_model))
            self.k_proj.weight.copy_(torch.eye(d_model))

    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_len, d_model)
        # 计算查询和键投影
        q = F.normalize(self.q_proj(hidden_states[:, :-1]), dim=-1)  # 忽略最后一个位置
        k = F.normalize(self.k_proj(hidden_states[:, 1:]), dim=-1)  # 忽略第一个位置
        
        # 计算余弦相似度
        cos_sim = torch.einsum('b l d, b l d -> b l', q, k)
        boundary_prob = torch.clamp((1 - cos_sim) / 2, min=0.0, max=1.0)  # (batch_size, seq_len-1)
        
        # 填充第一个位置的概率为1.0（强制边界）
        boundary_prob = F.pad(boundary_prob, (1, 0), value=1.0)  # (batch_size, seq_len)
        
        # 创建二元概率分布
        boundary_prob = torch.stack([1 - boundary_prob, boundary_prob], dim=-1)  # (batch_size, seq_len, 2)
        
        # 选择边界掩码（概率大于0.5的位置）
        selected_idx = torch.argmax(boundary_prob, dim=-1)  # (batch_size, seq_len)
        boundary_mask = selected_idx == 1  # 边界位置为True
        
        # 选择概率值
        selected_probs = boundary_prob.gather(dim=-1, index=selected_idx.unsqueeze(-1))
        
        return RoutingModuleOutput(boundary_prob, boundary_mask, selected_probs)

class ChunkLayer(nn.Module):
    def forward(self, hidden_states, boundary_mask):
        # hidden_states: (batch_size, seq_len, d_model)
        # boundary_mask: (batch_size, seq_len)
        compressed_states = []
        for i in range(hidden_states.size(0)):
            # 选择边界位置的向量
            compressed = hidden_states[i, boundary_mask[i]]
            compressed_states.append(compressed)
        # 填充或截断以使批次中序列长度一致（简化处理）
        max_len = max(c.size(0) for c in compressed_states)
        compressed_padded = []
        for c in compressed_states:
            if c.size(0) < max_len:
                pad = torch.zeros(max_len - c.size(0), c.size(1), device=c.device)
                compressed_padded.append(torch.cat([c, pad], dim=0))
            else:
                compressed_padded.append(c[:max_len])
        compressed_out = torch.stack(compressed_padded, dim=0)  # (batch_size, compressed_seq_len, d_model)
        return compressed_out

class DeChunkLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, compressed_states, boundary_mask, boundary_prob):
        # compressed_states: (batch_size, compressed_seq_len, d_model)
        # boundary_mask: (batch_size, seq_len) - 原始序列长度的掩码
        # boundary_prob: (batch_size, seq_len, 2) - 边界概率
        batch_size, seq_len = boundary_mask.shape
        device = compressed_states.device
        
        # 创建输出张量
        output_states = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        for i in range(batch_size):
            comp_idx = 0
            for j in range(seq_len):
                if boundary_mask[i, j]:
                    # 边界位置：直接使用压缩向量
                    output_states[i, j] = compressed_states[i, comp_idx]
                    comp_idx += 1
                else:
                    # 非边界位置：使用EMA平滑
                    # 简化：使用前一个位置的输出进行线性插值
                    p = boundary_prob[i, j, 1]  # 边界概率
                    if j == 0:
                        prev_value = torch.zeros(self.d_model, device=device)
                    else:
                        prev_value = output_states[i, j-1]
                    current_value = compressed_states[i, comp_idx-1] if comp_idx > 0 else torch.zeros(self.d_model, device=device)
                    output_states[i, j] = p * current_value + (1 - p) * prev_value
        return output_states
    
class RMSNorm(nn.Module):
    """RMS归一化层，替代LayerNorm"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / norm * self.weight

class MLA(nn.Module):
    """多头隐性注意力机制（Multi-Head Latent Attention）"""
    def __init__(self, d_model=576, num_heads=9, kv_compression_ratio=1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_compression_dim = self.head_dim // kv_compression_ratio
        
        # 查询投影（使用低秩适配）
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 键值压缩投影
        self.kv_compression = nn.Linear(d_model, num_heads * self.kv_compression_dim * 2, bias=False)
        
        # 输出投影
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 丢弃层
        self.attn_dropout = nn.Dropout(0.1)
        self.res_dropout = nn.Dropout(0.1)
        
        # 旋转位置编码参数
        self.register_buffer('freqs_cis', self.precompute_freqs_cis(2048, self.head_dim))

    def precompute_freqs_cis(self, seq_len, head_dim):
        """预计算旋转位置编码"""
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(seq_len).float()
        freqs = torch.outer(positions, theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数形式
        return freqs_cis

    def apply_rotary_emb(self, x, freqs_cis):
        """应用旋转位置编码"""
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis[:x.shape[1]].view(1, x.shape[1], 1, -1)
        x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
        return x_rotated.type_as(x)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        identity = x
        
        # 应用RMSNorm
        x_norm = RMSNorm(self.d_model)(x)
        
        # 查询投影
        q = self.q_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 键值压缩和分解
        kv_compressed = self.kv_compression(x_norm).view(
            batch_size, seq_len, self.num_heads, 2 * self.kv_compression_dim
        )
        k_compressed, v_compressed = torch.split(kv_compressed, self.kv_compression_dim, dim=-1)
        
        # 应用旋转位置编码到查询
        q = self.apply_rotary_emb(q, self.freqs_cis)
        
        # 注意力计算（简化版，实际MLA有更复杂的低秩交互）
        attn_weights = torch.matmul(q, k_compressed.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_weights = attn_weights + mask[:, :, :seq_len, :seq_len]
        
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(x)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 注意力输出
        attn_output = torch.matmul(attn_weights, v_compressed)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影和残差连接
        output = self.o_proj(attn_output)
        output = self.res_dropout(output)
        
        return output + identity

class Expert(nn.Module):
    """单个专家网络"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    """混合专家层（简化版）"""
    def __init__(self, d_model=576, d_ff=1536, num_experts=8, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 专家网络
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        
        # 门控网络
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # 共享专家（DeepSeek-V2特色）
        self.shared_expert = Expert(d_model, d_ff)

    def forward(self, x):
        identity = x
        batch_size, seq_len, _ = x.shape
        
        # 应用RMSNorm
        x_norm = RMSNorm(self.d_model)(x)
        
        # 门控逻辑
        gate_logits = self.gate(x_norm)  # [batch_size, seq_len, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # Top-k专家选择
        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # 初始化输出
        output = torch.zeros_like(x)
        
        # 专家计算（简化路由，实际有更复杂的负载均衡）
        for i, expert in enumerate(self.experts):
            expert_mask = (topk_indices == i).any(dim=-1)
            if expert_mask.any():
                expert_input = x_norm[expert_mask]
                expert_output = expert(expert_input)
                # 加权聚合
                weight_mask = topk_weights[expert_mask] * (topk_indices[expert_mask] == i).float()
                expert_weight = weight_mask.sum(dim=-1, keepdim=True)
                output[expert_mask] += expert_output * expert_weight
        
        # 添加共享专家输出
        shared_output = self.shared_expert(x_norm)
        output = output + shared_output
        
        return output + identity

class DeepSeekBlock(nn.Module):
    """完整的DeepSeek Transformer Block"""
    def __init__(self, d_model=576, num_heads=9, d_ff=1536, num_experts=8, top_k=2):
        super().__init__()
        self.mla = MLA(d_model, num_heads)
        self.moe = MoE(d_model, d_ff, num_experts, top_k)
        
    def forward(self, x, attention_mask=None):
        # MLA注意力部分
        x = self.mla(x, attention_mask)
        
        # MoE前馈网络部分
        x = self.moe(x)
        
        return x

class Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.block = DeepSeekBlock(d_model=d_model, num_heads=4, d_ff=d_model*4, num_experts=4)
    
    def forward(self, x):
        return self.block(x)

class MainNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.block = DeepSeekBlock(d_model=d_model, num_heads=4, d_ff=d_model*4, num_experts=4)
    
    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.block = DeepSeekBlock(d_model=d_model, num_heads=4, d_ff=d_model*4, num_experts=4)
    
    def forward(self, x):
        return self.block(x)
    
import numpy as np


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.dtype = np.uint8

    def encode(self, seqs: list[str], add_bos: bool = False, add_eos: bool = False, **kwargs) -> list[dict[str, np.ndarray]]:
        total_outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")

            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])
            text_byte = bytearray(text_byte)
            text_byte_ids = np.array(text_byte, dtype=self.dtype)

            total_outputs.append({"input_ids": text_byte_ids})

        return total_outputs

    def decode(self, tokens: np.ndarray | list[int], **kwargs) -> str:
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return bytearray(tokens).decode("utf-8", **kwargs)

# H-Net整体模型（单级层次）
class HNet(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder = Encoder(d_model)
        self.router = RoutingModule(d_model)
        self.chunk = ChunkLayer()
        self.main_net = MainNetwork(d_model)
        self.dechunk = DeChunkLayer(d_model)
        self.decoder = Decoder(d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # 编码器处理
        enc_output = self.encoder(x)  # (batch_size, seq_len, d_model)

        # "This is an apple."
        # ByteTokenizer
        # ['T', 'h', 'i', 's', ' ', ...]
        # RouterModule + Chunking - Dynamic Chunking
        # [['T', 'h', 'i', 's'], ]
        # Dechunk
        # ['T', 'h', 'i', 's', ' ', ...]

        
        # 动态分块：计算边界
        router_output = self.router(enc_output)
        boundary_prob, boundary_mask, selected_probs = router_output.boundary_prob, router_output.boundary_mask, router_output.selected_probs
        
        # 压缩序列
        compressed = self.chunk(enc_output, boundary_mask)  # (batch_size, compressed_seq_len, d_model)
        
        # 主网络处理
        main_output = self.main_net(compressed)  # (batch_size, compressed_seq_len, d_model)
        
        # 反压缩
        dechunk_output = self.dechunk(main_output, boundary_mask, boundary_prob)  # (batch_size, seq_len, d_model)
        
        # 解码器处理
        dec_output = self.decoder(dechunk_output)  # (batch_size, seq_len, d_model)
        
        return dec_output
    
from datasets import load_dataset
from transformers import AutoTokenizer
    
# ------------------ 训练代码 ------------------ 
def train_hnet():
    # 配置参数
    d_model = 576
    num_stages = 2
    batch_size = 4
    seq_len = 512
    learning_rate = 5e-4
    num_epochs = 3
    train_steps = 1000  # 简化训练步数

    # 加载 FineWeb-Edu 数据集
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=seq_len, padding="max_length")

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型和优化器
    model = HNet(d_model=d_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            if step >= train_steps:
                break
                
            inputs = batch["input_ids"]
            mask = batch["attention_mask"]
            
            optimizer.zero_grad()
            outputs = model(inputs, mask=mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), inputs.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

# 示例使用
if __name__ == "__main__":
    d_model = 64
    batch_size = 2
    seq_len = 10
    model = HNet(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")