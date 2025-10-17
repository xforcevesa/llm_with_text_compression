import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple
import os

import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Songti SC', 'STFangsong', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 或者使用更全面的字体列表
plt.rcParams['font.sans-serif'] = [
    'Songti SC',     # 宋体
    'STFangsong',    # 仿宋
    'STHeiti',       # 黑体
    'Kaiti SC',      # 楷体
    'PingFang SC',   # 苹方
    'Hiragino Sans GB' # 华文黑体
]
plt.rcParams['axes.unicode_minus'] = False

def calculate_effective_receptive_field(
    model: AutoModelForCausalLM,
    model_name: str,
    seq_length: int = 1024,
    hidden_size: Optional[int] = None,
    output_path: str = "./erf_plot.png",
    dpi: int = 300
) -> None:
    """
    计算并绘制语言模型的有效感受野
    
    Args:
        model: 加载的AutoModelForCausalLM模型
        model_name: 模型名称（用于图表标题）
        seq_length: 序列长度
        hidden_size: 隐藏层维度，如果为None则自动检测
        output_path: 输出图片路径
        dpi: 图片分辨率
    """
    
    # 自动检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    embedding_parameter = None
    
    # 自动检测隐藏层大小
    if hidden_size is None:
        if hasattr(model.config, 'hidden_size'):
            hidden_size = model.config.hidden_size
        elif hasattr(model.config, 'd_model'):
            hidden_size = model.config.d_model
        else:
            # 尝试从第一个线性层获取隐藏大小
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    hidden_size = module.in_features
                    break
            if hidden_size is None:
                raise ValueError("无法自动检测hidden_size，请手动指定")
        
    # 获取Embedding Weight
    for a, b in model.named_parameters():
        if b.shape[0] >= tokenizer.vocab_size and b.shape[-1] == hidden_size:
            print(a, list(b.shape), 'YES')
            embedding_parameter = b.detach()
    
    print(f"使用设备: {device}")
    print(f"序列长度: {seq_length}")
    print(f"隐藏层大小: {hidden_size}")
    
    # 创建输入embedding（批量大小=1，序列长度，隐藏层大小）
    # 使用正态分布初始化，确保数值稳定性
    # input_embeds = torch.randn(1, seq_length, hidden_size, device=device, requires_grad=True)
    # 使用随机Token输入，并转化成Embedding向量
    input_tokens = torch.randint(size=(1, seq_length), high=tokenizer.vocab_size - 100, device=device, low=100)
    input_embeds = F.embedding(input_tokens, embedding_parameter).detach().requires_grad_(True)
    
    # 前向传播
    if True:
        # 获取模型输出
        outputs = model(inputs_embeds=input_embeds, output_hidden_states=True)
        if True:
            # 修复：正确处理不同类型的输出对象
            if hasattr(outputs, 'hidden_states'):
                # 如果有hidden_states属性，取最后一层的隐藏状态
                last_hidden_state = outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                # 兼容旧版本API
                last_hidden_state = outputs.last_hidden_state
            else:
                # 如果都没有，尝试通过索引访问
                last_hidden_state = outputs[0] if isinstance(outputs, (tuple, list)) else None
                
            if last_hidden_state is None:
                raise ValueError("无法从模型输出中获取隐藏状态")
    
    # 选择目标位置（序列末尾）
    target_position = seq_length - 1
    target_logit = last_hidden_state[0, target_position, :].sum()  # 对所有特征维度求和
    
    # 计算梯度
    model.zero_grad()
    target_logit.backward()
    
    # 获取输入embedding的梯度
    grad = input_embeds.grad.data[0]  # 形状: (seq_length, hidden_size)
    
    # 计算每个位置的重要性（梯度范数）
    importance = torch.norm(grad, dim=1)  # 形状: (seq_length,)
    
    # 转换为numpy数组用于绘图
    importance_np = importance.cpu().numpy()
    positions = np.arange(seq_length)
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制原始重要性曲线
    plt.subplot(2, 1, 1)
    plt.plot(positions, importance_np, 'b-', linewidth=2, label='梯度重要性')
    plt.xlabel('输入位置')
    plt.ylabel('梯度范数')
    plt.title(f'{model_name} - 有效感受野分析\n目标位置: {target_position}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 绘制累积重要性曲线
    plt.subplot(2, 1, 2)
    cumulative_importance = np.cumsum(importance_np)
    cumulative_importance /= cumulative_importance[-1]  # 归一化
    
    plt.plot(positions, cumulative_importance, 'r-', linewidth=2, label='累积重要性')
    plt.xlabel('输入位置')
    plt.ylabel('累积梯度范数（归一化）')
    plt.title('累积有效感受野')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加重要统计信息
    threshold_50 = np.where(cumulative_importance >= 0.5)[0]
    threshold_90 = np.where(cumulative_importance >= 0.9)[0]
    
    erf_50 = threshold_50[0] if len(threshold_50) > 0 else seq_length
    erf_90 = threshold_90[0] if len(threshold_90) > 0 else seq_length
    
    plt.figtext(0.02, 0.02, 
                f'有效感受野统计:\n'
                f'50% 重要性位置: {erf_50}\n'
                f'90% 重要性位置: {erf_90}\n'
                f'最大重要性位置: {np.argmax(importance_np)}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"有效感受野分析完成!")
    print(f"50% 重要性位置: {erf_50}")
    print(f"90% 重要性位置: {erf_90}")
    print(f"图表已保存至: {output_path}")

def load_model_and_analyze_erf(
    model_name: str = "Qwen/Qwen3-0.6B",
    seq_length: int = 1024,
    output_dir: str = "./erf_results"
) -> None:
    """
    加载模型并分析有效感受野
    
    Args:
        model_name: 模型名称或路径
        seq_length: 序列长度
        output_dir: 输出目录
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"erf_{model_name.replace('/', '_')}.png")
    
    print(f"正在加载模型: {model_name}")
    
    try:
        # 加载模型（使用bfloat16以节省显存）
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        print("模型加载成功!")
        
        # 分析有效感受野
        calculate_effective_receptive_field(
            model=model,
            model_name=model_name,
            seq_length=seq_length,
            output_path=output_path
        )
        
    except Exception as e:
        print(f"错误: {e}")
        # 尝试使用float32加载
        print("尝试使用float32加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        calculate_effective_receptive_field(
            model=model,
            model_name=model_name,
            seq_length=seq_length,
            output_path=output_path
        )

# 针对Qwen3-0.6B的专用函数
def analyze_qwen3_erf(
    model_size: str = "0.6B",
    seq_length: int = 1024,
    output_dir: str = "./qwen_erf_results"
) -> None:
    """
    专门针对Qwen3系列模型的有效感受野分析
    
    Args:
        model_size: 模型大小 ('0.6B', '1.7B', '4B', 等)
        seq_length: 序列长度
        output_dir: 输出目录
    """
    
    model_name = f"Qwen/Qwen3-{model_size}"
    
    # 根据模型大小调整序列长度
    if model_size in ["0.6B", "1.7B"]:
        max_seq_len = 32768  # 小模型支持更长的上下文[3](@ref)
    else:
        max_seq_len = 131072  # 大模型支持更长的上下文
    
    seq_length = min(seq_length, max_seq_len)
    
    print(f"分析Qwen3-{model_size}的有效感受野...")
    print(f"最大支持序列长度: {max_seq_len}")
    print(f"实际使用序列长度: {seq_length}")
    
    load_model_and_analyze_erf(model_name, seq_length, output_dir)

# 使用示例
if __name__ == "__main__":
    # 示例1: 分析Qwen3-0.6B
    analyze_qwen3_erf("0.6B", seq_length=512)
    
    # 示例2: 分析任意模型
    load_model_and_analyze_erf("gpt2", seq_length=512)
    
    # 示例3: 自定义分析
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    # calculate_effective_receptive_field(model, "Qwen3-0.6B", seq_length=1024)