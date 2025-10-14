import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练模型和tokenizer（这里使用GPT-2小型模型作为示例，论文中使用的是Pythia或Llama-3）
model_name = "RWKV/RWKV7-Goose-World2.9-0.4B-HF"  # 可以替换为"EleutherAI/pythia-160m"或"meta-llama/Llama-3.2-1B"等，但需要相应权限和资源
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

# 冻结模型参数，只训练proto-tokens
for param in model.parameters():
    param.requires_grad = False
model.eval()  # 设置为评估模式，但允许训练嵌入

# 添加pad token如果不存在（某些模型如GPT-2没有pad token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 定义目标文本（示例文本，来自论文的PG-19数据集或自定义）
target_text = 'This is an example text that we want to reconstruct using proto-tokens. It should be long enough to test the method.'
# Tokenize目标文本
target_tokens = tokenizer.encode(target_text, return_tensors="pt").squeeze(0).to(device)  # 形状: [N]
N = len(target_tokens)  # 目标序列长度
print(f"Target text length: {N} tokens")

# 初始化proto-tokens e和m（可训练嵌入）
d_model = model.config.hidden_size  # 模型隐藏层维度
proto_token_number = 1 # proto-token e number
e = [nn.Parameter(torch.randn(d_model, device=device)) for _ in range(proto_token_number)]  # proto-token e
m = nn.Parameter(torch.randn(d_model, device=device))  # proto-token m（论文中可共享，这里先不共享）

# 定义优化器，只优化e和m
optimizer = AdamW(e + [m], lr=0.001, betas=(0.9, 0.99), weight_decay=0.01)

# 训练参数
num_iterations = 10000  # 迭代次数（论文中使用5000，但为演示减少）
print_interval = 100  # 打印间隔

# 训练循环
for iteration in range(num_iterations):
    optimizer.zero_grad()
    
    # 构建输入序列：Z = [e, m, m, ..., m]（长度为N）
    # 第一个位置是e，其余N-1个位置是m
    input_embeddings = torch.stack(e + [m] * (N - proto_token_number))  # 形状: [N, d_model]
    
    # 由于我们使用自定义嵌入，需要创建attention mask（因果掩码）
    attention_mask = torch.tril(torch.ones(N, N, device=device))  # 下三角矩阵，形状: [N, N]
    
    # 通过模型前向传播（使用inputs_embeds而不是input_ids）
    outputs = model(inputs_embeds=input_embeddings.unsqueeze(0),  # 添加batch维度: [1, N, d_model]
                   )    # 添加batch维度: [1, N, N]
    
    logits = outputs.logits.squeeze(0)  # 形状: [N, vocab_size]
    
    # 计算交叉熵损失：对于每个位置i，logits[i]预测目标token target_tokens[i]
    loss = nn.CrossEntropyLoss()(logits, target_tokens)
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()
    
    # 打印损失
    if iteration % print_interval == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}")
        
    # 早停检查：如果损失接近0，提前停止（论文中使用完美重建精度）
    if loss.item() < 0.1:  # 阈值可根据需要调整
        print(f"Early stopping at iteration {iteration} due to low loss.")
        break

# 评估重建精度
with torch.no_grad():
    # 使用训练后的e和m构建输入
    input_embeddings = torch.stack(e + [m] * (N - proto_token_number))
    attention_mask = torch.tril(torch.ones(N, N, device=device))
    outputs = model(inputs_embeds=input_embeddings.unsqueeze(0))
    logits = outputs.logits.squeeze(0)
    predicted_tokens = torch.argmax(logits, dim=-1)
    
    # 计算正确令牌数
    correct_tokens = (predicted_tokens == target_tokens).sum().item()
    accuracy = correct_tokens / N
    print(f"Reconstruction accuracy: {accuracy * 100:.2f}%")
    
    # 解码预测文本
    predicted_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True)
    print(f"Predicted text: {predicted_text}")

# 保存训练后的proto-tokens（可选）
torch.save({'e': e, 'm': m}, 'proto_tokens.pth')