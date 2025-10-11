import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# ==============================================================================
# 0. 配置与超参数 (Configuration)
# ==============================================================================
class Config:
    # 环境设置
    ENV_SIZE = 20  # 走廊长度
    TARGET_POS = 15 # 终点位置

    # 模型设置
    INPUT_SIZE = 1   # 状态是1维的（当前位置）
    HIDDEN_SIZE = 64 # RNN的隐藏层大小
    ACTION_SIZE = 3  # 动作空间：0:左, 1:不动, 2:右

    # 训练设置
    NUM_EPISODES = 1500
    MAX_STEPS_PER_EPISODE = 100
    LR = 0.001
    GAMMA = 0.98 # RL的折扣因子
    LAMBDA_PC = 0.5 # 预测编码损失的权重

# ==============================================================================
# 1. 简易环境 (Simple 1D Environment)
# ==============================================================================
class Simple1DEnv:
    def __init__(self):
        self.size = Config.ENV_SIZE
        self.target_pos = Config.TARGET_POS
        self.agent_pos = 0

    def reset(self):
        self.agent_pos = np.random.randint(0, self.size)
        return self.agent_pos

    def step(self, action):
        # 0: left, 1: stay, 2: right
        if action == 0:
            self.agent_pos -= 1
        elif action == 2:
            self.agent_pos += 1
        
        # 边界处理
        self.agent_pos = np.clip(self.agent_pos, 0, self.size - 1)

        reward = -0.1 # 每走一步都有轻微的惩罚，鼓励效率
        done = False
        if self.agent_pos == self.target_pos:
            reward = 10.0 # 到达终点获得巨大奖励
            done = True
            
        return self.agent_pos, reward, done

# ==============================================================================
# 2. “预测性能动者”模型架构 (Predictive Actor Architecture)
# ==============================================================================
class PredictiveActor(nn.Module):
    def __init__(self):
        super(PredictiveActor, self).__init__()
        # 1. 世界模型 (RNN) + 预测编码头
        self.rnn = nn.GRUCell(Config.INPUT_SIZE, Config.HIDDEN_SIZE)
        self.prediction_head = nn.Linear(Config.HIDDEN_SIZE, Config.INPUT_SIZE)
        
        # 2. 决策模型 (Actor-Critic)
        self.actor_head = nn.Linear(Config.HIDDEN_SIZE, Config.ACTION_SIZE)
        self.critic_head = nn.Linear(Config.HIDDEN_SIZE, 1)

    def forward(self, x, h):
        # x: 当前状态 [1, 1]
        # h: 上一时刻的隐藏状态 [1, HIDDEN_SIZE]
        
        # 更新世界模型的信念 (RNN)
        h_next = self.rnn(x, h)
        
        # 产生决策 (Actor-Critic)
        action_logits = self.actor_head(h_next)
        state_value = self.critic_head(h_next)
        
        # 产生对下一状态的预测 (Predictive Coding)
        predicted_next_x = self.prediction_head(h_next)
        
        return action_logits, state_value, predicted_next_x, h_next

# ==============================================================================
# 3. 主执行逻辑 (Main Execution Logic)
# ==============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    env = Simple1DEnv()
    model = PredictiveActor()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    mse_loss = nn.MSELoss()

    print("--- 开始在线强化学习与预测编码的联合训练 ---")
    total_rewards = []

    for episode in range(Config.NUM_EPISODES):
        # 初始化环境和隐藏状态
        state = env.reset()
        h = torch.zeros(1, Config.HIDDEN_SIZE)
        done = False
        episode_reward = 0
        
        for step in range(Config.MAX_STEPS_PER_EPISODE):
            # 1. 模型决策与预测
            state_tensor = torch.tensor([[state]], dtype=torch.float)
            logits, value, pred_next_state, h_next = model(state_tensor, h)

            # 2. 选择并执行动作
            prob = F.softmax(logits, dim=-1)
            m = Categorical(prob)
            action = m.sample().item()
            next_state, reward, done = env.step(action)
            episode_reward += reward

            # 3. 在线学习：计算联合损失并更新
            next_state_tensor = torch.tensor([[next_state]], dtype=torch.float)
            
            # 3.1 预测编码损失 (Loss_PC)
            # 模型的世界观是否准确？
            loss_pc = mse_loss(pred_next_state, next_state_tensor)

            # 3.2 强化学习损失 (Loss_RL)
            # 这个动作是好是坏？
            with torch.no_grad():
                _, next_value, _, _ = model(next_state_tensor, h_next)
                td_target = reward + Config.GAMMA * next_value * (1 - done)
            
            td_error = td_target - value
            
            loss_actor = -m.log_prob(torch.tensor(action)) * td_error.detach()
            loss_critic = F.smooth_l1_loss(value, td_target.detach())

            # 3.3 联合总损失
            total_loss = loss_actor + loss_critic + Config.LAMBDA_PC * loss_pc

            # 4. 执行在线更新
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 更新状态
            state = next_state
            h = h_next.detach() # 分离隐藏状态，使其不参与跨时间步的梯度计算

            if done:
                break
        
        total_rewards.append(episode_reward)
        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(f"Episode: {episode:04d} | Average Reward (last 100): {avg_reward:.2f}")

    print("\n--- 训练完成 ---")
    print(f"最终100个episodes的平均奖励: {np.mean(total_rewards[-100:]):.2f}")

if __name__ == '__main__':
    main()
