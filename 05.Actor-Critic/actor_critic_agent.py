# coding:utf-8


""" 方策に基づく手法 (Policy-based Methods)
    今回は，方策に基づく代表的な手法として，方策と価値関数の近似した
    Actor-Criticを扱います．
"""

# 必要なライブラリのインポート．
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

""" Actor-Critic
    今回は，方策(Actor)と価値関数(Critic)の両方をニューラルネットワークで近似します．
"""

# ActorとCriticのネットワーク(一部の重みを共有しています)．
class ActorCriticNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=16):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2a = nn.Linear(hidden_size, num_action)  # Actor独自のlayer．
        self.fc2c = nn.Linear(hidden_size, 1)           # Critic独自のlayer．
    
    def forward(self, x):
        h = F.elu(self.fc1(x))
        action_prob = F.softmax(self.fc2a(h), dim=-1)
        state_value = self.fc2c(h)
        return action_prob, state_value

# Actor-Criticエージェントのクラス．
class ActorCriticAgent:
    def __init__(self, num_state, num_action, gamma=0.99, lr=0.001):
        self.num_state = num_state
        self.gamma = gamma  # 割引率．
        self.acnet = ActorCriticNetwork(num_state, num_action)
        self.optimizer = optim.Adam(self.acnet.parameters(), lr=lr)
        self.memory = []  # （報酬，行動選択確率，状態価値）のtupleをlistで保存．
        
    # 方策を更新．
    def update_policy(self):
        R = 0
        actor_loss = 0
        critic_loss = 0
        for r, prob, v in self.memory[::-1]:
            R = r + self.gamma * R
            advantage = R - v
            actor_loss -= torch.log(prob) * advantage
            critic_loss += F.smooth_l1_loss(v, torch.tensor(R))
        actor_loss = actor_loss/len(self.memory)
        critic_loss = critic_loss/len(self.memory)
        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()
    
    # softmaxの出力が最も大きい行動を選択．
    def get_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action_prob, _ = self.acnet(state_tensor.data)
        action = torch.argmax(action_prob.squeeze().data).item()
        return action
        
    # カテゴリカル分布からサンプリングして行動を選択．
    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action_prob, state_value = self.acnet(state_tensor.data)
        action_prob, state_value = action_prob.squeeze(), state_value.squeeze()
        action = Categorical(action_prob).sample().item()
        return action, action_prob[action], state_value
    
    def add_memory(self, r, prob, v):
        self.memory.append((r, prob, v))
    
    def reset_memory(self):
        self.memory = []