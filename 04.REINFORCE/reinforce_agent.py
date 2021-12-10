# coding:utf-8


""" 方策に基づく手法 (Policy-based Methods)
    今回は，方策に基づく代表的な手法として，REINFORCEを扱います．
"""

# 必要なライブラリのインポート．
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

""" REINFORCE
    今回は，方策を直接ニューラルネットワークで関数近似します．
    (実際には，方策と価値関数の両方を関数近似するactor-criticの手法を用いることも多いです．)
    REINFORCEでは，確率的な方策を採用しています．
"""

# 方策のネットワークの定義．
class PolicyNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=16):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_action)
    
    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        action_prob = F.softmax(self.fc3(h), dim=-1)
        return action_prob

# REINFORCEエージェントのクラス．
class ReinforceAgent:
    def __init__(self, num_state, num_action, gamma=0.99, lr=0.001):
        self.num_state = num_state
        self.gamma = gamma  # 割引率．
        self.pinet = PolicyNetwork(num_state, num_action)
        self.optimizer = optim.Adam(self.pinet.parameters(), lr=lr)
        self.memory = []  # 報酬とそのときの行動選択確率のtupleをlistで保存．
    
    # 方策を更新．
    def update_policy(self):
        R = 0
        loss = 0
        # エピソード内の各ステップの収益を後ろから計算．
        for r, prob in self.memory[::-1]:
            R = r + self.gamma * R
            loss -= torch.log(prob) * R
        loss = loss/len(self.memory)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    # softmaxの出力が最も大きい行動を選択．
    def get_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action_prob = self.pinet(state_tensor.data).squeeze()
        action = torch.argmax(action_prob.data).item()
        return action
    
    # カテゴリカル分布からサンプリングして行動を選択．
    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action_prob = self.pinet(state_tensor.data).squeeze()
        action = Categorical(action_prob).sample().item()
        return action, action_prob[action]
    
    def add_memory(self, r, prob):
        self.memory.append((r, prob))
    
    def reset_memory(self):
        self.memory = []