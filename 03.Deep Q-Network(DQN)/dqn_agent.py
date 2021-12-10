# coding:utf-8


""" 価値に基づく手法 (Value-based Methods)
    今回は，価値に基づく代表的な手法として，
    ニューラルネットワークを用いて価値関数を近似した手法である．
    Deep Q-Network(DQN)を扱います．
"""

# 必要なライブラリのインポート．
import numpy as np
import copy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

""" Deep Q-Network(DQN)
    価値関数をニューラルネットワークによって近似することで，
    状態空間の離散化を行わずに連続値のまま扱うことができます．

    ・経験リプレイ(experience replay)の利用
    　リプレイバッファ(replay buffer)にこれまでの状態遷移を記録しておき，
    　そこからサンプルすることでQ関数のミニバッチ学習をします．

    ・固定したターゲットネットワーク(fixed target Q-network)の利用
    　学習する対象のQ関数と更新によって近づける目標値は，
    　同じパラメータを持つQ関数を利用しているため，そのまま勾配法による最適化を行うと，
    　元のQ値と目標値の両方が更新されてしまいます．
    　これを避けるために，目標値は固定した上でQ関数の最適化を行います．
"""

# Q関数の定義．
class QNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=16):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_action)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        h = F.elu(self.fc3(h))
        y = F.elu(self.fc4(h))
        return y

# リプレイバッファの定義．
class ReplayBuffer:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque([], maxlen = memory_size)
    
    def append(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        batch_indexes = np.random.randint(0, len(self.memory), size=batch_size)
        states      = np.array([self.memory[index]['state'] for index in batch_indexes])
        next_states = np.array([self.memory[index]['next_state'] for index in batch_indexes])
        rewards     = np.array([self.memory[index]['reward'] for index in batch_indexes])
        actions     = np.array([self.memory[index]['action'] for index in batch_indexes])
        dones   = np.array([self.memory[index]['done'] for index in batch_indexes])
        return {'states': states, 'next_states': next_states, 'rewards': rewards, 'actions': actions, 'dones': dones}

# DQNエージェントのクラス．
class DqnAgent:
    def __init__(self, num_state, num_action, gamma=0.99, lr=0.001, batch_size=32, memory_size=50000):
        self.num_state = num_state
        self.num_action = num_action
        self.gamma = gamma                          # 割引率．
        self.batch_size = batch_size                # Q関数の更新に用いる遷移の数．
        self.qnet = QNetwork(num_state, num_action)
        self.target_qnet = copy.deepcopy(self.qnet) # ターゲットネットワーク．
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(memory_size)
    
    # Q関数を更新．
    def update_q(self):
        batch = self.replay_buffer.sample(self.batch_size)
        q = self.qnet(torch.tensor(batch["states"], dtype=torch.float))
        targetq = copy.deepcopy(q.data.numpy())

        # maxQの計算．
        maxq = torch.max(self.target_qnet(torch.tensor(batch["next_states"],dtype=torch.float)), dim=1).values

        # targetqのなかで，バッチのなかで実際に選択されていた行動 batch["actions"][i] に対応する要素に対して，Q値のターゲットを計算してセット．
        # 注意：選択されていない行動のtargetqの値はqと等しいためlossを計算する場合には影響しない．
        for i in range(self.batch_size):

            # 終端状態の場合はmaxQを0にしておくと学習が安定します（ヒント：maxq[i] * (not batch["dones"][i])）
            targetq[i, batch["actions"][i]] = batch["rewards"][i] + self.gamma * maxq[i] * (not batch["dones"][i]) 

        self.optimizer.zero_grad()

        # lossとしてMSEを利用．
        loss = nn.MSELoss()(q, torch.tensor(targetq))
        loss.backward()
        self.optimizer.step()

        # ターゲットネットワークのパラメータを更新．
        self.target_qnet = copy.deepcopy(self.qnet)
    
    # Q値が最大の行動を選択．
    def get_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.num_state)
        action = torch.argmax(self.qnet(state_tensor).data).item()
        return action
    
    # ε-greedyに行動を選択．
    def get_action(self, state, episode):
        epsilon = 0.7 * (1/(episode+1))  # ここでは0.5から減衰していくようなεを設定．
        if epsilon <= np.random.uniform(0,1):
            action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.num_action)
        return action