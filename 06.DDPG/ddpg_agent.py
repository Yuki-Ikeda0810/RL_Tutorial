# coding:utf-8


""" 方策に基づく手法 (Policy-based Methods)
    今回は，方策に基づく代表的な手法として，連続値の行動空間を持つ問題に対応した
    Deep Deterministic Policy Gradient(DDPG)を扱います．
"""

# 必要なライブラリのインポート．
import numpy as np
import copy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

""" Deep Deterministic Policy Gradient(DDPG)
    今回は，方策(Actor)と価値関数(Critic)の両方をニューラルネットワークで近似します．
    全体として，連続値の行動空間に対してDQNを用いた手法になっています．
"""

# Actorのネットワーク．
class ActorNetwork(nn.Module):
    def __init__(self, num_state, action_space, hidden_size=16):
        super(ActorNetwork, self).__init__()
        self.action_mean = torch.tensor(0.5*(action_space.high+action_space.low), dtype=torch.float)
        self.action_halfwidth = torch.tensor(0.5*(action_space.high-action_space.low), dtype=torch.float)
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space.shape[0])

    def forward(self, s):
        h = F.elu(self.fc1(s))
        h = F.elu(self.fc2(h))
        a = self.action_mean + self.action_halfwidth*torch.tanh(self.fc3(h))
        return a

# Criticのネットワーク(状態と行動を入力にしてQ値を出力)．
class CriticNetwork(nn.Module):
    def __init__(self, num_state, action_space, hidden_size=16):
        super(CriticNetwork, self).__init__()
        self.action_mean = torch.tensor(0.5*(action_space.high+action_space.low), dtype=torch.float)
        self.action_halfwidth = torch.tensor(0.5*(action_space.high-action_space.low), dtype=torch.float)
        self.fc1 = nn.Linear(num_state+action_space.shape[0], hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space.shape[0])

    def forward(self, s, a):
        a = (a-self.action_mean)/self.action_halfwidth
        h = F.elu(self.fc1(torch.cat([s,a],1)))
        h = F.elu(self.fc2(h))
        q = self.fc3(h)
        return q

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

# DDPGエージェントのクラス．
class DdpgAgent:
    def __init__(self, observation_space, action_space, gamma=0.99, lr=1e-3, batch_size=32, memory_size=50000):
        self.num_state = observation_space.shape[0]
        self.num_action = action_space.shape[0]
        self.state_mean = 0.5*(observation_space.high + observation_space.low)
        self.state_halfwidth = 0.5*(observation_space.high - observation_space.low)
        self.gamma = gamma  # 割引率．
        self.batch_size = batch_size
        self.actor = ActorNetwork(self.num_state, action_space)
        self.actor_target = copy.deepcopy(self.actor)    # Actorのターゲットネットワーク．
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = CriticNetwork(self.num_state, action_space)
        self.critic_target = copy.deepcopy(self.critic)  # Criticのターゲットネットワーク．
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(memory_size)
    
    # 連続値の状態を[-1,1]の範囲に正規化．
    def normalize_state(self, state):
        state = (state-self.state_mean)/self.state_halfwidth
        return state
    
    # リプレイバッファからサンプルされたミニバッチをtensorに変換．
    def batch_to_tensor(self, batch):
        states = torch.tensor([self.normalize_state(s) for s in batch["states"]], dtype=torch.float)
        actions = torch.tensor(batch["actions"], dtype=torch.float)
        next_states = torch.tensor([self.normalize_state(s) for s in batch["next_states"]], dtype=torch.float)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float)
        return states, actions, next_states, rewards
    
    # actorとcriticを更新．
    def update(self):
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, rewards = self.batch_to_tensor(batch)
        # criticの更新．
        target_q = (rewards + self.gamma*self.critic_target(next_states, self.actor_target(next_states)).squeeze()).data
        q = self.critic(states, actions).squeeze()
        critic_loss = F.mse_loss(q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # actorの更新．
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ターゲットネットワークのパラメータを更新．
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
    
    # Q値が最大の行動を選択．
    def get_action(self, state):
        state_tensor = torch.tensor(self.normalize_state(state), dtype=torch.float).view(-1, self.num_state)
        action = self.actor(state_tensor).view(self.num_action)
        return action