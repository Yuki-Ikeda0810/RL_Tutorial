# coding:utf-8


""" Rollout Bufferクラス
    状態・行動・即時報酬・終了シグナル・確率密度の対数・次の状態をロールアウト1回分だけ保存します．
"""

# 必要なライブラリのインポート．
import torch
import numpy as np


""" Rollout Bufferクラス
"""
class RolloutBuffer:

    # コンストラクタ．
    def __init__(self, buffer_size, state_shape, action_shape, device):

        # 保存するデータ．
        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)

        # 次にデータを挿入するインデックス．
        self._p = 0

        # バッファのサイズ．
        self.buffer_size = buffer_size

    # Rollout Bufferに状態・行動・即時報酬・終了シグナル・確率密度の対数・次の状態を追加する関数．
    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self._p = (self._p + 1) % self.buffer_size
    
    # Rollout Bufferから状態・行動・即時報酬・終了シグナル・確率密度の対数・次の状態を返す関数．
    def get(self):
        assert self._p == 0, 'Buffer needs to be full before training.'
        return self.states, self.actions, self.rewards, self.dones, self.log_pis, self.next_states
