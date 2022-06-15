# coding:utf-8


""" Replay Bufferクラス
    状態・行動・即時報酬・終了シグナル・次の状態を保存します．
"""

# 必要なライブラリのインポート．
import torch
import numpy as np


""" Replay Bufferクラス
"""
class ReplayBuffer:
    
    # コンストラクタ．
    def __init__(self, buffer_size, state_shape, action_shape, device):
        
        # 次にデータを挿入するインデックス．
        self._p = 0
        
        # データ数．
        self._n = 0
        
        # リプレイバッファのサイズ．
        self.buffer_size = buffer_size

        # 保存するデータ．
        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
    
    # Replay Bufferに状態・行動・即時報酬・終了シグナル・次の状態を追加する関数．
    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)
    
    # 指定されたバッチサイズのミニバッチとして，
    # Replay Bufferから状態・行動・即時報酬・終了シグナル・次の状態を返す関数．
    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )
