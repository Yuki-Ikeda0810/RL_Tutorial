# coding:utf-8


""" Algorithmクラス
    抽象クラスを継承して，強化学習手法を記述したアルゴリズム(Trainerのself.algoの部分)を実装します．
"""

# 必要なライブラリのインポート．
from abc import ABC, abstractmethod
import torch


""" Algorithmクラス
"""
class Algorithm(ABC):

    # 確率論的な行動と，その行動の確率密度の対数 \log(\pi(a|s)) を返す．
    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()

    # 決定論的な行動を返す．
    def exploit(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    # 現在のトータルのステップ数(steps)を受け取り，アルゴリズムを学習するか否かを返す．
    @abstractmethod
    def is_update(self, steps):
        pass

    # 環境(env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を受け取り，
    # リプレイバッファへの保存などの処理を行い，状態・エピソードのステップ数を更新する．
    @abstractmethod
    def step(self, env, state, t, steps):
        pass

    # 1回分の学習を行う．
    @abstractmethod
    def update(self):
        pass