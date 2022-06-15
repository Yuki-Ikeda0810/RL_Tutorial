# coding:utf-8


""" Algorithmクラス
    様々な強化学習手法に共通する機能をまとめた抽象クラスです．
    このAlgorithmクラスを継承することで，強化学習手法を記述したアルゴリズム(Trainerのself.algoの部分)を実装します．
"""

# 必要なライブラリのインポート．
from abc import ABC, abstractmethod
import torch


""" Algorithmクラス
"""
class Algorithm(ABC):

    # 確率論的な行動と，その行動の確率密度の対数を返す関数．
    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()

    # 決定論的な行動を返す関数．
    def exploit(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    # 学習できるかどうかを判断する関数．
    @abstractmethod
    def is_update(self, episode):
        pass

    # エピソードの終了までステップを繰り返す関数．
    @abstractmethod
    def episode(self, env, episode):
        pass

    # 1回分の学習を行う関数．
    @abstractmethod
    def update(self):
        pass