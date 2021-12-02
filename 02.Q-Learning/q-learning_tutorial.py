# coding:utf-8


""" 価値に基づく手法 (Value-based Methods)
    今回は，価値に基づく代表的な手法として，Q-Learningを扱います．
"""

# 必要なライブラリのインポート．
import numpy as np
# import copy
# from collections import deque
import gym
from gym import wrappers
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.distributions import Categorical, Normal
# import matplotlib
# import matplotlib.animation as animation
import matplotlib.pyplot as plt

# from IPython import display
# from JSAnimation.IPython_display import display_animation
# from IPython.display import HTML


""" テーブル解法
    cartpoleは連続値をとる状態空間の問題設定です．
    テーブル解法では，本来は連続値として表現される状態空間を適当な間隔で区切って離散化します．
    ここでは，状態空間の各次元を6分割して，64×2=2692 個の値を持つ表としてQ関数を表現します．
"""

# 等差数列を生成する．
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num+1)[1:-1]

# 状態を離散化して対応するインデックスを返す関数(binの上限・下限はcartpole環境固有のものを用いています)
def discretize_state(observation, num_discretize):
    c_pos, c_v, p_angle, p_v = observation
    discretized = [
        np.digitize(c_pos, bins=bins(-2.4, 2.4, num_discretize)), 
        np.digitize(c_v, bins=bins(-3.0, 3.0, num_discretize)),
        np.digitize(p_angle, bins=bins(-0.5, 0.5, num_discretize)),
        np.digitize(p_v, bins=bins(-2.0, 2.0, num_discretize))
    ]
    return sum([x*(num_discretize**i) for i, x in enumerate(discretized)])


""" Q-Learning
    今回は，方策としてε-greedy方策を用います．
    この実装ではエピソードが途中で終了した場合はペナルティを本来の報酬から引いています．
    テーブルを用いてQ関数を表現した場合，学習効率が悪く不安定になりがちなためです．
    このような，学習を容易にするための追加的な報酬の設計をreward shapingといいます．
"""

class QLearningAgent: