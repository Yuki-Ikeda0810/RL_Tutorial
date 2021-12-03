# coding:utf-8


""" 価値に基づく手法 (Value-based Methods)
    今回は，価値に基づく代表的な手法として，SARSAを扱います．
"""

# 必要なライブラリのインポート．
import numpy as np


""" SARSA
    今回は，方策としてε-greedy方策を用います．
    この実装ではエピソードが途中で終了した場合はペナルティを本来の報酬から引いています．
    テーブルを用いてQ関数を表現した場合，学習効率が悪く不安定になりがちなためです．
    このような，学習を容易にするための追加的な報酬の設計をreward shapingといいます．
"""

# SARSAエージェントのクラス．
class SarsaAgent:
    def __init__(self, num_state, num_action, num_discretize, gamma=0.99, alpha=0.5, max_initial_q=0.1):

        self.num_action = num_action
        self.gamma = gamma  # 割引率．
        self.alpha = alpha  # 学習率．

        # Qテーブルを作成し乱数で初期化．
        self.qtable = np.random.uniform(low=-max_initial_q, high=max_initial_q, size=(num_discretize**num_state, num_action)) 
    
    # Qテーブルを更新．
    def update_qtable(self, state, action, reward, next_state, next_action):
        self.qtable[state, action] += self.alpha*(reward+self.gamma*self.qtable[next_state, next_action]-self.qtable[state, action])
    
    # Q値が最大の行動を選択．
    def get_greedy_action(self, state):
        action = np.argmax(self.qtable[state])
        return action
    
    # ε-greedyに行動を選択．
    def get_action(self, state, episode):
        epsilon = 0.7 * (1/(episode+1))  # ここでは0.5から減衰していくようなεを設定．
        if epsilon <= np.random.uniform(0,1):
            action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.num_action)
        return action