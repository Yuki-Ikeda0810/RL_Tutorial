# coding:utf-8


""" Trainer
    一定のステップ間データ収集・学習・評価を繰り返すTrainerクラスを利用します．
"""

# 必要なライブラリのインポート．
from time import time
from datetime import timedelta
import numpy as np
import gym
import matplotlib.pyplot as plt


""" Trainerクラス
"""
class Trainer:
    def __init__(self, env, env_test, algo, seed=0, num_steps=10**6, eval_interval=10**4, num_eval_episodes=3):

        self.env = env
        self.env_test = env_test
        self.algo = algo

        # 環境の乱数シードを設定する．
        self.env.seed(seed)
        self.env_test.seed(2**31-seed)

        # 平均収益を保存するための辞書．
        self.returns = {'step': [], 'return': []}

        # データ収集を行うステップ数．
        self.num_steps = num_steps

        # 評価の間のステップ数(インターバル)．
        self.eval_interval = eval_interval

        # 評価を行うエピソード数．
        self.num_eval_episodes = num_eval_episodes

    # num_stepsステップの間，データ収集・学習・評価を繰り返す．
    def train(self):

        # 学習開始の時間
        self.start_time = time()

        # エピソードのステップ数．
        t = 0

        # 環境を初期化する．
        state = self.env.reset()

        for steps in range(1, self.num_steps + 1):

            # 環境(self.env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            # アルゴリズムに渡し，状態・エピソードのステップ数を更新する．
            state, t = self.algo.step(self.env, state, t, steps)

            # アルゴリズムが準備できていれば，1回学習を行う．
            if self.algo.is_update(steps):
                self.algo.update()

            # 一定のインターバルで評価する．
            if steps % self.eval_interval == 0:
                self.evaluate(steps)

    # 複数エピソード環境を動かし，平均収益を記録する．
    def evaluate(self, steps):
        returns = []
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            done = False
            episode_return = 0.0

            while (not done):
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward

            returns.append(episode_return)

        mean_return = np.mean(returns)
        self.returns['step'].append(steps)
        self.returns['return'].append(mean_return)

        print(f'Num steps: {steps:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    # 1エピソード環境を動かし，mp4を再生する．
    def visualize(self):
        env = wrap_monitor(gym.make(self.env.unwrapped.spec.id))
        state = env.reset()
        done = False

        while (not done):
            action = self.algo.exploit(state)
            state, _, done, _ = env.step(action)

        del env
        return play_mp4()

    # 平均収益のグラフを描画する．
    def plot(self):
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.returns['step'], self.returns['return'])
        plt.xlabel('Steps', fontsize=24)
        plt.ylabel('Return', fontsize=24)
        plt.tick_params(labelsize=18)
        plt.title(f'{self.env.unwrapped.spec.id}', fontsize=24)
        plt.tight_layout()

    # 学習開始からの経過時間．
    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))