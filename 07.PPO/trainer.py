# coding:utf-8


""" 学習に必要な機能をまとめたクラス
    ここで，一定のエピソード間データ収集・学習・評価を繰り返すことで強化学習を行っていきます．
    また，学習後にシミュレーションや学習曲線を可視化することも行います．
"""

# 必要なライブラリのインポート．
from time import time
from datetime import timedelta
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
import os


""" Trainerクラス
    学習に必要な関数をまとめたクラスになります．
"""
class Trainer:

    # コンストラクタ．
    def __init__(self, env, env_test, algo, seed=0, num_episodes=10**6, 
                 eval_interval=10**4, num_eval_episodes=3, save_agent=False, agent_trigger=10):

        # 環境の設定．
        self.env = env
        self.env_test = env_test
        self.algo = algo

        # 環境の乱数シードの設定．
        self.env.seed(seed)
        self.env_test.seed(2**31-seed)

        # 平均収益を保存する辞書を作成．
        self.returns = {'episode': [], 'return': []}

        # データ収集を行う学習エピソード数．
        self.num_episodes = num_episodes

        # 評価までのエピソード数(インターバル)．
        self.eval_interval = eval_interval

        # 評価時のエピソード数．
        self.num_eval_episodes = num_eval_episodes

        # エージェントを保存するかしないかの設定．
        self.save_agent = save_agent
        
        # エージェントを保存する間隔の指定(エピソード数)．
        self.agent_trigger = agent_trigger

    # データ収集・学習・評価を繰り返す関数．
    def train(self):

        # 学習開始の時間を記録．
        self.start_time = time()

        # 指定したエピソードの間，データ収集・学習・評価を繰り返す．
        for episode in range(1, self.num_episodes+1):

            # 環境(self.env)，現在のエピソード数(episode)をアルゴリズムに渡し，
            # エピソードの終了までステップを繰り返す．
            self.algo.episode(self.env, episode)

            # 一定のインターバルでエージェントを評価．
            if episode % self.eval_interval == 0:
                self.evaluate(episode)

            # 学習したモデルを保存．
            if self.save_agent:
                os.makedirs('./agent', exist_ok=True)
                if (((episode) % self.agent_trigger) == 0):                    
                    save_agent_path = './agent/agent_episode_{}.pth'.format(episode)
                    torch.save(self.algo.actor.state_dict(), save_agent_path)

    # エージェントを評価する関数．
    def evaluate(self, episode):

        # 指定したエピソード分のシミュレータを動かし，その時の収益を記録．
        returns = []
        for _ in range(self.num_eval_episodes):
            episode_return = self.algo.episode(self.env, episode)
            returns.append(episode_return)

        # 指定エピソード分の平均収益を記録．
        mean_return = np.mean(returns)
        self.returns['episode'].append(episode)
        self.returns['return'].append(mean_return)

        # 指定エピソード分の評価結果を表示．
        # Episode     ： 現在の学習エピソード数．
        # Mean Reward ： 指定エピソード分の平均収益．
        # Time        ： 現在の学習時間．
        print('Episode {:>6} | Mean Reward {:>9.3f} | Time {}'.format(episode, mean_return, self.time))

    # シミュレータを動作させ，描画する関数．
    def visualize(self):

        # シミュレータ環境の構築．
        env = gym.make(self.env.unwrapped.spec.id)
        env.render()
        state = env.reset()

        # 1エピソードだけシミュレータを動作．
        done = False
        while not done:
            action = self.algo.exploit(state)
            state, _, done, _ = env.step(action)

        # 生成したインスタンスの削除．
        env.close()
        del env

    # 学習曲線(平均収益のグラフ)を描画する関数．
    def plot(self):

        # グラフの設定．
        plt.figure(figsize=(8,4))
        plt.plot(self.returns['episode'], self.returns['return'], color="b", ls="-", label='result', lw=1)

        # 軸ラベルの設定．
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Return', fontsize=12)
        plt.title(f'{self.env.unwrapped.spec.id}', fontsize=12)
        plt.tick_params(labelsize=10)

        # グラフの描画．
        plt.legend()
        plt.grid()
        plt.show()

    # 学習開始からの経過時間を計測する関数．
    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))