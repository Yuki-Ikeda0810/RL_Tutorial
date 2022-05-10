# coding:utf-8


""" 方策に基づく手法 (Policy-based Methods)
    今回は，方策に基づく代表的な手法として，方策と価値関数の近似した
    Soft Actor-Critic(SAC)を扱います．
"""

# 必要なライブラリのインポート．
import gym
import pybullet_envs

import sac_agent             #「sac_agent.py」をインポート．
from trainer import Trainer  #「training.py」の「Trainer」クラスをインポート．


""" 各種設定
    学習に必要なパラメータの設定をします．
"""

# InvertedPendulumのシミュレータ環境を用いる場合．
# ENV_ID        = 'InvertedPendulumBulletEnv-v0'
# SEED          = 0
# REWARD_SCALE  = 1.0
# NUM_STEPS     = 50000 # 学習ステップ数(5 * 10 ** 4)
# EVAL_INTERVAL = 1000 # 学習の更新ステップ数(10 ** 3)

# HalfCheetahのシミュレータ環境を用いる場合．
ENV_ID        = 'HalfCheetahBulletEnv-v0'
SEED          = 0
REWARD_SCALE  = 5.0
NUM_STEPS     = 1000000 # 学習ステップ数(10 ** 6)
EVAL_INTERVAL = 10000   # 学習の更新ステップ数(10 ** 4)

video_trigger   = 1 # 動画を保存する間隔の指定(エピソード数)．


""" 各種定義
    設定したパラメータを引数にして機能を定義します．
"""

# Gymの警告を一部無視する．
gym.logger.set_level(40)

# シミュレータ環境の構築．
env = gym.make(ENV_ID)
env_test = gym.make(ENV_ID)

# シミュレーションを動画で保存する．
env = gym.wrappers.RecordVideo(env,'./movie', episode_trigger=(lambda ep: ep % video_trigger == 0))
# env = gym.wrappers.Monitor(env, './movie', video_callable=(lambda ep: ep % video_trigger == 0))

# SACのインスタンスを生成．
algo = sac_agent.SAC(state_shape=env.observation_space.shape,
                     action_shape=env.action_space.shape,
                     seed=SEED,
                     reward_scale=REWARD_SCALE,
)

# Trainerのインスタンスを生成．
trainer = Trainer(env=env,
                  env_test=env_test,
                  algo=algo,
                  seed=SEED,
                  num_steps=NUM_STEPS,
                  eval_interval=EVAL_INTERVAL,
)


""" Actor-Criticエージェントの学習
    今回は，エージェントの学習に，OpenAI GymのCartPole-v0と呼ばれる，台車に振子がついた環境を利用します．
    参考：https://github.com/openai/gym/wiki/CartPole-v0
"""

trainer.train()

del env
del env_test
del algo
del trainer

# """ 最終的に得られた方策のテスト(可視化)
#     5回のエピソードをシミュレーターで動作させ，学習後のエージェントの方策を可視化します．
# """

# trainer.plot()
# trainer.visualize()