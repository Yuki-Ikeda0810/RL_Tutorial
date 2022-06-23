# coding:utf-8


""" 方策に基づく手法 (Policy-based Methods)
    今回は，方策に基づく手法の1つとして，方策と価値関数の近似した
    Proximal Policy Optimization(PPO)を扱います．
"""

# 必要なライブラリのインポート．
import gym
import pybullet_envs

import ppo_agent             #「ppo_agent.py」をインポート．
from trainer import Trainer  #「training.py」の「Trainer」クラスをインポート．


def main():

    """ 各種設定
        学習に必要なパラメータの設定をします．
        今回は，OpenAI Gymと呼ばれる，強化学習のシミュレータのライブラリを用いて環境を作成します．
        これにPyBulletと呼ばれる，ライブラリを読み込ませることで，物理エンジンを用いた環境を作成することができます．
    """
    # Inverted Pendulumと呼ばれる台車に振子がついた環境を利用する場合の設定．
    # 参考：https://github.com/bulletphysics/bullet3
    ENV_ID             =  'InvertedPendulumBulletEnv-v0'
    SEED               =  0       # 乱数を生成する設定値(再現性のある乱数を獲得できる)．
    NUM_EPISODES       =  1000     # 学習エピソード数())．
    EVAL_INTERVAL      =  10      # 評価までのエピソード数()．
    NUM_EVAL_EPISODES  =  3       # 評価時のエピソード数．

    # Half Cheetahと呼ばれる4足歩行のチータが半身(2足歩行)になった環境を利用する場合の設定．
    # 参考：https://github.com/bulletphysics/bullet3
    # ENV_ID             =  'HalfCheetahBulletEnv-v0'
    # SEED               =  0       # 乱数を生成する設定値(再現性のある乱数を獲得できる)．
    # NUM_EPISODES       =  1000    # 学習エピソード数()．
    # EVAL_INTERVAL      =  20      # 評価までのエピソード数()．
    # NUM_EVAL_EPISODES  =  3       # 評価時のエピソード数．

    # 保存する動画の設定．
    save_video         =  True    # 動画を保存するかしないかの設定．
    video_trigger      =  100     # 動画を保存する間隔の指定(エピソード数)．

    # 保存するエージェントの設定．
    save_agent         =  True    # エージェントを保存するかしないかの設定．
    agent_trigger      =  100     # エージェントを保存する間隔の指定(エピソード数)．


    """ 各種定義
        設定したパラメータを引数にして機能を定義します．
    """
    # Gymの警告を一部無視．
    gym.logger.set_level(40)

    # シミュレータ環境の構築．
    env = gym.make(ENV_ID)
    env_test = gym.make(ENV_ID)

    # シミュレーションを動画で保存する設定．
    if save_video:
        env = gym.wrappers.RecordVideo(env,'./movie/training', episode_trigger=(lambda ep: ep % video_trigger == 0))

    # PPOのインスタンスを生成．
    algo = ppo_agent.PPO(state_shape=env.observation_space.shape,
                         action_shape=env.action_space.shape,
                         seed=SEED,
    )

    # Trainerのインスタンスを生成．
    trainer = Trainer(env=env,
                      env_test=env_test,
                      algo=algo,
                      seed=SEED,
                      num_episodes=NUM_EPISODES,
                      eval_interval=EVAL_INTERVAL,
                      num_eval_episodes=NUM_EVAL_EPISODES, 
                      save_agent=save_agent,
                      agent_trigger=agent_trigger
    )


    """ PPOエージェントの学習
        上記で設定したパラメータを用いて，PPOエージェントの学習を行います．
    """
    # PPOエージェントの学習．
    trainer.train()

    # PPOエージェントの学習結果を表示．
    # trainer.visualize() # 1エピソード分のシミュレーション．
    trainer.plot()        # 学習曲線の表示．

    # 生成したインスタンスの削除．
    del env
    del env_test
    del algo
    del trainer


if __name__ == "__main__":
    main()