# coding:utf-8


""" 最終的に得られた方策のテスト
    学習したエージェントの方策をシミュレーションし，描画します．
"""

# 必要なライブラリのインポート．
import gym
import pybullet_envs
import torch

import sac_agent                       #「sac_agent.py」をインポート．
from calculation import reparameterize #「calculation.py」の「reparameterize」関数をインポート．


def main():

    """ 各種設定
        シミュレーションに必要なパラメータの設定をします．
        今回は，OpenAI Gymと呼ばれる，強化学習のシミュレータのライブラリを用いて環境を作成します．
        これにPyBulletと呼ばれる，ライブラリを読み込ませることで，物理エンジンを用いた環境を作成することができます．
    """
    # Inverted Pendulumと呼ばれる台車に振子がついた環境を利用する場合の設定．
    # 参考：https://github.com/bulletphysics/bullet3
    ENV_ID        =  'InvertedPendulumBulletEnv-v0'
    SEED          =  0                       # 乱数を生成する設定値(再現性のある乱数を獲得できる)．
    REWARD_SCALE  =  1.0                     # 報酬の大きさ．
    AGENT_NAME    = 'agent_episode_600.pth'  # 使用するエージェントの設定．

    # Half Cheetahと呼ばれる4足歩行のチータが半身(2足歩行)になった環境を利用する場合の設定．
    # 参考：https://github.com/bulletphysics/bullet3
    # ENV_ID       =  'HalfCheetahBulletEnv-v0'
    # SEED         =  0                       # 乱数を生成する設定値(再現性のある乱数を獲得できる)．
    # REWARD_SCALE =  5.0                     # 報酬の大きさ．
    # AGENT_NAME   = 'agent_episode_1000.pth' # 使用するエージェントの設定．

    # エピソード回数の指定．
    episode            = 4

    # 保存する動画の設定．
    save_video         =  True   # 動画を保存するかしないかの設定．
    video_trigger      =  2      # 動画を保存する間隔の指定(エピソード数)．


    """ 各種定義
        設定したパラメータを引数にして機能を定義します．
    """
    # Gymの警告を一部無視．
    gym.logger.set_level(40)

    # シミュレータ環境の構築．
    env = gym.make(ENV_ID)

    # シミュレーションを動画で保存する設定．
    if save_video:
        env = gym.wrappers.RecordVideo(env,'./movie/test', episode_trigger=(lambda ep: ep % video_trigger == 0))

    # SACのインスタンスを生成．
    algo = sac_agent.SAC(state_shape=env.observation_space.shape,
                        action_shape=env.action_space.shape,
                        seed=SEED,
                        reward_scale=REWARD_SCALE
    )

    # 学習したエージェントの読み込み．
    algo.actor.load_state_dict(torch.load('./agent/' + AGENT_NAME))


    """ 学習したSACエージェントのシミュレーション
        指定したエピソード数分，シミュレータを動作させます．
    """
    for i_episode in range(1, episode+1):

        # シミュレータ画面の出力．
        # PyBulletでは「render」を呼ぶタイミングがGymと異なる．
        env.render()

        # エピソードを開始(環境の初期化)．
        state = env.reset()

        # エピソードの終了条件まで繰り返す．
        done = False
        while not done:

            #  現在の状態に対して最適な行動を選択．
            action = algo.exploit(state)

            # 行動を実行し，次の状態，報酬，終端か否かの情報を取得．
            state, reward, done, info = env.step(action)

        print("Episode %d finished" % (i_episode))

    # 画面出力の終了．
    env.close()

    # 生成したインスタンスの削除．
    del env
    del algo


if __name__ == "__main__":
    main()