# coding:utf-8


""" OpenAI Gymと呼ばれる，強化学習のシミュレータのライブラリを用いて環境を作成します．
    これにPyBulletと呼ばれる，ライブラリを読み込ませることで，物理エンジンを用いた環境を作成することができます．
    今回は，HalfCheetahBulletEnv-v0と呼ばれる，4足歩行のチータが半身(2足歩行)になった環境を利用します．
    参考：https://github.com/bulletphysics/bullet3
"""

# 必要なライブラリのインポート．
import gym
import pybullet_envs

# エピソード回数の指定．
episode = 4

# 動画を保存する間隔の指定(エピソード数)．
video_trigger = 2

# Gymの警告を一部無視．
gym.logger.set_level(40)

# シミュレータ環境の構築．
env = gym.make('HalfCheetahBulletEnv-v0')

# シミュレーションを動画として保存する設定．
env = gym.wrappers.RecordVideo(env,'./movie', episode_trigger=(lambda ep: ep % video_trigger == 0))


""" シミュレーション チュートリアル
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

        # ランダムな行動を選択．
        # -1.0 〜 1.0 : 各ジョイントへの力 : 6次元の連続値．
        action = env.action_space.sample()

        # 行動を実行し，次の状態，報酬，終端か否かの情報を取得．
        # next_state : 26次元の連続値．
        # reward     : 前の行動によって達成された報酬の量．
        # done       : 環境をリセットするべきかの判断(True : ゲームオーバー、False : コンティニュー)．
        # info       : デバッグに役立つ診断情報．
        next_state, reward, done, info = env.step(action)

    print("Episode %d finished" % (i_episode))

# 画面出力の終了．
env.close()

# 生成したインスタンスの削除
del env
