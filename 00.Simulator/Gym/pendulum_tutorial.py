# coding:utf-8


""" OpenAI Gymと呼ばれる，強化学習のシミュレータ(ゲーム)のライブラリを用いて環境を作成します．
    今回は，Pendulum-v1と呼ばれる，台車に振子がついた環境を利用します．
    参考：https://github.com/openai/gym/wiki/Pendulum-v0
"""

# 必要なライブラリのインポート．
import gym

# シミュレーションの設定．
episode = 3        # エピソード回数の指定．
video_trigger = 1  # 動画を保存する間隔の指定(エピソード数)．

# Gymの警告を一部無視する．
gym.logger.set_level(40)

# シミュレータ環境の構築．
env = gym.make('Pendulum-v1')

# シミュレーションを動画で保存する．
env = gym.wrappers.RecordVideo(env,'./movie', episode_trigger=(lambda ep: ep % video_trigger == 0))

# 指定したエピソード数分，シミュレータを動作させる．
for i_episode in range(episode):

    # エピソードを開始(環境の初期化)．
    state = env.reset()

    done = False
    while not done:

        # シミュレータ画面の出力．
        env.render()

        # ランダムな行動を選択．
        # -2.0 〜 2.0 : ジョイントへの力．
        action = env.action_space.sample()

        # 行動を実行し，次の状態，報酬，終端か否かの情報を取得．
        # next_state : (ジョイントのx座標(cos)(-1.0 1.0)，ジョイントのy座標(sin)(-1.0 1.0)，ジョイントの角速度(-8.0 8.0))．
        # reward : 前の行動によって達成された報酬の量．
        # done : 環境をリセットするべきかの判断(True : ゲームオーバー、False : コンティニュー)．
        # info : デバッグに役立つ診断情報．
        next_state, reward, done, info = env.step(action)

    print("Episode %d finished" % (i_episode+1))

# 画面出力の終了．
env.close()