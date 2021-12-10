# coding:utf-8


""" OpenAI Gymと呼ばれる，強化学習のシミュレータのライブラリを用いて環境を作成します．
    今回は，CartPole-v0と呼ばれる，台車に振子がついた環境を利用します．
    参考：https://github.com/openai/gym/wiki/CartPole-v0
"""

# 必要なライブラリのインポート．
import gym
from gym import wrappers

# シミュレータ環境の構築．
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, "./movie", force=True)

# 5回のエピソードをシミュレーターで動作させる．
for i_episode in range(5):

    # エピソードを開始(環境の初期化)．
    state = env.reset()

    # シミュレータ画面の出力．
    env.render()

    done = False
    while not done:

        # ランダムな行動を選択．
        # 0 : カートを左へ．
        # 1 : カートを右へ．
        action = env.action_space.sample()

        # 行動を実行し，次の状態，報酬，終端か否かの情報を取得．
        # next_state : (カートの位置(-4.8 4.8)，カートの速度(-Inf Inf)，ポールの角度(-24 deg 24 deg)，ポールの角速度(-Inf Inf))．
        # reward : 前の行動によって達成された報酬の量．
        # done : 環境をリセットするべきかの判断(True : ゲームオーバー、False : コンティニュー)．
        # info : デバッグに役立つ診断情報．
        next_state, reward, done, info = env.step(action)

# 画面出力の終了．
env.close()