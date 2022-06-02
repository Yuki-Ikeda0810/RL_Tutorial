# coding:utf-8


""" OpenAI Gymと呼ばれる，強化学習のシミュレータ(ゲーム)のライブラリを用いて環境を作成します．
    今回は，CartPole-v0と呼ばれる，台車に振子がついた環境を利用します．
    参考：https://github.com/openai/gym/wiki/CartPole-v0
"""

# 必要なライブラリのインポート．
import gym

# エピソード回数の指定．
episode = 3 

# Gymの警告を一部無視．
gym.logger.set_level(40)

# シミュレータ環境の構築．
env = gym.make('CartPole-v0')

# シミュレーションを動画として保存する設定．
env = gym.wrappers.RecordVideo(env,'./movie')


""" シミュレーション チュートリアル
    指定したエピソード数分，シミュレータを動作させます．
"""

for i_episode in range(1, episode+1):
    
    # エピソードを開始(環境の初期化)．
    state = env.reset()

    done = False
    while not done:

        # シミュレータ画面の出力．
        env.render()

        # ランダムな行動を選択．
        # 0, 1 : カートを左へ, カートを右へ : 1次元の離散値．
        action = env.action_space.sample()

        # 行動を実行し，次の状態，報酬，終端か否かの情報を取得．
        # next_state : カートの位置(-4.8 4.8)，カートの速度(-Inf Inf)，ポールの角度(-24 deg 24 deg)，ポールの角速度(-Inf Inf) : 4次元の連続値．
        # reward     : 前の行動によって達成された報酬の量．
        # done       : 環境をリセットするべきかの判断(True : ゲームオーバー、False : コンティニュー)．
        # info       : デバッグに役立つ診断情報．
        next_state, reward, done, info = env.step(action)

    print("Episode %d finished" % (i_episode))

# 画面出力の終了．
env.close()