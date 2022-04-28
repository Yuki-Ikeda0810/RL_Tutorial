# coding:utf-8


""" 価値に基づく手法 (Value-based Methods)
    今回は，価値に基づく代表的な手法として，Q-Learningを扱います．
"""

# 必要なライブラリのインポート．
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt

# 「q_learning_agent.py」をインポート．
import q_learning_agent


""" 各種設定
    学習に必要なパラメータの設定をします．
"""

# 学習の設定．
num_episode    = 1200 # 学習エピソード数．
penalty        = 10   # 途中でエピソードが終了したときのペナルティ．
num_discretize = 6    # 状態空間の分割数．

# ログ用の設定．
episode_rewards = []
num_average_epidodes = 10

env = gym.make('CartPole-v0')          # シミュレータ環境の構築．
max_steps = env.spec.max_episode_steps # エピソードの最大ステップ数．

# SARSAエージェントのインスタンスを作成．
agent = q_learning_agent.QLearningAgent(env.observation_space.shape[0], env.action_space.n, num_discretize)


""" テーブル解法
    cartpoleは連続値をとる状態空間の問題設定です．
    テーブル解法では，本来は連続値として表現される状態空間を適当な間隔で区切って離散化します．
    ここでは，状態空間の各次元を6分割して，64×2=2692 個の値を持つ表としてQ関数を表現します．
"""

# 状態を離散化して対応するインデックスを返す関数．
# binの上限・下限はcartpole環境固有のものを用いています．
def discretize_state(observation, num_discretize):
    c_pos, c_v, p_angle, p_v = observation

    # np.digitize : 数値が指定した配列(これをbinという)の，どの位置にあるかを返します．
    # np.linspace : 指定された等差数列を生成します．
    discretized = [
        np.digitize(c_pos, bins=np.linspace(-2.4, 2.4, num_discretize+1)[1:-1]),
        np.digitize(c_v, bins=np.linspace(-3.0, 3.0, num_discretize+1)[1:-1]),
        np.digitize(p_angle, bins=np.linspace(-0.5, 0.5, num_discretize+1)[1:-1]),
        np.digitize(p_v, bins=np.linspace(-2.0, 2.0, num_discretize+1)[1:-1])
    ]
    return sum([x*(num_discretize**i) for i, x in enumerate(discretized)])


""" Q-Learningエージェントの学習
    今回は，エージェントの学習に，OpenAI GymのCartPole-v0と呼ばれる，台車に振子がついた環境を利用します．
    参考：https://github.com/openai/gym/wiki/CartPole-v0
"""

for episode in range(num_episode):
    observation = env.reset()                             # エピソードを開始(環境の初期化)．
    state = discretize_state(observation, num_discretize) # 観測の離散化(状態のインデックスを取得)．
    
    episode_reward = 0
    for t in range(max_steps):
        action = agent.get_action(state, episode)         #  行動を選択．
        observation, reward, done, _ = env.step(action)   # 行動を実行し，次の状態，報酬，終端か否かの情報を取得．
        
        # もしエピソードの途中で終了してしまったらペナルティを加える．
        if done and t < max_steps - 1:
            reward = - penalty
        episode_reward += reward

        next_state = discretize_state(observation, num_discretize) # 観測の離散化(状態のインデックスを取得)．
        agent.update_qtable(state, action, reward, next_state)     # Q値の表を更新
        state = next_state                                         # 状態を更新

        if done:
            break

    episode_rewards.append(episode_reward)
    if episode % 50 == 0:
        print("Episode %4d finished | Episode reward %f" % (episode, episode_reward))

# 累積報酬の移動平均を表示．
moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
plt.plot(np.arange(len(moving_average)),moving_average)
plt.title('Q-Learning: average rewards in %d episodes' % num_average_epidodes)
plt.xlabel('episode')
plt.ylabel('rewards')
plt.show()

# 画面出力の終了．
env.close()


""" 最終的に得られた方策のテスト(可視化)
    5回のエピソードをシミュレーターで動作させ，学習後のエージェントの方策を可視化します．
"""

# シミュレータ環境の構築．
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, "./movie", force=True)

# 5回のエピソードをシミュレーターで動作させる．
for i_episode in range(5):
    observation = env.reset()                             # エピソードを開始(環境の初期化)．
    state = discretize_state(observation, num_discretize) # 観測の離散化(状態のインデックスを取得)．
    env.render()                                          # シミュレータ画面の出力．

    done = False
    while not done:
        action = agent.get_greedy_action(state)                    # エージェントによる行動を取得．
        next_observation, reward, done, _ = env.step(action)       # 行動を実行し，次の状態，報酬，終端か否かの情報を取得．
        state = discretize_state(next_observation, num_discretize) # 観測の離散化(状態のインデックスを取得)．
        env.render()                                               # シミュレータ画面の出力．

# 画面出力の終了．
env.close()