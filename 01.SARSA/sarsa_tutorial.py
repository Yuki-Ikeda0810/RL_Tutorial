# coding:utf-8


""" 価値に基づく手法 (Value-based Methods)
    今回は，価値に基づく代表的な手法として，SARSAを扱います．
"""

# 必要なライブラリのインポート．
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt


""" テーブル解法
    cartpoleは連続値をとる状態空間の問題設定です．
    テーブル解法では，本来は連続値として表現される状態空間を適当な間隔で区切って離散化します．
    ここでは，状態空間の各次元を6分割して，64×2=2692 個の値を持つ表としてQ関数を表現します．
"""

# 等差数列を生成する．
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num+1)[1:-1]

# 状態を離散化して対応するインデックスを返す関数．
# binの上限・下限はcartpole環境固有のものを用いています．
def discretize_state(observation, num_discretize):
    c_pos, c_v, p_angle, p_v = observation
    discretized = [
        np.digitize(c_pos, bins=bins(-2.4, 2.4, num_discretize)), 
        np.digitize(c_v, bins=bins(-3.0, 3.0, num_discretize)),
        np.digitize(p_angle, bins=bins(-0.5, 0.5, num_discretize)),
        np.digitize(p_v, bins=bins(-2.0, 2.0, num_discretize))
    ]
    return sum([x*(num_discretize**i) for i, x in enumerate(discretized)])


""" SARSA
    今回は，方策としてε-greedy方策を用います．
    この実装ではエピソードが途中で終了した場合はペナルティを本来の報酬から引いています．
    テーブルを用いてQ関数を表現した場合，学習効率が悪く不安定になりがちなためです．
    このような，学習を容易にするための追加的な報酬の設計をreward shapingといいます．
"""

# SARSAエージェントのクラス．
class SarsaAgent:
    def __init__(self, num_state, num_action, num_discretize, gamma=0.99, alpha=0.5, max_initial_q=0.1):

        self.num_action = num_action
        self.gamma = gamma  # 割引率．
        self.alpha = alpha  # 学習率．

        # Qテーブルを作成し乱数で初期化．
        self.qtable = np.random.uniform(low=-max_initial_q, high=max_initial_q, size=(num_discretize**num_state, num_action)) 
    
    # Qテーブルを更新．
    def update_qtable(self, state, action, reward, next_state, next_action):
        self.qtable[state, action] += self.alpha*(reward+self.gamma*self.qtable[next_state, next_action]-self.qtable[state, action])
    
    # Q値が最大の行動を選択．
    def get_greedy_action(self, state):
        action = np.argmax(self.qtable[state])
        return action
    
    # ε-greedyに行動を選択．
    def get_action(self, state, episode):
        epsilon = 0.7 * (1/(episode+1))  # ここでは0.5から減衰していくようなεを設定．
        if epsilon <= np.random.uniform(0,1):
            action = self.get_greedy_action(state)
        else:
            action = np.random.choice(self.num_action)
        return action


""" 各種設定
    学習に必要なパラメータの設定をします．
"""

# 学習の設定．
num_episode    = 1200 # 学習エピソード数．
penalty        = 10   # 途中でエピソードが終了したときのペナルティ．
num_discretize = 6    # 状態空間の分割数．

# ログ用の設定．
episode_rewards      = []
num_average_epidodes = 10


""" SARSAエージェントの学習
    今回は，エージェントの学習に，CartPole-v0と呼ばれる，台車に振子がついた環境を利用します．
    参考：https://github.com/openai/gym/wiki/CartPole-v0
"""

# シミュレータ環境の構築．
env = gym.make('CartPole-v0')

# エピソードの最大ステップ数．
max_steps = env.spec.max_episode_steps

# SARSAエージェントのインスタンスを作成．
agent = SarsaAgent(env.observation_space.shape[0], env.action_space.n, num_discretize)

for episode in range(num_episode):

    # エピソードを開始(環境の初期化)．
    observation = env.reset()

    # 観測の離散化(状態のインデックスを取得)．
    state = discretize_state(observation, num_discretize)

    # 行動を選択．
    action = agent.get_action(state, episode)

    episode_reward = 0
    for t in range(max_steps):
        observation, reward, done, _ = env.step(action)

        # もしエピソードの途中で終了してしまったらペナルティを加える．
        if done and t < max_steps - 1:
            reward = - penalty
        episode_reward += reward

        # 行動を実行し，次の状態，報酬，終端か否かの情報を取得．
        next_state = discretize_state(observation, num_discretize)

        # 次の行動を選択．
        next_action = agent.get_action(next_state, episode)

        # Q値の表を更新．
        agent.update_qtable(state, action, reward, next_state, next_action)
        
        # 状態と行動を更新
        state, action = next_state, next_action
        
        if done:
            break

    episode_rewards.append(episode_reward)
    if episode % 50 == 0:
        print("Episode %d finished | Episode reward %f" % (episode, episode_reward))

# 学習途中の累積報酬の移動平均を表示．
moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
plt.plot(np.arange(len(moving_average)),moving_average)
plt.title('SARSA: average rewards in %d episodes' % num_average_epidodes)
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

    # エピソードを開始(環境の初期化)．
    observation = env.reset()
    state = discretize_state(observation, num_discretize)
    
    # シミュレータ画面の出力．
    env.render()

    done = False
    while not done:

        # エージェントによる行動を取得．
        action = agent.get_greedy_action(state)
        
        # 行動を実行し，次の状態，報酬，終端か否かの情報を取得．
        next_observation, reward, done, _ = env.step(action)
        state = discretize_state(next_observation, num_discretize)

# 画面出力の終了．
env.close()