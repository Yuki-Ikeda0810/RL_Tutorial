# coding:utf-8


""" 方策に基づく手法 (Policy-based Methods)
    今回は，方策に基づく代表的な手法として，方策と価値関数の近似した
    Actor-Criticを扱います．
"""

# 必要なライブラリのインポート．
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt

# 「actor_critic_agent.py」をインポート．
import actor_critic_agent


""" 各種設定
    学習に必要なパラメータの設定をします．
"""

# 学習の設定．
num_episode         = 1200   # 学習エピソード数．
# penalty           = 10     # 途中でエピソードが終了したときのペナルティ．

# ログ用の設定．
episode_rewards = []
num_average_epidodes = 10

env = gym.make('CartPole-v0')          # シミュレータ環境の構築．
max_steps = env.spec.max_episode_steps # エピソードの最大ステップ数．

# REINFORCEエージェントのインスタンスを作成．
agent = actor_critic_agent.ActorCriticAgent(env.observation_space.shape[0], env.action_space.n)


""" Actor-Criticエージェントの学習
    今回は，エージェントの学習に，OpenAI GymのCartPole-v0と呼ばれる，台車に振子がついた環境を利用します．
    参考：https://github.com/openai/gym/wiki/CartPole-v0
"""

for episode in range(num_episode):
    state = env.reset()  # envからは4次元の連続値の観測が返ってくる．
    episode_reward = 0
    for t in range(max_steps):
        action, prob, state_value = agent.get_action(state)  # 行動を選択．
        next_state, reward, done, _ = env.step(action)
#         # もしエピソードの途中で終了してしまったらペナルティを加える．
#         if done and t < max_steps - 1:
#             reward = - penalty
        episode_reward += reward
        agent.add_memory(reward, prob, state_value)
        state = next_state
        if done:
            agent.update_policy()
            agent.reset_memory()
            break
    episode_rewards.append(episode_reward)
    if episode % 50 == 0:
        print("Episode %4d finished | Episode reward %f" % (episode, episode_reward))

# 累積報酬の移動平均を表示．
moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
plt.plot(np.arange(len(moving_average)),moving_average)
plt.title('Actor-Critic: average rewards in %d episodes' % num_average_epidodes)
plt.xlabel('episode')
plt.ylabel('rewards')
plt.show()

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
    env.render()                                          # シミュレータ画面の出力．

    done = False
    while not done:
        action = agent.get_greedy_action(state)           # エージェントによる行動を取得．
        state, reward, done, _ = env.step(action)         # 行動を実行し，次の状態，報酬，終端か否かの情報を取得．
        env.render()                                      # シミュレータ画面の出力．

# 画面出力の終了．
env.close()