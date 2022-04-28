# coding:utf-8


""" 方策に基づく手法 (Policy-based Methods)
    今回は，方策に基づく代表的な手法として，連続値の行動空間を持つ問題に対応した
    Deep Deterministic Policy Gradient(DDPG)を扱います．
"""

# 必要なライブラリのインポート．
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt

# 「ddpg_agent.py」をインポート．
import ddpg_agent


""" 各種設定
    学習に必要なパラメータの設定をします．
"""

# 学習の設定．
num_episode         = 250   # 学習エピソード数．
memory_size         = 50000 # replay bufferの大きさ．
initial_memory_size = 1000  # 最初に貯めるランダムな遷移の数．

# ログ用の設定．
episode_rewards = []
num_average_epidodes = 10

env = gym.make('Pendulum-v1')          # シミュレータ環境の構築．
max_steps = env.spec.max_episode_steps # エピソードの最大ステップ数．

# REINFORCEエージェントのインスタンスを作成．
agent = ddpg_agent.DdpgAgent(env.observation_space, env.action_space, memory_size=memory_size)


""" Actor-Criticエージェントの学習
    今回は，エージェントの学習に，OpenAI GymのPendulum-v0と呼ばれる，台車に振子がついた環境を利用します．
    参考：https://github.com/openai/gym/wiki/Pendulum-v0
"""

# 最初にreplay bufferにランダムな行動をしたときのデータを入れる．
state = env.reset()
for step in range(initial_memory_size):
    action = env.action_space.sample() # ランダムに行動を選択．
    next_state, reward, done, _ = env.step(action)
    transition = {
        'state': state,
        'next_state': next_state,
        'reward': reward,
        'action': action,
        'done': int(done)
    }
    agent.replay_buffer.append(transition)
    state = env.reset() if done else next_state
print('%d Data collected' % (initial_memory_size))


for episode in range(num_episode):
    state = env.reset()  # envからは3次元の連続値の観測が返ってくる．
    episode_reward = 0
    for t in range(max_steps):
        action = agent.get_action(state).data.numpy()  # 行動を選択．
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        transition = {
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'action': action,
            'done': int(done)
        }
        agent.replay_buffer.append(transition)
        agent.update()  # ActorとCriticを更新．
        state = next_state
        if done:
            break
    episode_rewards.append(episode_reward)
    if episode % 20 == 0:
        print("Episode %4d finished | Episode reward %f" % (episode, episode_reward))

# 累積報酬の移動平均を表示．
moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
plt.plot(np.arange(len(moving_average)),moving_average)
plt.title('DDPG: average rewards in %d episodes' % num_average_epidodes)
plt.xlabel('episode')
plt.ylabel('rewards')
plt.show()

env.close()


""" 最終的に得られた方策のテスト(可視化)
    5回のエピソードをシミュレーターで動作させ，学習後のエージェントの方策を可視化します．
"""

# シミュレータ環境の構築．
env = gym.make('Pendulum-v1')
env = wrappers.Monitor(env, "./movie", force=True)

# 3回のエピソードをシミュレーターで動作させる．
for i_episode in range(3):
    observation = env.reset()                             # エピソードを開始(環境の初期化)．
    env.render()                                          # シミュレータ画面の出力．

    done = False
    while not done:
        action = agent.get_action(state).detach().numpy() # エージェントによる行動を取得．
        state, reward, done, _ = env.step(action)         # 行動を実行し，次の状態，報酬，終端か否かの情報を取得．
        env.render()                                      # シミュレータ画面の出力．
    
# 画面出力の終了．
env.close()