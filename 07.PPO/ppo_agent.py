# coding:utf-8


""" Proximal Policy Optimization(PPO)のエージェントクラス
    Algorithmクラスを継承して，Proximal Policy Optimization(PPO)のアルゴリズム(Trainerのself.algoの部分)を実装します．
"""

# 必要なライブラリのインポート．
import os
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from calculation  import reparameterize  #「calculation.py」の「reparameterize」関数をインポート．
from algorithm    import Algorithm       #「algorithm.py」の「Algorithm」クラスをインポート．
from replaybuffer import ReplayBuffer    #「replaybuffer.py」の「ReplayBuffer」クラスをインポート．


""" PPO Actorクラス
"""
class PPOActor(nn.Module):

    # コンストラクタ．
    def __init__(self, state_shape, action_shape):
        super().__init__()

        # Actorネットワーク．
        # nn.Linear ： 全結合(線型変換)．
        # nn.ReLU   ： 活性化(ReLU)．
        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_shape[0]),
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    # 順伝播．
    # ミニバッチ中の状態(states)に対して，決定論な行動を返す関数．
    def forward(self, states):
        return torch.tanh(self.net(states))
    
    # ミニバッチ中の状態(states)に対して，決定論な行動とその行動の確率密度の対数を返す関数．
    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    # ミニバッチ中の状態(states)と行動(actions)に対して，現在の方策における行動の確率密度の対数を返す関数．
    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


""" PPO Criticクラス
"""
class PPOCritic(nn.Module):

    # コンストラクタ．
    def __init__(self, state_shape, action_shape):
        super().__init__()

        # Criticネットワーク1．
        # nn.Linear ： 全結合(線型変換)．
        # nn.ReLU   ： 活性化(ReLU)．        
        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    # 順伝播．
    # ミニバッチ中の状態(states)に対して，状態価値を返す関数．
    def forward(self, states):
        return self.net(states)

""" アドバンテージ(行動価値 - 状態価値)を推定する関数
    PPOでは，Generalized Advantage Estimation(GAE)を用いて，状態価値のターゲットとGAEを計算する． 
"""
def calculate_advantage(values, rewards, dones, next_values, gamma=0.995, lambd=0.997):

    # TD誤差を計算．
    deltas = rewards + gamma * next_values * (1 - dones) - values

    # GAEを初期化．
    advantages = torch.empty_like(rewards)

    # 終端ステップを計算．
    advantages[-1] = deltas[-1]

    # 終端ステップの1つ前から，順番にGAEを計算．
    for t in reversed(range(rewards.size(0) - 1)):
        advantages[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * advantages[t + 1]

    # 状態価値のターゲットをλ-収益として計算．
    targets = advantages + values

    # GAEを標準化．
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return targets, advantages


""" PPOクラス
"""
class PPO(Algorithm):

    # コンストラクタ．
    def __init__(self, state_shape, action_shape, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 seed=0, batch_size=64, gamma=0.995, lr_actor=3e-4, lr_critic=3e-4, rollout_length=2048, 
                 num_updates=10, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=0.5):
        """
        def __init__(self, state_shape, action_shape, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                 seed=0, batch_size=256, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4, replay_size=10**6, 
                 start_episodes=10**4, tau=5e-3, alpha=0.2, reward_scale=1.0, max_episode_steps = 1000):
        """
        super().__init__()

        # シードの設定．
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # ロールアウトバッファ．
        self.buffer = RolloutBuffer(buffer_size=rollout_length,
                                    state_shape=state_shape,
                                    action_shape=action_shape,
                                    device=device
        )

        """Actor-Criticのネットワークを構築
        """
        # Actorネットワークを構築．
        self.actor = PPOActor(state_shape=state_shape,
                              action_shape=action_shape,
        ).to(device)

        # Criticネットワークを構築．
        self.critic = PPOCritic(state_shape=state_shape
        ).to(device)

        # オプティマイザの構築．
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # その他パラメータ．
        self.learning_steps = 0
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.num_updates = num_updates
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        """ self.start_episodes = start_episodes
        """

    # 環境(env)，現在のエピソード数(episode)をアルゴリズムに渡し，
    # エピソードの終了までステップを繰り返す関数．
    def episode(self, env, episode):

        # 環境の初期化．
        steps = 0
        episode_return = 0
        state = env.reset()
        done = False

        # エピソードの終了条件まで繰り返す．
        while not done:
            steps += 1

            # 学習初期の一定期間(start_episodes)はランダムに行動(多様なデータの収集を促進)．
            if episode <= self.start_episodes:
                action = env.action_space.sample()
            else:
                action = self.exploit(state)

            # 行動を実行し，次の状態，報酬，終端か否かの情報を取得．
            next_state, reward, done, _ = env.step(action)

            # 【補足】
            # ゲームオーバーによってエピソード終了した場合には，「done_masked=True」が適切である．
            # もし，ゲームオーバーではなく，最大ステップ数に到達したことでエピソードが終了した場合は，
            # 本来であればその先も試行が継続するはずなので，終了シグナルは「False」にする．
            # 以下の実装では，"たまたま"最大ステップ数でゲームオーバーとなった場合に「done_masked=False」になってしまう．
            # この場合は稀で，多くの実装ではその誤差を無視しているため，今回もこれについては無視する．
            if steps == self.max_episode_steps:
                done_masked = False
            else:
                done_masked = done

            # リプレイバッファにデータを追加．
            self.buffer.append(state, action, reward, done_masked, next_state)

            # アルゴリズムが準備できている場合に学習．
            if self.is_update(episode):
                self.update()

            state = next_state
            episode_return += reward

        return episode_return
    
    """
    def step(self, env, state, t, steps):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)

        # ゲームオーバーではなく，最大ステップ数に到達したことでエピソードが終了した場合は，
        # 本来であればその先も試行が継続するはず．よって，終了シグナルをFalseにする．
        # NOTE: ゲームオーバーによってエピソード終了した場合には， done_masked=True が適切．
        # しかし，以下の実装では，"たまたま"最大ステップ数でゲームオーバーとなった場合には，
        # done_masked=False になってしまう．
        # その場合は稀で，多くの実装ではその誤差を無視しているので，今回も無視する．
        if t == env._max_episode_steps:
            done_masked = False
        else:
            done_masked = done

        # バッファにデータを追加する．
        self.buffer.append(state, action, reward, done_masked, log_pi, next_state)

        # エピソードが終了した場合には，環境をリセットする．
        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    # 学習できるかどうかを判断する関数．
    def is_update(self, episode):

        # ロールアウト1回分のデータが溜まったら学習．
        return episode % self.rollout_length == 0
    

    def update(self):
        self.learning_steps += 1

        states, actions, rewards, dones, log_pis, next_states = self.buffer.get()

        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
        targets, advantages = calculate_advantage(values, rewards, dones, next_values, self.gamma, self.lambd)

        # バッファ内のデータを num_updates回ずつ使って，ネットワークを更新する．
        for _ in range(self.num_updates):
            # インデックスをシャッフルする．
            indices = np.arange(self.rollout_length)
            np.random.shuffle(indices)

            # ミニバッチに分けて学習する．
            for start in range(0, self.rollout_length, self.batch_size):
                idxes = indices[start:start+self.batch_size]
                self.update_critic(states[idxes], targets[idxes])
                self.update_actor(states[idxes], actions[idxes], log_pis[idxes], advantages[idxes])

    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        # 学習を安定させるヒューリスティックとして，勾配のノルムをクリッピングする．
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, log_pis_old, advantages):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        mean_entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * advantages
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages
        loss_actor = torch.max(loss_actor1, loss_actor2).mean() - self.coef_ent * mean_entropy

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        # 学習を安定させるヒューリスティックとして，勾配のノルムをクリッピングする．
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()
    """