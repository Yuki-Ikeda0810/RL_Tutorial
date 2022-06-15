# coding:utf-8


""" Soft Actor-Critic(SAC)のエージェントクラス
    Algorithmクラスを継承して，Soft Actor-Critic(SAC)のアルゴリズム(Trainerのself.algoの部分)を実装します．
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


""" SAC Actorクラス
"""
class SACActor(nn.Module):

    # コンストラクタ．
    def __init__(self, state_shape, action_shape):
        super().__init__()

        # Actorネットワーク．
        # nn.Linear ： 全結合(線型変換)．
        # nn.ReLU   ： 活性化(ReLU)．
        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * action_shape[0])
        )

    # 順伝播．
    # ミニバッチ中の状態(states)に対して，決定論な行動を返す関数．
    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])
    
    # ミニバッチ中の状態(states)に対して，決定論な行動とその行動の確率密度の対数を返す関数．
    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)

        # 数値計算を安定させるために，ネットワークから出力された標準偏差の対数を(−20,2)の範囲にクリッピング．
        return reparameterize(means, log_stds.clamp(-20, 2))


""" SAC Criticクラス
"""
class SACCritic(nn.Module):

    # コンストラクタ．
    def __init__(self, state_shape, action_shape):
        super().__init__()

        # Clipped Double Qというテクニックにより，2つの独立したネットワークを使用．

        # Criticネットワーク1．
        # nn.Linear ： 全結合(線型変換)．
        # nn.ReLU   ： 活性化(ReLU)．        
        self.net1 = nn.Sequential(
            nn.Linear(state_shape[0] + action_shape[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # Criticネットワーク2．
        # nn.Linear ： 全結合(線型変換)．
        # nn.ReLU   ： 活性化(ReLU)．
        self.net2 = nn.Sequential(
            nn.Linear(state_shape[0] + action_shape[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    # 順伝播．
    # ミニバッチ中の状態(states)と行動(actions)に対して，2つの状態行動価値を返す関数．
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net1(x), self.net2(x)


""" SACクラス
"""
class SAC(Algorithm):

    # コンストラクタ．
    def __init__(self, state_shape, action_shape, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                 seed=0, batch_size=256, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4, replay_size=10**6, 
                 start_episodes=10**4, tau=5e-3, alpha=0.2, reward_scale=1.0, max_episode_steps = 1000):
        super().__init__()

        # シードの設定．
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # リプレイバッファ．
        self.buffer = ReplayBuffer(buffer_size=replay_size,
                                   state_shape=state_shape,
                                   action_shape=action_shape,
                                   device=device
        )

        """Actor-Criticのネットワークを構築
        """
        # Actorネットワークを構築．
        self.actor = SACActor(state_shape=state_shape,
                              action_shape=action_shape
        ).to(device)

        # Criticネットワークを構築．
        self.critic = SACCritic(state_shape=state_shape,
                                action_shape=action_shape
        ).to(device)

        # Criticのターゲットネットワークを構築．
        self.critic_target = SACCritic(state_shape=state_shape,
                                       action_shape=action_shape
        ).to(device).eval()

        # Criticのターゲットネットワークの重みを初期化し，勾配計算を無効化．
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # オプティマイザの構築．
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # その他パラメータ．
        self.learning_steps = 0
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.start_episodes = start_episodes
        self.tau = tau
        self.alpha = alpha
        self.reward_scale = reward_scale
        self.max_episode_steps = max_episode_steps

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

    # 学習できるかどうかを判断する関数．
    def is_update(self, episode):

        # 学習初期の一定期間(start_episodes)は学習しない．
        return episode >= max(self.start_episodes, self.batch_size)
    
    # ActorやCriticのパラメータを更新する関数．
    def update(self):
        self.learning_steps += 1

        # 指定したバッチサイズ分のミニバッチを作成．
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        # ActorやCriticのパラメータを更新．
        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)
        self.update_target()
    
    # Criticネットワークのパラメータを更新する関数．
    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        # Criticネットワークの損失を算出．
        # 損出関数として平均二乗ベルマン誤差を使用． 
        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        # 【補足】
        # 2つのネットワークで同一のターゲットを用いる．
        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        # 逆伝播．
        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

    # Actorネットワークのパラメータを更新する関数．
    def update_actor(self, states):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)

        # Actorネットワークの損失を算出．
        loss_actor = (self.alpha * log_pis - torch.min(qs1, qs2)).mean()

        # 逆伝播．
        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

    # Criticのターゲットネットワークのパラメータを更新する関数．
    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)