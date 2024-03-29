# **Reinforcement Learning Tutorial**

様々な強化学習(Reinforcement Learning)手法をまとめたチュートリアル用のリポジトリです．

強化学習(Reinforcement Learning)は，機械学習の一種であり，コンピューターエージェントが動的環境と，繰り返し試行錯誤のやりとりを重ねることによってタスクを実行できるようになる手法です．
この学習手法により，エージェントは，タスクの報酬を最大化する一連の意思決定を行うことができます．
人間が介入したり，タスクを達成するために明示的にプログラムしたりする必要がないことが特徴です．

:warning: 本リポジトリでは，実機での学習も想定して，全てエピソードに対する報酬で評価を行うようにしています．最近は，ステップ数に対する報酬で強化学習手法の評価を行う例をよく見かけますが，これでは学習しているエピソードの途中のステップで評価に移ってしまいます．シミュレーション環境でしたら，評価後に学習していたエピソードとステップを簡単に復元できますが，実機では復元することは困難です．そのため，ステップを最後までやり切ったエピソード毎での評価を行うように実装しています．

## **目次**

### [**1. 開発環境について**](#開発環境について)
1. [動作環境](#1-動作環境)
2. [必要ライブラリのインストール](#2-必要ライブラリのインストール)

### [**2. シミュレータチュートリアル**](#シミュレータチュートリアル)
1. [CartPole](#1-cartpole)
2. [Pendulum](#2-pendulum)

### [**3. 強化学習チュートリアル**](#強化学習チュートリアル)
　**＜価値に基づく手法(Value-based Methods)＞**
1. テーブル解法による手法
    1. [SARSA](#1-sarsa)
    2. [Q-Learning](#2-q-learning)
2. NNによる価値関数の近似による手法
    1. [Deep Q-Network(DQN)](#3-deep-q-networkdqn)

　**＜方策に基づく手法(Policy-based Methods)＞**
1. 方策の近似による手法(方策勾配法)
    1. [REINFORCE](#4-reinforce)
2. 方策と価値関数の近似による手法
    1. [Actor-Critic](#5-actor-critic)
1. 連続値行動空間に対する手法
    1. [Deep Deterministic Policy Gradient(DDPG)](#6-deep-deterministic-policy-gradientddpg)

<br>

## **開発環境について**

### 1. 動作環境

以下の開発環境で動作させることを想定しています．
- Ubuntu     : 18.04
- Python     : 3.6
- NumPy      : 1.19.5
- Pytorch    : 1.9.1 (+cu102)
- OpenAI Gym : 0.21.0
- PyBullet   : 3.0.8

### 2. 必要ライブラリのインストール

Reinforcement Learning Tutorialに必要なライブラリをインストールする必要があります．
コマンドプロンプトを起動して，以下のコマンドを実行してください．

「pip」をアップデートします(念のため)．
```bash
$ python3 -m pip install --upgrade pip
```

「Numpy」をインストールします．
```bash
$ python3 -m pip install numpy
```

「matplotlib」をインストールします．
```bash
$ python3 -m pip install matplotlib
```

「Pytorch」をインストールします．
```bash
$ python3 -m pip install torch
$ python3 -m pip install torchvision
$ python3 -m pip install tqdm
```

「OpenAI Gym」をインストールします．
```bash
$ python3 -m pip install gym==0.21.0
```

「PyBullet」をインストールします．
```bash
$ python3 -m pip install pybullet==3.0.8
```

<br>

## **シミュレータチュートリアル**

### 1. CartPole

「RL_Tutorial/00.Simulator/Gym」のディレクトリ階層まで移動します．
以下のコマンドを実行することでシミュレータを動かします．
```bash
$ python3 cartpole_tutorial.py
```

### 2. Pendulum

「RL_Tutorial/00.Simulator/Gym」のディレクトリ階層まで移動します．
以下のコマンドを実行することでシミュレータを動かします．
```bash
$ python3 pendulum_tutorial.py
```

### 3. Inverted Pendulum

「RL_Tutorial/00.Simulator/PyBullet」のディレクトリ階層まで移動します．
以下のコマンドを実行することでシミュレータを動かします．
```bash
$ python3 inverted_pendulum_tutorial.py
```

### 4. Half Cheetah

「RL_Tutorial/00.Simulator/PyBullet」のディレクトリ階層まで移動します．
以下のコマンドを実行することでシミュレータを動かします．
```bash
$ python3 half_cheetah_tutorial.py
```

<br>

## **強化学習チュートリアル**

### 1. SARSA

「RL_Tutorial/01.SARSA」のディレクトリ階層まで移動します．
以下のコマンドを実行することで学習します．
```bash
$ python3 sarsa_tutorial.py
```

### 2. Q-Learning

「RL_Tutorial/02.Q-Learning」のディレクトリ階層まで移動します．
以下のコマンドを実行することで学習します．
```bash
$ python3 q_learning_tutorial.py
```

### 3. Deep Q-Network (DQN)

「RL_Tutorial/03.DQN」のディレクトリ階層まで移動します．
以下のコマンドを実行することで学習します．
```bash
$ python3 dqn_tutorial.py
```

### 4. REINFORCE

「RL_Tutorial/04.REINFORCE」のディレクトリ階層まで移動します．
以下のコマンドを実行することで学習します．
```bash
$ python3 reinforce_tutorial.py
```

### 5. Actor-Critic

「RL_Tutorial/05.Actor-Critic」のディレクトリ階層まで移動します．
以下のコマンドを実行することで学習します．
```bash
$ python3 actor_critic_tutorial.py
```

### 6. Deep Deterministic Policy Gradient (DDPG)

「RL_Tutorial/06.DDPG」のディレクトリ階層まで移動します．
以下のコマンドを実行することで学習します．
```bash
$ python3 ddpg_tutorial.py
```

### 7. Proximal Policy Optimization (PPO)

「RL_Tutorial/07.PPO」のディレクトリ階層まで移動します．
以下のコマンドを実行することで学習します．
```bash
$ python3 ppo_tutorial.py
```

### 8. Soft Actor-Critic (SAC)

「RL_Tutorial/08.SAC」のディレクトリ階層まで移動します．
以下のコマンドを実行することで学習します．
```bash
$ python3 sac_tutorial.py
```
