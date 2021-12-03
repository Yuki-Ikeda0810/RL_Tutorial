# **Reinforcement Learning Tutorial**

様々な強化学習(Reinforcement Learning)手法をまとめたチュートリアル用のリポジトリです．

<br>

## **目次**

1. [**開発環境について**](#開発環境について)
    1. [動作環境](#1-動作環境)
    2. [必要ライブラリのインストール](#2-必要ライブラリのインストール)

2. [**強化学習チュートリアル**](#強化学習チュートリアル)
    1. [Gym](#1-gym)
    2. [SARSA](#2-sarsa)
    3. [Q-Learning](#3-q-learning)
    4. [Deep Q-Network(DQN)](#3-deep-q-networkdqn)

<br>

## **開発環境について**

### 1. 動作環境

以下の開発環境で動作させることを想定しています．
- Ubuntu     : 18.04
- Python     : 3.6
- NumPy      : 1.19.5
- Pytorch    : 1.9.1 (+cu102)
- OpenAI Gym : 0.21.0

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
$ python3 -m pip install gym
$ python3 -m pip install 'gym[all]'
```

<br>

## **強化学習チュートリアル**

### 1. Gym

### 2. SARSA

### 3. Q-Learning

### 4. Deep Q-Network(DQN)
