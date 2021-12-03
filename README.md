# **Reinforcement Learning Tutorial**

様々な強化学習(Reinforcement Learning)手法をまとめたチュートリアル用のリポジトリです．

## **目次**

1. [**開発環境について**](#開発環境について)
    1. [Ubuntu](#1-ubuntu)
    2. [ROS](#2-ros)
    3. [Docker](#3-docker)

---

2. [**開発の進め方**](#開発の進め方)
    1. [コーディングスタイル](#1-コーディングスタイル)
    2. [Gitの使用方法](#2-gitの使用方法)
    3. [Gitの命名規則](#3-gitの命名規則)
    4. [Dockerの使用方法](#4-dockerの使用方法)
    5. [Docker Workspaceの使用方法](#5-docker-workspaceの使用方法)
    6. [ROSの使用方法](#6-rosの使用方法)

---

3. [**SOBITSのロボットについて**](#sobitsのロボットについて)
    1. <a href="https://gitlab.com/TeamSOBITS/sobit_education" target="_blank">SOBIT EDUを動かす</a>
    2. <a href="https://gitlab.com/TeamSOBITS/sobit_mini" target="_blank">SOBIT MINIを動かす</a>
    3. <a href="https://gitlab.com/TeamSOBITS/sobit_pro" target="_blank">SOBIT PROを動かす</a>

## **開発環境について**
以下の開発環境で動作させることを想定しています．
- Ubuntu : 18.04
- Python : 2.7-3.6 (デフォルト:2.7)


pipで強化学習ライブラリ「OpenAI Gym」をインストールする手順は下記の通りです。

コマンドプロンプトを起動します。

下記のコマンドを実行し、念のためにpipをアップデートします。
```bash
$ python3 -m pip install --upgrade pip
```

下記のコマンドを実行し、「OpenAI Gym」をインストールします。
```bash
$ python3 -m pip install gym
```

「Atari社のゲーム」やその他の追加機能をフルで入れる場合は以下のコマンドを実行します。
```bash
$ python3 -m pip install 'gym[all]'
```

下記のサンプルプログラムが実行できればインストール成功です。
```
$ python3
$ import gym
```
