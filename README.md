# Reinforcement Learning Tutorial

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
