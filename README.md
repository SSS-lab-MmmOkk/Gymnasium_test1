# SUMO, Gymnasium, Mesa連携 横断歩道シミュレーション

## 概要

このプロジェクトは、SUMO、Gymnasium、Mesaを連携させ、横断歩道における車両と歩行者の相互作用をシミュレートするものです。

## 使い方

1. **依存関係のインストール**

   ```bash
   pip install -r requirements.txt
   ```

2. **SUMOのインストール**

   Debian/Ubuntuベースのシステムでは、以下のコマンドでSUMOをインストールできます。

   ```bash
   sudo apt-get update && sudo apt-get install -y sumo sumo-tools
   ```

3. **シミュレーションの実行**

   ```bash
   python main.py
   ```

   これにより、5回の繰り返しテストが実行されます。

## ファイル構成

- `main.py`: シミュレーションの実行ファイル
- `cross_env.py`: Gymnasiumのカスタム環境
- `agents.py`: Mesaのエージェント（歩行者、車両）
- `cross.sumocfg`: SUMOの設定ファイル
- `cross.net.xml`: SUMOのネットワークファイル
- `cross.rou.xml`: SUMOのルートファイル
- `nodes.xml`: SUMOのノード定義ファイル
- `edges.xml`: SUMOのエッジ定義ファイル
- `requirements.txt`: Pythonの依存関係ファイル
