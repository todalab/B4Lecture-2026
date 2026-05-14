# 解答例

このディレクトリには第6回課題の解答例を置いている。

## 実行方法

```bash
uv run python answer/prepare_speech_commands.py
uv run python answer/listen_data.py
uv run python answer/train_mlp.py --input answer/data/speech_commands_subset.npz --fig-dir answer/fig --epochs 200 --hidden 64 --lr 0.1
uv run python answer/gradient_check.py
```

初回の `prepare_speech_commands.py` 実行時に, Speech Commands の公式 tar.gz を `answer/tfds_data/` にダウンロードし, 課題用サブセットを `answer/data/` に保存する。
`answer/data/` と `answer/tfds_data/` は大きいため Git 管理しない。
画像は `answer/fig/` に保存される。
`listen_data.py` は `answer/audio/` に WAV ファイルを書き出す。
macOS では `--play` を付けると最初の書き出し音を再生する。
課題として取り組む場合は, ルートの `data/` に作ることもできる。

```bash
uv run python answer/prepare_speech_commands.py --output data/speech_commands_subset.npz --data-dir data/speech_commands_raw
```

## ファイル構成

- `prepare_speech_commands.py`: Speech Commands 公式データから課題用 NPZ を作る
- `listen_data.py`: データを WAV として書き出して聴けるようにする
- `data_utils.py`: データ読み込み, 特徴量抽出, 可視化用の補助関数
- `mlp.py`: numpy によるニューラルネットワーク本体
- `train_mlp.py`: 学習, 評価, 学習曲線と混同行列の保存
- `gradient_check.py`: バックプロパゲーションと数値微分の比較

## 実装のポイント

`mlp.py` では, 1層の隠れ層を持つ多クラス分類器を実装している。
順伝播では `linear -> ReLU -> linear -> softmax` を計算し, 損失にはクロスエントロピーを使う。
逆伝播では連鎖律により各パラメータの勾配を求め, 勾配降下法で更新している。

`gradient_check.py` では, 数値微分で求めた勾配とバックプロパゲーションで求めた勾配の差を表示する。
数値微分は理解しやすいが, パラメータ数が増えると非常に遅くなる。

Speech Commands は実際の発話データなので, 合成波形よりも話者差や発音差の影響が大きい。
100% に近い精度が簡単に出る課題ではないため, 特徴量やモデル構造の工夫を比較しやすい。
