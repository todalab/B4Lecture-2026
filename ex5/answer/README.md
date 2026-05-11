# 解答例

このディレクトリには第6回課題の解答例を置いている。

## 実行方法

```bash
uv run python answer/generate_data.py
uv run python answer/listen_data.py
uv run python answer/train_mlp.py --input answer/data/synthetic_music.npz --fig-dir answer/fig --epochs 200 --hidden 64 --lr 0.1
uv run python answer/gradient_check.py
```

データは `answer/data/`, 画像は `answer/fig/` に保存される。
`listen_data.py` は `answer/audio/` に WAV ファイルを書き出す。
macOS では `--play` を付けると最初の書き出し音を再生する。
課題として配布するときは, 生成済みの `synthetic_music.npz` をルートの `data/` に置く想定である。
`generate_data.py` はデータの作り方を示す参考実装であり, 学生が必ず実行する前提ではない。

## ファイル構成

- `generate_data.py`: 合成音楽データを作る
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

現在のデータでは, 上の実行例の設定で test accuracy はおよそ 0.66 になる。
100% に近い精度が簡単に出る課題ではないため, 特徴量やモデル構造の工夫を比較しやすい。
