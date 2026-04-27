# 解答例

このディレクトリには第6回課題の解答例を置いている．

## ファイル構成

```
.
├── generate_data.py   # データ生成スクリプト
├── hmm.py             # Forward / Viterbi アルゴリズム実装
├── evaluate.py        # 混同行列・正解率・可視化ユーティリティ
└── main.py            # メインスクリプト（実験の実行）
```

## 実行方法

### 1. データ生成

```bash
uv run python generate_data.py
```

`data/` ディレクトリに `data1.pickle` 〜 `data4.pickle` が生成される．

### 2. 課題の実行

```bash
# 1つのデータセット
uv run python main.py --data data/data1.pickle

# 全データセットをまとめて実行
uv run python main.py --data data/data1.pickle data/data2.pickle data/data3.pickle data/data4.pickle
```

画像は `fig/` に保存される．

## 実装の説明

### hmm.py

- **`forward_algorithm(O, PI, A, B)`**：対数スケールで前向き確率を計算し，log P(O|λ) を返す．
  アンダーフロー対策として `log-sum-exp` トリックを使用している．

- **`viterbi_algorithm(O, PI, A, B)`**：対数スケールで Viterbi 変数 δ を計算し，最適パスの対数確率を返す．
  漸化式の `sum` を `max` に置き換えただけなので Forward と構造はほぼ同一．

- **`predict_models(outputs, PI, A, B, algorithm)`**：
  全系列 × 全モデルのスコアを計算し，`argmax` で推定モデルを返す．

### evaluate.py

- **`confusion_matrix`**：O(p) で混同行列を構築．
- **`accuracy`**：正解率を計算．
- **`plot_results`**：Forward / Viterbi それぞれの混同行列と正解率・計算時間をまとめた図を保存．

## 結果の考察ポイント

- Forward と Viterbi の正解率の差：全パスを考慮する Forward の方が理論上は正確だが，実用上は Viterbi と大差ないことが多い
- 計算時間の差：Viterbi は `sum` の代わりに `max` を使うため，Forward よりわずかに高速
- Left-to-Right vs Ergodic：Left-to-Right は遷移が制限されるため，モデルの識別がしやすくなる傾向がある
- クラス数が増えると混同行列が大きくなり，誤分類パターンが見えやすくなる
