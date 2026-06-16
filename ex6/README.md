# 第6回B4輪講課題 — Transformer実装

## 概要

> [!NOTE]
> - 本課題では**AIツールの使用を許可**します（コンペ後のため、効率的な学習を重視）
> - ただし、Transformerの仕組みを自分で理解してから実装すること
> - `torch.nn.Transformer` などの高レベルAPIは使用禁止（自分で組み立てる意味がなくなるため）

本課題では **Transformerアーキテクチャを一から実装し、言語モデルとして動かす** ことを目標とします。

---

## 課題の進め方

### ステップ1: スケルトンを実装する

`transformer_skeleton.py` の `★ 実装箇所` を全て埋めてください。

実装が必要な箇所は以下の3クラスです：

| クラス | 実装箇所 |
|---|---|
| `PositionalEncoding` | `__init__` でpe行列を構築、`forward` で入力に加算 |
| `MultiHeadAttention` | `__init__` でLinear層を定義、`scaled_dot_product_attention` と `forward` を実装 |
| `TransformerBlock` | `__init__` で各レイヤーを定義、`forward` でResidual + LayerNorm |

それ以外（`FeedForward`, `LanguageModel`, `data_loader.py`, `training_utils.py`）は提供済みです。

### ステップ2: テストを全て通す

```bash
cd ex6
python test_implementation.py
```

全6テストが `✅` になれば実装完了です。

```
🎉 All tests passed! (6/6)
実装が完了しています。main.py で学習を開始できます。
```

### ステップ3: 実際に学習を回してスケーリングを確認する

```bash
# まず小さいモデルで動作確認
python main.py --model_size tiny --dataset shakespeare --epochs 5

# スケーリング実験（各モデルサイズで実行）
python main.py --model_size tiny   --dataset shakespeare --epochs 10
python main.py --model_size small  --dataset shakespeare --epochs 10
python main.py --model_size medium --dataset shakespeare --epochs 10
```

---

## データセット

データセットはリポジトリに同梱済みです。ダウンロード不要です。

```
ex6/data/
├── shakespeare.txt          # シェイクスピア作品（約1MB、文字レベル）
└── wikitext-2/
    ├── train.txt            # WikiText-2 訓練データ（単語レベル）
    └── valid.txt            # WikiText-2 検証データ
```

### 2つのデータセットの使い方

各データセットで **別々にモデルを学習** し、結果を比較してください。混在させません。

| データセット | 粒度 | 用途 |
|---|---|---|
| Shakespeare | 文字レベル | 実装テスト用（小さくて高速） |
| WikiText-2 | 単語レベル | スケーリング実験のメイン |

---

## 提供ファイル一覧

| ファイル | 状態 | 説明 |
|---|---|---|
| `transformer_skeleton.py` | **要実装** | Transformer本体（★箇所を埋める） |
| `data_loader.py` | 完成済み | データの読み込みとトークナイズ |
| `training_utils.py` | 完成済み | 学習ループ・LRスケジューラ・logging |
| `evaluate.py` | 完成済み | 評価・可視化 |
| `test_implementation.py` | 完成済み | 実装確認テスト |
| `main.py` | 完成済み | 学習のエントリポイント |

---

## スケーリング実験

異なるモデルサイズで学習し、パラメータ数と性能の関係を確認してください。

| モデル | 層数 | 隠れ次元 | ヘッド数 | パラメータ数 |
|--------|------|----------|----------|--------------|
| Tiny   | 4    | 128      | 4        | ~1M          |
| Small  | 6    | 256      | 8        | ~5M          |
| Medium | 8    | 512      | 8        | ~20M         |
| Large  | 12   | 768      | 12       | ~85M         |

---

## 実験結果（PRに記載すること）

PRを送る際は、以下の表を埋めてください（自分で実測した値）。

### 参考設定

```
BATCH_SIZE = 32, SEQ_LEN = 128
```

### Shakespeare（文字レベル）

| モデル | Val Perplexity | GPU Memory | 学習時間/epoch |
|--------|---------------|-----------|---------------|
| Tiny   |               |           |               |
| Small  |               |           |               |
| Medium |               |           |               |

### GPU使用量の計測方法

別ターミナルで以下を実行しながら学習するだけです：

```bash
# 1秒ごとにGPUメモリ使用量を表示
watch -n 1 nvidia-smi

# または1行で流す（ログに残しやすい）
nvidia-smi dmon -s mu -d 1
```

学習中の最大値を記録してください。

---

## 評価指標

**Perplexity**（困惑度）: 言語モデルの標準的な性能指標。小さいほど良い。

$$\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i \mid w_1,\ldots,w_{i-1})\right)$$

スケーリング法則の確認: パラメータ数 vs Perplexity を対数スケールでプロットしてください。

---

## 発展課題（任意）

余力があれば取り組んでください。

- **RoPE**（Rotary Position Embedding）の実装
- **KV-Cache** による推論高速化
- **Flash Attention** の調査・実装
- **Mixed Precision Training**（`torch.cuda.amp`）による学習高速化
- データサイズ・モデルサイズの同時スケーリング実験

---

## PRの要件

- `test_implementation.py` の全テストが通っていること
- 学習結果（Perplexity・GPU Memory・学習時間）の表を埋めること
- 学習曲線のグラフを添付すること（`evaluate.py` で生成できます）
- 生成テキストの例を1つ載せること
- 実行コマンドと計算環境（GPU型番など）を記載すること

---

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556)
