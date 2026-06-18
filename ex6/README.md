# 第6回B4輪講課題 — Transformer実装と機械翻訳

## 概要

> [!NOTE]
> - 本課題では**AIツールの使用を許可**します（コンペ後のため、効率的な学習を重視）
> - ただし、Transformerの核心部分は自分で理解して実装すること
> - PyTorchの基本機能のみを使用し、`torch.nn.Transformer`やattention関連などの高レベルAPIは使用禁止

本課題では **Encoder-Decoder Transformer** を実装し、**英語 → 日本語 翻訳** タスクで動作を確認する。  
"Attention Is All You Need"（Vaswani et al., 2017）の主要コンポーネントを穴埋め形式で実装することで、現代LLMの基礎となるアーキテクチャを深く理解することを目的とする。

---

## 背景・課題設定

Transformerは、2017年に提案された Self-Attention を中核とするアーキテクチャで、言語・画像・音声・音楽分野でも有力などの幅広いタスクで最高性能を達成している。

本課題を通じて、以下の目標を達成することが望まれる：

### **目標**
1. **Attentionの式の数学的な意味・解釈と長所を説明できる**
2. **Transformerとは何か説明できる**
3. **LLMの本質について説明できる**

---

## 前準備

依存ライブラリをインストールしてください：

```bash
pip install -r requirements.txt
```

---

## データセット

### **BSD ビジネス対話コーパス（Business Scene Dialogue）**

- **出典**: `ryo0634/bsd_ja_en`（HuggingFace Datasets）
- **内容**: ビジネスシーンの英日対話文
- **英語**: 単語レベルトークナイズ（デフォルト語彙サイズ 8,000）
- **日本語**: 文字レベルトークナイズ（デフォルト語彙サイズ 4,000）

### **データの取得**

`main.py` 実行時に `datasets` ライブラリが HuggingFace Hub から自動ダウンロードされます。

実行時のログ例（初回はダウンロードが走ります）：

```
data_loader - INFO - Loading ryo0634/bsd_ja_en ...
data_loader - INFO - Loaded 24000 sentence pairs (en→ja)
data_loader - INFO - Vocabulary size: 8000 (word-level)
data_loader - INFO - Vocabulary size: 2651 (char-level)
data_loader - INFO - Train: 22800 pairs (357 batches)
data_loader - INFO -  Val:  1200 pairs  (19 batches)
```

---

## 課題

### 6-1 Transformer の実装

`transformer_skeleton.py` 内の `#TODO` を埋めて、以下のクラスを完成させてください。各項に掲載している参照先にある数式を読み取って実装に反映させてみてください。
※ 完成後、`raise NotImplementedError`を削除することを忘れないでください。

#### **6-1-1 PositionalEncoding**
`Attention is All You Need`の`3.5 Positional Encoding`を参照

#### **6-1-2 MultiHeadAttention**
`Attention is All You Need`の`3.2 Attention`を参照

#### **6-1-3 FeedForward**
`Attention is All You Need`の`3.3 Position-wise Feed-Forward Networks`を参照

#### **6-1-4 EncoderBlock**
`Attention is All You Need`の`3.1 Encoder and Decoder Stacks` — Encoder の段落を参照

#### **6-1-5 DecoderBlock**
`Attention is All You Need`の`3.1 Encoder and Decoder Stacks` — Decoder の段落を参照

#### **6-1-6 TranslationModel**
`Attention is All You Need`の`3 Model Architecture` および `Figure 1` を参照

---

### 6-2 スケーリング実験

異なるモデルサイズで翻訳性能を比較します。そのための評価スクリプトevaluate.pyを作成し、評価を行ってください。以下の3点を出力してください。

#### **評価指標**
- **Perplexity（困惑度）**: 低いほど性能が良い
- **ChrF スコア**: 高いほど性能が良い
- 任意の入力に対する翻訳結果


#### **モデルサイズ設定**（`get_model_config` で定義済み）

| モデル | Encoder層 | Decoder層 | d_model | n_heads | d_ff  | パラメータ数（目安）|
|--------|-----------|-----------|---------|---------|-------|---------------------|
| tiny   | 2         | 2         | 128     | 4       | 512   | 2.5M                |
| small  | 3         | 3         | 256     | 8       | 1,024 | 8.5M                |
| medium | 4         | 4         | 256     | 8       | 1,024 | 10M                 |
| large  | 6         | 6         | 512     | 8       | 2,048 | 50M                 |

---

### 6-3 効率化手法の試行

`training_utils.py` に完成されたコードが提供されています。以下の2点について、コードを読んで仕組みを理解し、自分で調整をして精度や効率が変化するか観察してください。

#### **6-3-1 Gradient Accumulation**
- バッチサイズを疑似的に大きくする手法
- `--grad_accumulation 4` のように指定可能

#### **6-3-2 Learning Rate Scheduling（Warmup + Cosine Decay）**
- 学習率を動的に変更する手法
- `--warmup_steps 4000` のように指定可能（デフォルト: 4000）

---

## 実装仕様

### **提供ファイル**

| ファイル | 内容 |
|----------|------|
| `transformer_skeleton.py` | 実装箇所（`#TODO`）が明示されたTransformer |
| `data_loader.py` | BSD コーパスのロード・トークナイズ・DataLoader |
| `training_utils.py` | 学習ループ・ロギング・サンプル翻訳表示関数の集合 |
| `test_implementation.py` | 実装の妥当性をテストするスクリプト |
| `main.py` | 学習実行スクリプト |

### **実行コマンド**

```bash
# 実装のチェック
python test_implementation.py

# tiny モデルで動作確認（推奨スタート）
python main.py --model_size tiny --epochs 10

# small モデルで本格学習
python main.py --model_size small --epochs 30 --batch_size 128

# オプション一覧
python main.py --help
```

### **バッチデータの形式**

DataLoader は各バッチを `(src, tgt_in, tgt_out)` の3タプルで返します：

| テンソル | 形状 | 内容 |
|----------|------|------|
| `src` | `(batch, src_len)` | Encoder 入力（英語、EOS 付き） |
| `tgt_in` | `(batch, tgt_len)` | Decoder 入力（BOS から始まる日本語） |
| `tgt_out` | `(batch, tgt_len)` | Decoder 正解（EOS で終わる日本語） |

`tgt_in` と `tgt_out` の関係（Teacher Forcing）：

```
tgt_in  = [BOS, char1, char2, char3]
tgt_out = [char1, char2, char3, EOS]
```

---

## 実験設定

### **デフォルトハイパーパラメータ**

```python
# main.py のデフォルト値
BATCH_SIZE = 64
MAX_LEN    = 64    # 最大トークン長（英語：単語数、日本語：文字数）
EPOCHS     = 20
DROPOUT    = 0.1
WARMUP_STEPS = 4000

# モデルサイズ別の学習率
LR_TINY   = 1e-3
LR_SMALL  = 5e-4
LR_MEDIUM = 3e-4
LR_LARGE  = 1e-4
```

---

## 出力例

### **学習ログ（small モデル、10エポック）**

```
2026-06-17 19:55:30 - Epoch 2/10
2026-06-17 19:55:35 - Train - Loss: 4.6973, LR: 7.43e-05
2026-06-17 19:55:35 - Val   - Loss: 3.9560, Perplexity: 52.25
2026-06-17 19:55:35 - --- Sample translations (epoch 2) ---
2026-06-17 19:55:35 -   EN: I will check the schedule .
2026-06-17 19:55:35 -   JA: そうです。
...
2026-06-17 19:56:25 - Epoch 10/10
2026-06-17 19:56:25 - Train - Loss: 2.4041, LR: 3.71e-04
2026-06-17 19:56:25 - Val   - Loss: 2.2110, Perplexity: 9.13
2026-06-17 19:56:25 - --- Sample translations (epoch 10) ---
2026-06-17 19:56:25 -   EN: I will check the schedule .
2026-06-17 19:56:25 -   JA: その後は、私のメールを送ります。
2026-06-17 19:56:25 -   EN: Thank you for your help .
2026-06-17 19:56:25 -   JA: ありがとうございます、ありがとうございます。
2026-06-17 19:56:25 -   EN: Please send me the report .
2026-06-17 19:56:25 -   JA: それでは、送りますね。
2026-06-17 19:56:25 -   EN: The meeting is at three o'clock .
2026-06-17 19:56:25 -   JA: 10月の10日の10時ですね。
2026-06-17 19:56:27 - Training completed in 63.5s
```

---

## 発展課題（任意）

### **実験の拡張**
- BLEU スコアによる翻訳品質の定量評価（Tokenizerの変更）
- データサイズを変えたときの性能変化の観察
- より大きな語彙・長い系列での実験（別コーパスの適用）
- 別タスクへの応用（Q&Aボット、画像認識など）
- 学習率を変えることによるLossや性能の変化の観察

---

## 発表内容（次週）

以下を周囲にわかるように説明すること：

1. **実装した Transformer アーキテクチャの説明**
   - Self-Attention・Cross-Attention の仕組み
   - Encoder-Decoder 構造とその役割
   - 実装で工夫した点・苦労した点

2. **スケーリング実験の結果**
   - モデルサイズと Perplexity の関係
   - 学習時間・GPU メモリ使用量の比較  
     （ヒント: 学習後に `torch.cuda.max_memory_allocated()` で最大使用量を確認できる）

3. **考察・今後の展望**
   - 翻訳結果を見た所感
   - より良い翻訳に向けて何が必要か

---

## 注意事項

- 自分の作業ブランチで課題を行うこと
- プルリクエストを送る際には**学習結果（Perplexity 推移グラフ）と翻訳例も載せること**
- プルリクエストのコメントには実行コマンドと計算環境（GPU名、学習時間など）も記載すること
- 作業前にリポジトリを最新版に更新すること

```bash
git checkout main
git fetch upstream
git merge upstream/main
```

### **計算リソースについて**

- RTX 4070 相当の GPU を推奨（small モデルで ~64 秒/10エポック）
- GPU が利用できない場合は CPU でも実行可能（tiny モデル推奨）
- メモリ不足の場合は `--batch_size` や `--max_len` を下げること

---

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Visual explanation
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Code walkthrough
- [ryo0634/bsd_ja_en](https://huggingface.co/datasets/ryo0634/bsd_ja_en) — BSD データセット
