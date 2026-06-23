# What

## 6-1 Transformer の実装

#### **6-1-1 PositionalEncoding**
- 偶数列に$\sin{\frac{pos}{10000^{2i/d_{model}}}}$を、奇数列に$\cos{\frac{pos}{10000^{2i/d_{model}}}}$を格納した。
- 計算の際にlogとexpを使った式に変形して、オーバーフローを防いだ。

#### **6-1-2 MultiHeadAttention**
- アテンション重みを$ \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) $として計算した。
- バッチやヘッドの次元が含まれるので転置ではなく一時的にtransposeで次元を入れ替えて計算した。
- マスクの箇所を$ -\infty $にして、softmaxの計算で0になるようにした。
- ドロップアウトは、アテンション重みの計算後に適用した。
- アテンション重みとVの行列積を計算した。

#### **6-1-3 FeedForward**
- 2層の全結合ネットワークを実装した。
- 1層目の活性化関数はReLUを使用した。

#### **6-1-4 EncoderBlock**
- 自己注意 → Add&Norm → FFN → Add&Norm の順に処理を行うように実装した。
- Add&Normの前にドロップアウトを適用した。

#### **6-1-5 DecoderBlock**
- マスク自己注意 → Add&Norm → クロスアテンション → Add&Norm → FFN → Add&Norm の順に処理を行うように実装した。
- Add&Normの前にドロップアウトを適用した。

#### **6-1-6 TranslationModel**
- ソースの有効トークン位置を示すマスクを生成した。
- ターゲットの未来遮蔽マスクと PAD 除外マスクを合成して生成した。
- Encoder の順伝播を実装した。
- Decoder の順伝播を実装した。
- 出力の次元を語彙サイズに射影する線形層を実装した。
- 損失計算を実装した。
- CrossEntropyLoss は内部で softmax を計算するので、logits をそのまま渡すようにした。


### **モデルサイズ設定**
| モデル | epochs | batch_size |
|--------|--------|------------|
| tiny   | 10     | 64         |
| small  | 30     | 128        |
| medium | 40     | 256        |
| large  | 50     | 512        |

### 実行コマンド
```
# 環境構築
uv sync

# tiny モデルで学習
uv run main.py --model_size tiny --epochs 10 --batch_size 64

# small モデルで学習
uv run main.py --model_size small --epochs 30 --batch_size 128

# medium モデルで学習
uv run main.py --model_size medium --epochs 40 --batch_size 256

# large モデルで学習
uv run main.py --model_size large --epochs 50 --batch_size 512
```

---


## 6-2 スケーリング実験

### **評価指標**
- **Perplexity（困惑度）**: 正解となる単語の予測確率をもとに計算され、低いほど性能が良い。`training_utils.py`の`evaluate`関数を使用して計算した。
- **ChrF スコア**: 文字レベルのn-gramの一致度に基づくF値で計算され、高いほど性能が良い。トークナイズの影響を受けにくい。`sacrebleu`ライブラリの`corpus_chrf`関数を使用して計算した。
- 任意の入力に対する翻訳結果

### **実行コマンド**
```
# 単一モデルを評価
uv run evaluate.py --model_size tiny

# 学習済み全モデルを横並びで比較
uv run evaluate.py --compare

# 任意の英文を翻訳
uv run evaluate.py --model_size large \
  --translate "I will check the schedule ." "Thank you for your help ."
```

### **評価結果**

| モデル | Perplexity | ChrF スコア |
|--------|------------|-------------|
| tiny   | 9.28      | 11.95       |
| small  | 8.18      | 13.79       |
| medium | 8.22       | 13.40       |
| large  | 8.58       | 13.60       |

largeモデルについてパラメータ数に対してスコアがあまり伸びなかったので300エポックまで学習して各エポックごとのスコアを確認したが、これ以上の伸びは見られなかった。

- tinyモデルの翻訳結果
```
EN: I will check the schedule .
JA: 私のメールを送ります。
EN: Thank you for your help .
JA: お願いします、お願いします。
EN: Please send me the report .
JA: メールの件は送りをお願いします。
EN: The meeting is at three o'clock .
JA: 10時では10時です。
```

- smallモデルの翻訳結果
```
EN: I will check the schedule .
JA: スケジュールを確認してみます。
EN: Thank you for your help .
JA: どうもありがとうございました。
EN: Please send me the report .
JA: 報告書を送ってください。
EN: The meeting is at three o'clock .
JA: 1時から3時以上です。
```

- mediumモデルの翻訳結果
```
EN: I will check the schedule .
JA: その後に、スケジュールを確認します。
EN: Thank you for your help .
JA: ありがとうございます。
EN: Please send me the report .
JA: 確認の後にお願いします。
EN: The meeting is at three o'clock .
JA: 1時から15時からのミーティングがあるのです。
```

- largeモデルの翻訳結果
```
EN: I will check the schedule .
JA: 確認して、確認しますね。
EN: Thank you for your help .
JA: ありがとうございました。
EN: Please send me the report .
JA: では、私のメールを送ってください。
EN: The meeting is at three o'clock .
JA: 1時から1時から1時です。
```

---

## 6-3 効率化手法の試行

### **6-3-1 Gradient Accumulation**
- バッチサイズを疑似的に大きくする手法
- 複数バッチ間にわたって更新を行わずに勾配を蓄積し、指定した回数のバッチ処理後にまとめて更新する。
grad_accumulation以外のオプションは同じままにして、以下のコマンドで学習を行った。
```
$ uv run main.py --model_size large --epochs 50 --batch_size 512 --grad_accumulation 2
$ uv run main.py --model_size large --epochs 100 --batch_size 512 --grad_accumulation 2
```
| モデル | エポック数 | grad_accumulation | Perplexity | ChrF スコア |
|--------|------------|-------------------|------------|-------------|
| large  | 50          | 1                 | 8.58       | 13.60       |
| large  | 50         | 2                 | 23.92      | 9.05
| large  | 100        | 2                 | 10.11      | 11.75       |

- grad_accumulationを2に設定した場合、Perplexityが大幅に悪化し、ChrFスコアも低下した。これは更新の回数が半分になるため、学習が十分に進まなかった可能性がある。
- エポック数を倍にしたところ、性能は改善したが、grad_accumulationを使用しない場合と比較すると、Perplexityは悪化し、ChrFスコアも低下した。

### **6-3-2 Learning Rate Scheduling（Warmup + Cosine Decay）**
- 学習率を動的に変更する手法
- warmup期間中は学習率を徐々に増加させ、その後はコサイン関数に基づいて減少させる。
- 学習初期の不安定な更新を防ぎ、学習の収束を促進する効果がある。
- 学習終盤での過学習を抑制する効果もある。
- バッチサイズが512、エポック数が80のlargeモデルで、学習データは19,000のとき総ステップ数は3,040であり、warmupステップ数を総ステップ数の5%(152ステップ)、10%(304ステップ)、20%(608ステップ)に設定して学習を行った。
```
# 総ステップ数の5%をwarmupに設定
$ uv run main.py --model_size large --epochs 80 --batch_size 512 --warmup_steps 152
# 総ステップ数の10%をwarmupに設定
$ uv run main.py --model_size large --epochs 80 --batch_size 512 --warmup_steps 304
# 総ステップ数の20%をwarmupに設定
$ uv run main.py --model_size large --epochs 80 --batch_size 512 --warmup_steps 608
```
| モデル | エポック数 | warmup_steps | Perplexity | ChrF スコア |
|--------|------------|--------------|------------|-------------|
| large  | 80         | 152          | 8.56       | 13.79       |
| large  | 80         | 304          | 8.63      | 13.60      |
| large  | 80         | 608          | 8.60       | 13.71       |

あまり差はみられなかった。
---

# Result


# チェック項目
## PR作成前
- タイトルは「[名前] exXX 解答」になっている
- コメントが適切に書けている
- 変数名，関数名はわかりやすいものになっている
- 出力画像も添付する
- どうしても解決できない部分がある場合は、その詳細も書くこと


## PR作成後
- [ ] Reviewersを設定
  - [ ] Reviewerを[スプシ](https://docs.google.com/spreadsheets/d/107ZnRHP5DPrRuTkXAlcfQkIh0TwovE6wnDbFSHcUfew/edit?gid=1854832715#gid=1854832715)から探す
- [ ] Assigneesを設定
  - [ ] 自分自身を入れる
  - [ ] Reviewerを入れる
- [ ] SlackでReviewerにReviewを依頼
- [ ] 発表者の場合
  - [ ] 発表資料の作成
  - [ ] 発表資料をnas01の `internal/発表資料/B4輪講/2026/第XX回` へアップロード