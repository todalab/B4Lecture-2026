# 第6回B4輪講課題 — Transformer実装とスケーリング

## 概要

> [!NOTE]
> - 本課題では**AIツールの使用を許可**します（コンペ後のため、効率的な学習を重視）
> - ただし、Transformerの核心部分は自分で理解して実装すること
> - PyTorchの基本機能のみを使用し、torch.nn.Transformerなどの高レベルAPIは使用禁止

本課題では、**より柔軟なモデルへ大規模モデルと，それを支える枠組み**をテーマとし、Transformerアーキテクチャの実装とスケーリング法則の検証を行う。

---

## 背景・課題設定

近年、大規模言語モデル（LLM）の発展により、自然言語処理の性能が飛躍的に向上している。
これらのモデルの根幹をなすのがTransformerアーキテクチャである。

本課題では以下を通じて、大規模モデルの基礎を理解する：

### **目標**
1. **Transformerの基本構造を理解し、一から実装する**
2. **モデルサイズと性能の関係（スケーリング法則）を実験的に検証する**
3. **効率的な学習・推論手法を実装する**

---

## データセット

### **言語モデリングタスク**

#### **6.1 小規模データセット（Shakespeare）**
- シェイクスピア作品の文字レベル言語モデリング
- 文字数：約100万文字
- 語彙サイズ：65文字（英数字 + 記号）

#### **6.2 中規模データセット（WikiText-2）**
- Wikipedia記事の単語レベル言語モデリング  
- 語彙サイズ：約33,000語
- 学習データ：約200万語

### **データセット使用方法**

**基本戦略**: 各データセットで別々にモデルを学習し、結果を比較する

1. **Shakespeare（文字レベル）**
   - 小規模なので短時間でTransformerの動作を確認
   - 最初の実装テストに使用
   
2. **WikiText-2（単語レベル）**
   - より実用的な言語モデリングタスク
   - スケーリング実験のメインデータセット

**データの取得**:
データセットは `data/` ディレクトリに含まれています。大きくないのでリポジトリに同梱しました。

```
data/
├── shakespeare.txt      # シェイクスピア作品 (約1MB)
├── wikitext-2/         # WikiText-2データセット
│   ├── train.txt
│   └── valid.txt
```

---

## 課題

### 6-1 基本Transformer実装

#### **6-1-1 Multi-Head Self-Attention**
- Scaled Dot-Product Attentionの実装
- Multi-Head機構の実装
- 位置エンコーディング（Positional Encoding）の実装

#### **6-1-2 Transformer Block**
- Feed-Forward Networkの実装
- Layer Normalizationの実装
- Residual Connectionの実装

#### **6-1-3 言語モデル全体**
- Embedding層 + Positional Encoding
- 複数のTransformer Blockの積み重ね
- 出力層（Language Modeling Head）

### 6-2 スケーリング実験

異なるモデルサイズで性能を比較し、スケーリング法則を検証する：

#### **パラメータ設定**

| モデル | 層数 | 隠れ次元 | ヘッド数 | パラメータ数 |
|--------|------|----------|----------|--------------|
| Tiny   | 4    | 128      | 4        | ~0.5M        |
| Small  | 6    | 256      | 8        | ~2M          |
| Medium | 8    | 512      | 8        | ~8M          |
| Large  | 12   | 768      | 12       | ~25M         |

#### **評価指標**
- **Perplexity**: 言語モデルの性能指標
- **Training Loss**: 学習の進行状況
- **Training Time**: 各モデルの学習時間

### 6-3 効率化手法の実装

#### **6-3-1 Gradient Accumulation**
- バッチサイズを疑似的に大きくする手法
- メモリ効率の改善

#### **6-3-2 Learning Rate Scheduling**
- Warmup + Cosine Decay
- 学習の安定化

---

## 実装仕様

### **提供ファイルと実装箇所**

課題の負担を軽減し、Transformerの核心部分に集中できるよう、以下のファイルを提供します：

#### **提供ファイル**
- `transformer_skeleton.py` - 実装すべき箇所が明示されたスケルトン
- `data_loader.py` - データローダ（完成版）
- `training_utils.py` - 学習ループとロギング（完成版）
- `evaluate.py` - 評価・可視化（完成版）

#### **実装が必要な箇所（transformer_skeleton.py内）**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # ★ 実装箇所 1: 必要なパラメータの初期化
        
    def forward(self, query, key, value, mask=None):
        # ★ 実装箇所 2: Scaled Dot-Product Attention
        # ヒント: Q, K, V行列の変換 → Attention計算 → 出力変換
        pass

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # ★ 実装箇所 3: レイヤーの定義
        
    def forward(self, x, mask=None):
        # ★ 実装箇所 4: Residual Connection + Layer Normalization
        # ヒント: x + Attention(x), x + FFN(x)
        pass

class PositionalEncoding(nn.Module):
    def forward(self, x):
        # ★ 実装箇所 5: Sinusoidal positional encoding
        # ヒント: sin/cosを使った位置エンコーディング
        pass
```

### **学習とロギング**

学習ループとロギングは完成版を提供します（`training_utils.py`）。
適切なログ管理を学ぶため、以下のツールを使用します：

```python
import logging
from tqdm import tqdm
import wandb  # オプション: 実験管理

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # 学習ステップ（完成版で提供）
        
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Training - Average Loss: {avg_loss:.4f}")
    return avg_loss
```

---

## 評価指標

### **Perplexity（困惑度）**

言語モデルの性能を測る標準的な指標：

$$
\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i|w_1,\ldots,w_{i-1})\right)
$$

- 値が小さいほど性能が高い
- 理想的には1.0に近づく

### **スケーリング法則**

モデルサイズ$N$（パラメータ数）と性能の関係：

$$
L(N) = \alpha N^{-\beta} + \gamma
$$

- $\alpha, \beta, \gamma$：実験的に決定する定数
- 対数スケールでプロットして線形関係を確認

---

## 実験設定

### **ハイパーパラメータ**

```python
# 共通設定
BATCH_SIZE = 32
SEQ_LEN = 128
EPOCHS = 20
DROPOUT = 0.1

# 学習率（モデルサイズに応じて調整）
LR_TINY = 3e-4
LR_SMALL = 1e-4
LR_MEDIUM = 5e-5
LR_LARGE = 1e-5
```

### **実験手順**

1. **データ前処理**
   - トークナイゼーション
   - 系列分割（sequence chunking）
   - Train/Validation split

2. **モデル学習**
   - 各サイズのモデルを学習
   - 学習曲線の記録
   - 最良モデルの保存

3. **評価・可視化**
   - Perplexityの比較
   - パラメータ数vs性能のプロット
   - 学習時間の比較

---

## 出力例

### **予想される学習結果（WikiText-2、10エポック）**

```
Model: Tiny (0.8M params)
Epoch 10: Train Loss = 3.85, Val Perplexity = 47.2, Time = 90s, GPU Memory: ~1GB

Model: Small (4.8M params)  
Epoch 10: Train Loss = 3.42, Val Perplexity = 30.4, Time = 240s, GPU Memory: ~2GB

Model: Medium (76M params)
Epoch 10: Train Loss = 3.01, Val Perplexity = 20.3, Time = 720s, GPU Memory: ~6GB

Model: Large (162M params)
Epoch 10: Train Loss = 2.78, Val Perplexity = 16.1, Time = 1200s, GPU Memory: ~12GB
```

**注意**: 
- GPU Memory使用量はバッチサイズ32、シーケンス長128での値
- CPUでも実行可能ですが、Largeモデルは時間がかかります
- メモリ不足の場合はバッチサイズを調整してください

### **スケーリング法則の可視化**

![scaling_law.png](./fig/scaling_law.png)

### **生成例（Large model）**

```
Input: "To be or not to be"
Generated: "To be or not to be, that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune..."
```

---

## 発展課題（任意）

### **6.1 アーキテクチャの改良**
- **RoPE（Rotary Position Embedding）**の実装
- **SwiGLU活性化関数**の実装  
- **Group Query Attention**の実装

### **6.2 効率化手法**
- **Mixed Precision Training**: torch.cuda.ampを使った効率的学習
- **KV-Cache**による推論高速化
- **Gradient Checkpointing**によるメモリ節約
- **Flash Attention**の調査・実装

### **6.3 スケーリング実験の拡張**
- データサイズとモデルサイズの同時スケーリング
- 計算量（FLOPs）とのrelation分析
- より大規模なモデル（100M+ params）での実験

---

## 発表内容（次週）

以下を周囲にわかるように説明すること：

1. **実装したTransformerアーキテクチャの説明**
   - Self-Attentionの仕組み
   - 実装で工夫した点、苦労した点

2. **スケーリング実験の結果**
   - パラメータ数と性能の関係
   - 学習時間・メモリ使用量の比較
   - スケーリング法則の検証結果

3. **考察・今後の展望**
   - 大規模モデルの可能性と課題
   - 実際のLLMとの比較
   - 効率化手法の有効性

---

## 注意事項

- 自分の作業ブランチで課題を行うこと
- プルリクエストをおくる際には**学習結果のグラフと生成例も載せること**
- プルリクエストのコメントには，実行したコマンドと計算環境も記載すること
- 作業前にリポジトリを最新版に更新すること

```bash
$ git checkout main
$ git fetch upstream
$ git merge upstream/main
```

### **計算リソースについて**

- GPUが利用できない場合はCPUでも実行可能
- Large modelは学習時間が長いため、計画的に実行すること
- メモリ不足の場合はbatch sizeやsequence lengthを調整すること

---

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3 paper
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Chinchilla paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation