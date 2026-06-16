"""
Transformer Skeleton for Ex6 B4 Lecture
実装すべき箇所は ★ で示されています

学生は以下の箇所を実装する必要があります：
1. MultiHeadAttention の初期化と forward
2. TransformerBlock の初期化と forward
3. PositionalEncoding の実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置エンコーディング - Sinusoidal positional encoding を実装してください"""

    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # ★ 実装箇所 5: Sinusoidal positional encoding
        # ヒント:
        # - pe = torch.zeros(max_seq_len, d_model) でエンコーディング行列を作成
        # - position = torch.arange(0, max_seq_len).unsqueeze(1) で位置インデックス
        # - div_term で周波数項を計算
        # - 偶数インデックスにsin、奇数インデックスにcosを適用
        # - register_buffer でpe を登録

        # TODO: ここに実装
        pass

    def forward(self, x):
        # ★ 実装箇所 5-2: エンコーディングを入力に加算
        # ヒント: x + self.pe[:x.size(1), :] のようにして加算

        # TODO: ここに実装
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention機構"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        # ★ 実装箇所 1: 必要なパラメータの初期化
        # ヒント:
        # - d_model % n_heads == 0 をチェック
        # - self.d_k = d_model // n_heads で各ヘッドの次元数
        # - Query, Key, Value用のLinear層を定義（w_q, w_k, w_v）
        # - 出力用のLinear層を定義（w_o）
        # - Dropout層を定義

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        # TODO: self.d_k = ?

        # TODO: Linear layers を定義
        # self.w_q = nn.Linear(?, ?)
        # self.w_k = nn.Linear(?, ?)
        # self.w_v = nn.Linear(?, ?)
        # self.w_o = nn.Linear(?, ?)

        # TODO: Dropout layer
        # self.dropout = nn.Dropout(?)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Scaled Dot-Product Attention"""
        # ★ 実装箇所 2-1: Attention の計算
        # ヒント:
        # - scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # - マスクがある場合は大きな負の値で置換
        # - softmax で正規化
        # - v と掛け合わせて出力

        d_k = q.size(-1)

        # TODO: Attention scores の計算
        # scores = ?

        # TODO: マスク適用（もしある場合）
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)

        # TODO: Softmax + Dropout
        # attention_weights = F.softmax(scores, dim=-1)
        # attention_weights = self.dropout(attention_weights)

        # TODO: Attentionの適用
        # output = torch.matmul(attention_weights, v)

        # return output, attention_weights

        # 暫定的に入力をそのまま返す（実装後に削除）
        return q, None

    def forward(self, query, key, value, mask=None):
        """Multi-Head Attention の forward pass"""
        # ★ 実装箇所 2: Multi-Head機構の実装
        # ヒント:
        # 1. Linear変換でQ,K,Vを作成
        # 2. (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k) に変形
        # 3. scaled_dot_product_attention を呼び出し
        # 4. (batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model) に戻す
        # 5. 最終Linear層を適用

        batch_size = query.size(0)
        seq_len = query.size(1)

        # TODO: Linear変換
        # q = self.w_q(query)  # (batch, seq_len, d_model)
        # k = self.w_k(key)
        # v = self.w_v(value)

        # TODO: Multi-head用に変形
        # q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # → (batch, n_heads, seq_len, d_k)

        # TODO: Attention適用
        # attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # TODO: 形状を元に戻す
        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # TODO: 最終Linear層
        # output = self.w_o(attn_output)

        # 暫定的に入力をそのまま返す（実装後に削除）
        return query, None


class FeedForward(nn.Module):
    """Position-wise Feed Forward Network（完成版）"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """単一のTransformerブロック"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        # ★ 実装箇所 3: レイヤーの定義
        # ヒント:
        # - MultiHeadAttention インスタンスを作成
        # - FeedForward インスタンスを作成（完成版があります）
        # - LayerNormalization を2つ作成（norm1, norm2）
        # - Dropout を作成

        # TODO: 各レイヤーを定義
        # self.attention = MultiHeadAttention(?, ?, ?)
        # self.feed_forward = FeedForward(?, ?, ?)
        # self.norm1 = nn.LayerNorm(?)
        # self.norm2 = nn.LayerNorm(?)
        # self.dropout = nn.Dropout(?)

    def forward(self, x, mask=None):
        """TransformerBlock の forward pass"""
        # ★ 実装箇所 4: Residual Connection + Layer Normalization
        # ヒント:
        # 1. Self-attention: norm(x + attention(x))
        # 2. Feed-forward: norm(x + feed_forward(x))
        # Pre-normalization と Post-normalization の2つの方式があります
        # ここではPost-normalization（原論文）を実装してください

        # TODO: Self-attention with residual connection
        # attn_output, _ = self.attention(x, x, x, mask)
        # x = self.norm1(x + self.dropout(attn_output))

        # TODO: Feed-forward with residual connection
        # ff_output = self.feed_forward(x)
        # x = self.norm2(x + self.dropout(ff_output))

        # 暫定的に入力をそのまま返す（実装後に削除）
        return x


class LanguageModel(nn.Module):
    """完全なTransformer言語モデル（一部完成版）"""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embedding層（完成版）
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding（要実装）
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer Blocks（要実装）
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 最終レイヤー（完成版）
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 重み初期化（完成版）
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """重み初期化（完成版）"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_causal_mask(self, seq_len):
        """因果マスクの作成（完成版）"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(self, x, targets=None):
        """言語モデルの forward pass"""
        seq_len = x.size(1)

        # Token embeddings
        x = self.embedding(x) * math.sqrt(self.d_model)  # スケーリング

        # Positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 因果マスクを作成
        mask = self.create_causal_mask(seq_len).to(x.device)

        # Transformer blocks を適用
        for block in self.transformer_blocks:
            x = block(x, mask)

        # 最終レイヤー
        x = self.ln_f(x)
        logits = self.head(x)

        # 損失計算（学習時のみ）
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


def get_model_config(model_size):
    """モデルサイズ設定（完成版）"""
    configs = {
        'tiny': {'n_layers': 4, 'd_model': 128, 'n_heads': 4, 'd_ff': 512},
        'small': {'n_layers': 6, 'd_model': 256, 'n_heads': 8, 'd_ff': 1024},
        'medium': {'n_layers': 8, 'd_model': 512, 'n_heads': 8, 'd_ff': 2048},
        'large': {'n_layers': 12, 'd_model': 768, 'n_heads': 12, 'd_ff': 3072}
    }
    return configs[model_size]


if __name__ == "__main__":
    # 基本テスト用のコード
    print("Transformer Skeleton テスト")

    config = get_model_config('tiny')
    model = LanguageModel(
        vocab_size=1000,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=128
    )

    # テスト用データ
    x = torch.randint(0, 1000, (2, 32))  # (batch_size=2, seq_len=32)

    print(f"Input shape: {x.shape}")
    try:
        logits, loss = model(x, x)  # 自分自身をターゲットとして使用
        print(f"Output shape: {logits.shape}")
        print(f"Model test successful!")
    except Exception as e:
        print(f"実装が必要です: {e}")
        print("★マークの箇所を実装してから再実行してください")