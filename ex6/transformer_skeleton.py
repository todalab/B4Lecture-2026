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

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention機構"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Scaled Dot-Product Attention"""
        d_k = q.size(-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)

        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        """Multi-Head Attention の forward pass"""
        batch_size = query.size(0)
        seq_len = query.size(1)

        q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        output = self.w_o(attn_output)

        return output, attn_weights


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

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """TransformerBlock の forward pass"""
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

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

    def generate(self, context, max_tokens, temperature=1.0, top_k=None):
        """テキストを自己回帰的に生成"""
        self.eval()
        generated = context.clone()

        for _ in range(max_tokens):
            with torch.no_grad():
                logits, _ = self.forward(generated)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                generated = torch.cat([generated, next_token], dim=1)

                if generated.size(1) >= self.max_seq_len:
                    break

        return generated


def get_model_config(model_size):
    """モデルサイズ設定（完成版）"""
    configs = {
        'tiny': {'n_layers': 4, 'd_model': 128, 'n_heads': 4, 'd_ff': 512},
        'small': {'n_layers': 6, 'd_model': 256, 'n_heads': 8, 'd_ff': 1024},
        'medium': {'n_layers': 8, 'd_model': 512, 'n_heads': 8, 'd_ff': 2048},
        'large': {'n_layers': 12, 'd_model': 768, 'n_heads': 12, 'd_ff': 3072}
    }
    return configs[model_size]


def count_parameters(model):
    """モデルのパラメータ数をカウント"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Transformer 実装テスト")

    for size in ['tiny', 'small']:
        config = get_model_config(size)
        model = LanguageModel(
            vocab_size=1000,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_seq_len=128
        )

        param_count = count_parameters(model)
        print(f"{size.title()} model: {param_count:,} parameters ({param_count/1e6:.2f}M)")

        x = torch.randint(0, 1000, (2, 32))
        logits, loss = model(x, x)
        print(f"  Output shape: {logits.shape}, Loss: {loss.item():.4f}")
