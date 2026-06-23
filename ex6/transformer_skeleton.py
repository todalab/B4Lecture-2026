"""
Ex6 B4講義 - 翻訳タスク用 Transformer

このファイルは、Transformer モデルの実装を行うためのコードです.
以下のクラスのTODOと記載されている箇所を実装してください:
- PositionalEncoding
    - __init__
    - forward
- MultiHeadAttention
    - scaled_dot_product_attention
    - forward
- FeedForward
    - forward
- EncoderBlock
    - forward
- DecoderBlock
    - forward
- TranslationModel
    - encode
    - decode
    - forward
    - generate
"""

from typing import Optional

import torch
import torch.nn as nn
from numpy import cos, fromfunction, sin, where
from torch import Tensor, cat, inf, softmax, tensor


class PositionalEncoding(nn.Module):
    """位置エンコーディング (Sinusoidal Positional Encoding).

    Args:
        d_model(int): 埋め込み次元
        dropout(float): ドロップアウト率
        max_seq_len(int): 最大系列長

    Functions:
        __init__(d_model, dropout, max_seq_len): 初期化コンストラクタ
        forward(x): 順伝播計算

    Returns:
        torch.Tensor: 位置エンコーディング行列 (1, max_seq_len, d_model)

    （ヒント）位置エンコーディングの計算式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # (max_seq_len, d_model) の位置エンコーディング行列を作成
        positional_encoding = fromfunction(
            lambda pos, n: where(
                n % 2 == 0,
                sin(pos / 10000 ** (n / d_model)),
                cos(pos / 10000 ** ((n - 1) / d_model)),
            ),
            (max_seq_len, d_model),
        )
        positional_encoding = tensor(positional_encoding, dtype=torch.float32)

        self.pe: Tensor
        self.register_buffer(
            "pe", positional_encoding.unsqueeze(0)
        )  # GPU 上で定数テンソルとして保持

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """生成したエンコーディングの順伝播計算
        Args:
            x(batch, seq_len, d_model): 入力
        Returns:
            output(batch, seq_len, d_model): 位置エンコーディングが加算された出力
        """
        m = x.size(1)
        output = self.dropout(x + self.pe[:, :m])
        return output


class MultiHeadAttention(nn.Module):
    """マルチヘッドアテンション (Multi-Head Attention).

    Args:
        d_model(int): 埋め込み次元
        n_heads(int): ヘッド数
        dropout(float): ドロップアウト率

    Functions:
        __init__(d_model, n_heads, dropout): 初期化コンストラクタ
        scaled_dot_product_attention(q, k, v, mask): スケールドドットプロダクトアテンションの計算
        forward(query, key, value, mask=None): アテンションの順伝播計算

    (ヒント) Multi-Head Attention の計算:
        self-attention  の場合: query = key = value = x
        cross-attention の場合: query = x, key = value = (encoderの出力)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model は n_heads で割り切れる必要があります"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[Tensor] = None,
    ):
        """スケールドドットプロダクトアテンションの計算
        Args:
            q(batch, heads, seq_len, d_k): クエリ
            k(batch, heads, seq_len, d_k): キー
            v(batch, heads, seq_len, d_k): バリュー
            mask(batch, 1, seq_len, seq_len): アテンションスコアのマスク
        Returns:
            output(batch, heads, seq_len, d_k): アテンション出力
            attn_weights(batch, heads, seq_len, seq_len): アテンション重み
        """
        scores: Tensor = q @ k.transpose(2, 3) / self.d_k**0.5
        if mask is not None:
            scores = scores.masked_fill(~mask, -inf)
        attention_weights = softmax(scores, dim=-1)
        output = attention_weights @ v

        return output, attention_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[Tensor] = None,
    ):
        """アテンションの順伝播計算
        Args:
            query: (batch, q_len, d_model)
            key:   (batch, k_len, d_model)
            value: (batch, k_len, d_model)
            mask:  PAD 位置が False のマスク (batch, k_len)
                   または未来トークンが False の因果マスク (batch, q_len, k_len)
        Returns:
            output(batch, q_len, d_model): アテンション出力
            attn_weights(batch, heads, q_len, k_len): アテンション重み
        """
        batch_size = query.size(0)

        def split_heads(x: Tensor) -> Tensor:
            # split (d_model) into (n_heads, d_k) and transpose
            return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        q, k, v = map(split_heads, (self.w_q(query), self.w_k(key), self.w_v(value)))

        # Correct mask dimensionality
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        concatenated = output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        output = self.w_o(concatenated)

        return output, attn_weights


class FeedForward(nn.Module):
    """位置ごとのフィードフォワードネットワーク (Position-wise Feed-Forward Network, FFN).

    Args:
        d_model(int): 埋め込み次元
        d_ff(int): 中間層の次元 (通常 d_model の 4 倍)
        dropout(float): ドロップアウト率

    Functions:
        __init__(d_model, d_ff, dropout): 初期化コンストラクタ
        forward(x): 順伝播計算

    Returns:
        torch.Tensor(batch, seq_len, d_model): 出力

    （ヒント）FFN の計算式:
        FFN(x) = linear2(dropout(ReLU(linear1(x))))
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.dropout,
            self.linear2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN の順伝播計算.

        Args:
            x(batch, seq_len, d_model): 入力
        Returns:
            output(batch, seq_len, d_model): 出力
        """
        return self.layer(x)


class EncoderBlock(nn.Module):
    """Encoder の 1 ブロック.

    Args:
        d_model(int): 埋め込み次元
        n_heads(int): アテンションのヘッド数
        d_ff(int): FFN の中間層次元
        dropout(float): ドロップアウト率

    Functions:
        __init__(d_model, n_heads, d_ff, dropout): 初期化コンストラクタ
        forward(x, src_mask): 順伝播計算

    Returns:
        torch.Tensor(batch, src_len, d_model): 出力

    （ヒント）処理の流れ:
        x → 自己注意 → Add&Norm → FFN → Add&Norm → 出力
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_mask: Optional[Tensor] = None
    ) -> torch.Tensor:
        """Encoder ブロックの順伝播計算.

        Args:
            x(batch, src_len, d_model): 入力
            src_mask(batch, src_len): 有効トークンが True、PAD が False のマスク
        Returns:
            (batch, src_len, d_model): 出力
        """

        y, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(y))

        y = self.feed_forward(x)
        x = self.norm2(x + self.dropout(y))

        return x


class DecoderBlock(nn.Module):
    """Decoder の 1 ブロック.

    Args:
        d_model(int): 埋め込み次元
        n_heads(int): アテンションのヘッド数
        d_ff(int): FFN の中間層次元
        dropout(float): ドロップアウト率

    Functions:
        __init__(d_model, n_heads, d_ff, dropout): 初期化コンストラクタ
        forward(x, encoder_out, tgt_mask, src_mask): 順伝播計算

    Returns:
        torch.Tensor(batch, tgt_len, d_model): 出力

    （ヒント）クロスアテンションの query / key / value:
        query = decoder 側の表現 (翻訳先言語)
        key   = encoder 出力     (翻訳元言語)
        value = encoder 出力     (翻訳元言語)

    （ヒント）処理の流れ:
        x → マスク自己注意 → Add&Norm → クロスアテンション → Add&Norm → FFN → Add&Norm → 出力
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> torch.Tensor:
        """Decoder ブロックの順伝播計算.

        Args:
            x(batch, tgt_len, d_model): ターゲット埋め込み
            encoder_out(batch, src_len, d_model): encoder の出力
        Returns:
            torch.Tensor(batch, tgt_len, d_model): 出力
        """

        y, _ = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(y))

        y, _ = self.cross_attn(x, encoder_out, encoder_out, src_mask)
        x = self.norm2(x + self.dropout(y))

        y = self.feed_forward(x)
        x = self.norm3(x + self.dropout(y))

        return x


class TranslationModel(nn.Module):
    """機械翻訳用 Transformer.

    英語 (src) を Encoder で符号化し、Decoder で日本語 (tgt) を逐次生成する.

    Args:
        src_vocab_size(int): ソース語彙サイズ (英語)
        tgt_vocab_size(int): ターゲット語彙サイズ (日本語)
        d_model(int): 埋め込み次元
        n_heads(int): アテンションのヘッド数
        n_encoder_layers(int): Encoder ブロック数
        n_decoder_layers(int): Decoder ブロック数
        d_ff(int): FFN の中間層次元
        max_seq_len(int): 最大系列長
        dropout(float): ドロップアウト率
        pad_idx(int): パディングトークンのインデックス

    Functions:
        _init_weights(module): 重み初期化
        _make_src_mask(src): ソースの有効トークン位置を示すマスクを生成
        _make_tgt_mask(tgt): ターゲットの未来遮蔽マスクと PAD 除外マスクを合成して生成
        encode(src, src_mask): Encoder の順伝播
        decode(tgt, encoder_out, tgt_mask, src_mask): Decoder の順伝播
        forward(src, tgt, targets): モデル全体の順伝播 (訓練時)
        generate(src, bos_idx, eos_idx, max_len): 推論時の文生成 (Greedy)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_heads: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # ソース・ターゲットで別々の埋め込み行列
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_encoder_layers)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_decoder_layers)
            ]
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """ソースの有効トークン位置を示すマスクを生成する.

        バッチ処理で系列長を揃えるために挿入した PAD トークンを、
        アテンション計算から除外するために使用する。

        Args:
            src(batch, src_len): ソーストークン列
        Returns:
            torch.Tensor(batch, src_len): 有効トークンが True、PAD が False のブール型テンソル
        """
        return src != self.pad_idx

    def _make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """ターゲットの未来遮蔽マスクと PAD 除外マスクを合成して生成する.

        デコーダが未来のトークンを参照しないよう、下三角行列で制限する。
        さらに系列長を揃えるために挿入した PAD トークンの位置も False にする。

        Args:
            tgt(batch, tgt_len): ターゲットトークン列
        Returns:
            torch.Tensor(batch, tgt_len, tgt_len): 参照可能な位置が True、未来・PAD が False のマスク
        """
        tgt_len = tgt.size(1)
        causal_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device)
        ).bool()  # (tgt_len, tgt_len)
        pad_mask = (tgt != self.pad_idx).unsqueeze(1)  # (batch, 1, tgt_len)
        return causal_mask.unsqueeze(0) & pad_mask  # (batch, tgt_len, tgt_len)

    def encode(
        self, src: torch.Tensor, src_mask: Optional[Tensor] = None
    ) -> torch.Tensor:
        """Encoder の順伝播計算.

        埋め込み → 位置エンコーディング → N 層の EncoderBlock

        Args:
            src(batch, src_len): ソーストークン列
            src_mask(batch, src_len): 有効トークンであり、パディングの無い位置を示すマスク
        Returns:
            torch.Tensor(batch, src_len, d_model): encoder の出力表現
        """
        x = self.pos_encoding(self.src_embedding(src) * self.d_model**0.5)
        for block in self.encoder_blocks:
            x = block(x, src_mask)

        return x

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> torch.Tensor:
        """Decoder の順伝播計算.

        埋め込み → 位置エンコーディング → N 層の DecoderBlock

        Args:
            tgt(batch, tgt_len): ターゲットトークン列 (BOS から始まる)
            encoder_out(batch, src_len, d_model): encoder の出力
            tgt_mask(batch, tgt_len, tgt_len): 未来遮蔽と PAD 除外を合わせたマスク
            src_mask(batch, src_len): 有効トークンであり、パディングの無い位置を示すマスク
        Returns:
            torch.Tensor(batch, tgt_len, d_model): decoder の出力表現
        """

        x = self.pos_encoding(self.tgt_embedding(tgt) * self.d_model**0.5)
        for block in self.decoder_blocks:
            x = block(x, encoder_out, tgt_mask, src_mask)

        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        targets: Optional[Tensor] = None,
    ):
        """
        Args:
            src(batch, src_len): ソーストークン列
            tgt(batch, tgt_len): ターゲット入力  (BOS から始まる)
            targets(batch, tgt_len): ターゲット正解  (EOS で終わる, None なら損失計算なし)

        Returns:
            logits(batch, tgt_len, tgt_vocab_size):
            loss(float): 損失値
        """
        src_mask = self._make_src_mask(src)
        tgt_mask = self._make_tgt_mask(tgt)

        encoded = self.encode(src, src_mask)
        decoded = self.decode(tgt, encoded, tgt_mask, src_mask)

        logits = self.output_proj(decoded)
        loss = None
        if targets is not None:
            crossEntropyLoss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
            loss = crossEntropyLoss(logits.transpose(1, 2), targets)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 100,
    ) -> torch.Tensor:
        """Greedy デコーディング（貪欲法による逐次生成）

        Args:
            src:     (batch, src_len)
            bos_idx: BOS トークンのインデックス
            eos_idx: EOS トークンのインデックス
            max_len: 最大生成長

        Returns:
            generated(batch, generated_len): BOS を含む
        """
        self.eval()

        batch_size = src.size(0)
        src_mask = self._make_src_mask(src)
        encoded = self.encode(src, src_mask)
        generated = torch.full((batch_size, 1), bos_idx, device=src.device)

        for _ in range(max_len):
            tgt_mask = self._make_tgt_mask(generated)
            decoded = self.decode(generated, encoded, tgt_mask, src_mask)
            logits = self.output_proj(decoded)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = cat([generated, next_token], dim=1)
            if (next_token == eos_idx).all():
                break

        return generated


def get_model_config(model_size: str) -> dict:
    configs = {
        "tiny": {
            "n_encoder_layers": 2,
            "n_decoder_layers": 2,
            "d_model": 128,
            "n_heads": 4,
            "d_ff": 512,
        },
        "small": {
            "n_encoder_layers": 3,
            "n_decoder_layers": 3,
            "d_model": 256,
            "n_heads": 8,
            "d_ff": 1024,
        },
        "medium": {
            "n_encoder_layers": 4,
            "n_decoder_layers": 4,
            "d_model": 256,
            "n_heads": 8,
            "d_ff": 1024,
        },
        "large": {
            "n_encoder_layers": 6,
            "n_decoder_layers": 6,
            "d_model": 512,
            "n_heads": 8,
            "d_ff": 2048,
        },
    }
    return configs[model_size]
