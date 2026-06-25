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

import torch
import torch.nn as nn


# 参考：https://qiita.com/snsk871/items/93aba7ad74cace4abc62
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

        # TODO: (max_seq_len, d_model) の位置エンコーディング行列を作成
        # (max_seq_len, d_model) の位置エンコーディング行列を作成
        # すべて 0 で初期化し、後で sin/cos の値を埋め込んでいく
        positional_encoding = torch.zeros(max_seq_len, d_model)

        # 各位置 pos = 0, 1, ..., max_seq_len-1 を列ベクトル (max_seq_len, 1) として作成
        # unsqueeze(1) で形状を (max_seq_len,) → (max_seq_len, 1) に変換し、後でブロードキャスト可能にする
        # 要素ごとの積を取るときにpositionの各要素についてすべてのdiv_termの要素が掛けられて行列になる
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 分母の 10000^(2i/d_model) を計算するための係数
        # 10000はハイパーパラメータで、cosやsinが別の位置で同じ値を取らないようにするためのもの
        # 数値安定性のため exp(log) 形式で書き換える:
        #   1 / 10000^(2i/d_model) = exp(-2i/d_model * log(10000))
        # torch.arange(0, d_model, 2) は 2i (i=0,1,...,d_model/2-1) に対応
        # 元の式は1/（大きな数）の形で2数の差が大きいが、log 10000は小さな数なので安定
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        # 偶数列 (2i 列目) に sin(pos / 10000^(2i/d_model)) を代入
        # position: (max_seq_len, 1), div_term: (d_model/2,) → ブロードキャストで (max_seq_len, d_model/2)
        # 0::2は0からストライド2で偶数列を選択するスライス表現
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # 奇数列 (2i+1 列目) に cos(pos / 10000^(2i/d_model)) を代入
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # バッチ次元を追加して (1, max_seq_len, d_model) にする (forward で加算する際の形状を合わせるため)
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer(
            "pe", positional_encoding
        )  # GPU 上で定数テンソルとして保持

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """生成したエンコーディングの順伝播計算
        Args:
            x(batch, seq_len, d_model): 入力
        Returns:
            output(batch, seq_len, d_model): 位置エンコーディングが加算された出力
        """
        # TODO
        # :x.size(1)で入力系列の長さを取得して、必要な分だけ位置エンコーディング行列からスライスする
        x = x + self.pe[:, : x.size(1), :]
        # ドロップアウトを適用することで、位置情報についての過学習を防ぐ
        output = self.dropout(x)
        return output


# 参考：https://developers.agirobots.com/jp/multi-head-attention/
# attentionはCNNと比較してより広い範囲の文脈を含めた潜在表現を計算でき、RNNと比較して並列計算が可能で高速に学習できるいいとこどり
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

    # QueryとKeyのすべての組み合わせで積をとる(Q@K^T)ことでアテンション重みを獲得する。これは各QueryとKeyの類似度を表す。
    # Valueにアテンション重みを掛けると、類似度が高い単語の値が大きくなる重み付き平均になる
    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
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
        # TODO
        d_k = q.size(-1)

        # q と k の内積を計算してスケーリングする
        # matmul は通常の行列積で @ 演算子と同じ
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float)
        )
        # マスクが与えられている場合は、マスクされた位置のスコアを -inf にして softmax 後に 0 になるようにする
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        # アテンション重みを softmax で計算する
        # 各 query 行に対して、key 方向の合計が 1 になるように正規化するために、dim=-1 を指定して softmax を適用
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # アテンション重みと v を掛け合わせて出力を計算する
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
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
        # PADは空白の意味
        batch_size = query.size(0)

        # TODO
        # query, key, value をそれぞれ (batch, seq_len, d_model) → (batch, seq_len, n_heads, d_k) に変換
        # view関数で行列の形状を変換（viewの直前で転置とかするとバグるらしい）
        # transpose(1, 2) （軸入れ替え）で (batch, n_heads, seq_len, d_k) に変換する
        # 適当に分割しているように見えるが、線形層で意味がきちんと分かれるよう学習されるので問題ない
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # mask を (batch, n_heads, q_len, k_len) にブロードキャスト可能な形に整える
        # - (batch, k_len)            : PAD マスク       → (batch, 1, 1, k_len)
        # - (batch, q_len, k_len)     : 因果 + PAD マスク → (batch, 1, q_len, k_len)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        # 出力を (batch, n_heads, seq_len, d_k) → (batch, seq_len, n_heads, d_k) に変換
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        return output, attn_weights


class FeedForward(nn.Module):
    """位置ごとのフィードフォワードネットワーク (Position-wise Feed-Forward Network, FFN).RNNの対義語といえる.

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN の順伝播計算.

        Args:
            x(batch, seq_len, d_model): 入力
        Returns:
            output(batch, seq_len, d_model): 出力
        """
        # TODO
        # ReLU活性化関数を適用して非線形変換を行う
        output = self.linear2(self.dropout(torch.relu(self.linear1(x))))

        return output


# 参考：https://developers.agirobots.com/jp/multi-head-attention/
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

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """Encoder ブロックの順伝播計算.

        Args:
            x(batch, src_len, d_model): 入力
            src_mask(batch, src_len): 有効トークンが True、PAD が False のマスク
        Returns:
            (batch, src_len, d_model): 出力
        """
        # TODO
        # 自己注意 → Add&Norm → FFN → Add&Norm の順に処理
        residual = x
        x, _ = self.self_attn(x, x, x, src_mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)

        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.norm2(x + residual)

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
        tgt_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decoder ブロックの順伝播計算.

        Args:
            x(batch, tgt_len, d_model): ターゲット埋め込み
            encoder_out(batch, src_len, d_model): encoder の出力
        Returns:
            torch.Tensor(batch, tgt_len, d_model): 出力
        """
        # TODO
        # マスク自己注意 → Add&Norm → クロスアテンション → Add&Norm → FFN → Add&Norm の順に処理
        residual = x
        x, _ = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)

        residual = x
        x, _ = self.cross_attn(x, encoder_out, encoder_out, src_mask)
        x = self.dropout(x)
        x = self.norm2(x + residual)

        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.norm3(x + residual)

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
        # nn.Embeddingは巨大なルックアップテーブルでトークンごとに埋め込みベクトルを持つ
        # 入力[0,1]のワンホットベクトルとの積になるので、誤差逆伝播できる
        # padding_idxを指定することで空白を意味するトークンの埋め込みベクトルをゼロに固定することができる
        # 初期化時にPAD行を0にして、学習中もPAD行の勾配を0にすることができる
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

    # 初期化を上書き
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # normal_ は平均0、標準偏差0.02の正規分布で初期化する関数
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 正規分布で初期化したのでもう一度PAD行を0にする
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

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """Encoder の順伝播計算.

        埋め込み → 位置エンコーディング → N 層の EncoderBlock

        Args:
            src(batch, src_len): ソーストークン列
            src_mask(batch, src_len): 有効トークンであり、パディングの無い位置を示すマスク
        Returns:
            torch.Tensor(batch, src_len, d_model): encoder の出力表現
        """
        # TODO
        # ソースのトークンを埋め込み
        # 論文の3.4節にあるように\sqrt(d_model)を掛ける。おそらく、位置エンコーディングに対してソースの潜在表現の影響を大きくするため
        x = self.src_embedding(src) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float)
        )
        # 位置エンコーディングを加算
        x = self.pos_encoding(x)
        # N 層の EncoderBlock を順に適用
        # self.encoder_blocksにはEncoderBlockのリストが入っているので、for文で順に適用する
        for block in self.encoder_blocks:
            x = block(x, src_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
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
        # TODO
        # ターゲットの埋め込みと位置エンコーディングを計算し、N 層の DecoderBlock を順に適用する
        # encodeと同様に\sqrt(d_model)を掛ける
        x = self.tgt_embedding(tgt) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float)
        )
        # 位置エンコーディングを加算
        x = self.pos_encoding(x)
        # N 層の DecoderBlock を順に適用
        for block in self.decoder_blocks:
            x = block(x, encoder_out, tgt_mask, src_mask)
        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        targets: torch.Tensor = None,
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
        # TODO

        # ソースの有効トークン位置を示すマスクを生成(PAD トークンの位置が False)
        src_mask = self._make_src_mask(src)
        # ターゲットの未来遮蔽マスクと PAD 除外マスクを合成して生成
        tgt_mask = self._make_tgt_mask(tgt)

        # Encoder の順伝播
        encoder_out = self.encode(src, src_mask)
        # Decoder の順伝播
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_mask)
        # 出力を語彙サイズに射影（線形層）
        logits = self.output_proj(decoder_out)

        # 損失計算
        loss = None
        if targets is not None:
            # CrossEntropyLoss は内部で softmax を計算するので、logits をそのまま渡す
            # ignore_index=pad_idx で PAD トークンの損失を無視する
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
            # logits: (batch, tgt_len, tgt_vocab_size) → (batch*tgt_len, tgt_vocab_size)
            # targets: (batch, tgt_len) → (batch*tgt_len)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )  # CrossEntropyLoss は内部で softmax を計算するので、logits をそのまま渡す

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

        # TODO
        # ソースの有効トークン位置を示すマスクを生成
        # 生成タスクでは未来遮断は必要ないので、src_mask のみを作成する
        src_mask = self._make_src_mask(src)
        # Encoder の順伝播
        encoder_out = self.encode(src, src_mask)

        # 生成結果を格納するテンソルを初期化 (BOS トークンで開始)
        # batch_sizeはバッチの大きさではなく数
        batch_size = src.size(0)
        # 文頭(BOS)のトークンですべてのバッチを初期化
        generated = torch.full(
            (batch_size, 1), bos_idx, dtype=torch.long, device=src.device
        )

        for _ in range(max_len):
            # ターゲットの未来遮蔽マスクと PAD 除外マスクを合成して生成
            tgt_mask = self._make_tgt_mask(generated)
            # Decoder の順伝播
            decoder_out = self.decode(generated, encoder_out, tgt_mask, src_mask)
            # 出力を語彙サイズに射影（線形層）
            logits = self.output_proj(decoder_out)
            # 最後のトークンの予測を取得 (batch, tgt_vocab_size)
            next_token_logits = logits[:, -1, :]
            # 貪欲法で次のトークンを選択 (最大値のインデックス)
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(
                1
            )  # (batch, 1)
            # 生成結果に次のトークンを追加
            generated = torch.cat((generated, next_tokens), dim=1)

            # すべてのシーケンスが EOS トークンで終了した場合、ループを終了
            if (next_tokens == eos_idx).all():
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
