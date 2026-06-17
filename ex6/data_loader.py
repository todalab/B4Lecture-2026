"""
Data Loader for Ex6 B4 Lecture - 英日翻訳タスク
データセット: ryo0634/bsd_ja_en (BSD ビジネス対話コーパス)

使用例:
    train_loader, val_loader, src_tok, tgt_tok = create_data_loaders()
"""

import logging
from collections import Counter
from functools import partial
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


class WordTokenizer:
    """単語レベル / 文字レベル トークナイザー

    Args:
        char_level: True のとき文字レベルでトークン化する（日本語向け）

    特殊トークン:
        <PAD> = 0  パディング
        <UNK> = 1  未知語
        <BOS> = 2  文の先頭 (Begin Of Sequence)
        <EOS> = 3  文の末尾 (End Of Sequence)
    """

    def __init__(self, char_level: bool = False):
        self.char_level = char_level
        self.word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.idx2word = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}

    def _tokenize(self, text: str) -> List[str]:
        if self.char_level:
            return list(text)        # 文字レベル: 日本語（スペースなし言語）
        return text.lower().split()  # 単語レベル: 英語

    def build(self, sentences: List[str], max_vocab: int = 8000) -> "WordTokenizer":
        """訓練データから語彙を構築する"""
        counter: Counter = Counter()
        for sent in sentences:
            counter.update(self._tokenize(sent))
        for token, _ in counter.most_common(max_vocab - len(SPECIAL_TOKENS)):
            if token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
        logger.info(
            f"Vocabulary size: {len(self.word2idx)} ({'char' if self.char_level else 'word'}-level)"
        )
        return self

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = True,
        max_len: int = None,
    ) -> List[int]:
        """テキストをトークン ID 列に変換する"""
        tokens = self._tokenize(text)
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        if add_bos:
            ids = [BOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int]) -> str:
        """トークン ID 列をテキストに変換する (特殊トークンを除外)"""
        tokens = []
        for i in ids:
            if i == EOS_IDX:
                break
            if i in (PAD_IDX, BOS_IDX):
                continue
            tokens.append(self.idx2word.get(i, "<UNK>"))
        sep = "" if self.char_level else " "
        return sep.join(tokens)

    def __len__(self) -> int:
        return len(self.word2idx)


class TranslationDataset(Dataset):
    """英日翻訳データセット

    __getitem__ は (src_ids, tgt_in_ids, tgt_out_ids) を返す:
        src_ids:     [..., EOS]        エンコーダ入力 (英語)
        tgt_in_ids:  [BOS, ...]        デコーダ入力   (BOS から始まる日本語)
        tgt_out_ids: [..., EOS]        デコーダ正解   (EOS で終わる日本語)
    """

    def __init__(
        self,
        src_data: List[List[int]],
        tgt_data: List[List[int]],
        max_len: int = 64,
    ):
        pairs = [
            (s, t)
            for s, t in zip(src_data, tgt_data)
            if 1 <= len(s) <= max_len and 1 <= len(t) <= max_len + 1
        ]
        self.src_data = [p[0] for p in pairs]
        self.tgt_data = [p[1] for p in pairs]
        logger.info(f"Dataset: {len(self.src_data)} sentence pairs")

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = self.src_data[idx]  # [..., EOS]
        tgt = self.tgt_data[idx]  # [..., EOS]

        tgt_in = [BOS_IDX] + tgt[:-1]  # [BOS, char1, char2, ...]
        tgt_out = tgt                   # [char1, char2, ..., EOS]

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt_in, dtype=torch.long),
            torch.tensor(tgt_out, dtype=torch.long),
        )


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_idx: int = PAD_IDX,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """バッチ内の系列をパディングして揃える"""
    src_batch, tgt_in_batch, tgt_out_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_in_padded = pad_sequence(tgt_in_batch, batch_first=True, padding_value=pad_idx)
    tgt_out_padded = pad_sequence(tgt_out_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_in_padded, tgt_out_padded


def load_bsd_ja_en(max_samples: int = 100_000) -> Tuple[List[str], List[str]]:
    """ryo0634/bsd_ja_en をロードする

    BSD (Business Scene Dialogue) 英日対訳コーパス。
    初回実行時にダウンロードされ、以降はキャッシュから読み込まれます。

    Returns:
        (en_sentences, ja_sentences) のタプル
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    logger.info("Loading ryo0634/bsd_ja_en ...")
    ds = load_dataset("ryo0634/bsd_ja_en", split="train")

    en_sentences, ja_sentences = [], []
    for item in ds:
        en_sentences.append(item["en_sentence"])
        ja_sentences.append(item["ja_sentence"])
        if len(en_sentences) >= max_samples:
            break

    logger.info(f"Loaded {len(en_sentences)} sentence pairs (en→ja)")
    return en_sentences, ja_sentences


def create_data_loaders(
    max_len: int = 64,
    batch_size: int = 64,
    src_vocab_size: int = 8000,
    tgt_vocab_size: int = 4000,
    train_split: float = 0.95,
    max_samples: int = 100_000,
    data_dir: str = "data",
) -> Tuple[DataLoader, DataLoader, "WordTokenizer", "WordTokenizer"]:
    """英日翻訳用データローダーを作成する

    Args:
        max_len:        最大トークン長 (英語は単語数、日本語は文字数)
        batch_size:     バッチサイズ
        src_vocab_size: 英語語彙サイズ (単語レベル)
        tgt_vocab_size: 日本語語彙サイズ (文字レベル: 4000 程度で十分)
        train_split:    訓練データの割合
        max_samples:    使用する最大サンプル数

    Returns:
        train_loader, val_loader, src_tokenizer (英語), tgt_tokenizer (日本語)
    """
    src_sentences, tgt_sentences = load_bsd_ja_en(max_samples)

    # 訓練・検証分割
    split_idx = int(len(src_sentences) * train_split)
    train_src, val_src = src_sentences[:split_idx], src_sentences[split_idx:]
    train_tgt, val_tgt = tgt_sentences[:split_idx], tgt_sentences[split_idx:]

    # 語彙構築 (訓練データのみ)
    logger.info("Building vocabularies ...")
    src_tokenizer = WordTokenizer(char_level=False).build(train_src, max_vocab=src_vocab_size)
    tgt_tokenizer = WordTokenizer(char_level=True).build(train_tgt, max_vocab=tgt_vocab_size)

    def encode_all(sentences, tokenizer):
        return [tokenizer.encode(s, add_eos=True, max_len=max_len) for s in sentences]

    train_src_ids = encode_all(train_src, src_tokenizer)
    train_tgt_ids = encode_all(train_tgt, tgt_tokenizer)
    val_src_ids = encode_all(val_src, src_tokenizer)
    val_tgt_ids = encode_all(val_tgt, tgt_tokenizer)

    train_dataset = TranslationDataset(train_src_ids, train_tgt_ids, max_len)
    val_dataset = TranslationDataset(val_src_ids, val_tgt_ids, max_len)

    _collate = partial(collate_fn, pad_idx=PAD_IDX)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} pairs ({len(train_loader)} batches)")
    logger.info(f"Val:   {len(val_dataset)} pairs ({len(val_loader)} batches)")
    return train_loader, val_loader, src_tokenizer, tgt_tokenizer
