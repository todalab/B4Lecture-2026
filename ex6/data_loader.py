"""
Data Loader for Ex6 B4 Lecture - 完成版
Shakespeare と WikiText-2 データセットのローダー
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """テキストデータセット（文字レベル・単語レベル共通）"""

    def __init__(self, data, seq_len):
        """
        Args:
            data: エンコードされたテキストデータ（整数のリスト）
            seq_len: シーケンス長
        """
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # 入力
        y = torch.tensor(chunk[1:], dtype=torch.long)   # ターゲット（1つシフト）
        return x, y


def load_shakespeare_data(data_dir="data"):
    """Shakespeare データセットを読み込み"""
    shakespeare_path = os.path.join(data_dir, "shakespeare.txt")

    if not os.path.exists(shakespeare_path):
        raise FileNotFoundError(
            f"Shakespeare dataset not found at {shakespeare_path}. "
            "Please ensure data/shakespeare.txt exists."
        )

    logger.info(f"Loading Shakespeare dataset from {shakespeare_path}")

    with open(shakespeare_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 文字レベルの語彙を作成
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # テキストをエンコード
    data = [char_to_idx[ch] for ch in text]

    logger.info(f"Shakespeare dataset loaded: {len(text)} characters, {vocab_size} unique characters")

    return data, vocab_size, char_to_idx, idx_to_char


def load_wikitext2_data(data_dir="data"):
    """WikiText-2 データセットを読み込み"""
    train_path = os.path.join(data_dir, "wikitext-2", "train.txt")
    valid_path = os.path.join(data_dir, "wikitext-2", "valid.txt")

    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        raise FileNotFoundError(
            f"WikiText-2 dataset not found at {data_dir}/wikitext-2/. "
            "Please ensure data/wikitext-2/train.txt and valid.txt exist."
        )

    logger.info(f"Loading WikiText-2 dataset from {data_dir}/wikitext-2/")

    # 訓練データを読み込み
    with open(train_path, "r", encoding="utf-8") as f:
        train_text = f.read()

    # 簡単なトークナイゼーション（単語レベル）
    # 実用的には subword tokenizer (BPE, SentencePiece) を使用
    words = train_text.lower().split()

    # 語彙を作成（頻度順にソート）
    from collections import Counter
    word_counts = Counter(words)

    # 頻度が低い単語は <UNK> に置換（語彙サイズを制限）
    vocab_size = 10000  # 語彙サイズを制限
    most_common_words = [word for word, _ in word_counts.most_common(vocab_size - 2)]

    # 特殊トークンを追加
    vocab = ["<PAD>", "<UNK>"] + most_common_words

    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    # テキストをエンコード
    def encode_text(text):
        words = text.lower().split()
        return [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in words]

    train_data = encode_text(train_text)

    # 検証データも処理
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_text = f.read()
    valid_data = encode_text(valid_text)

    logger.info(f"WikiText-2 dataset loaded: {len(train_data)} train tokens, {len(valid_data)} valid tokens, vocab size: {len(vocab)}")

    return train_data, valid_data, len(vocab), word_to_idx, idx_to_word


def create_data_loaders(dataset_name, seq_len=128, batch_size=32, train_split=0.9, data_dir="data"):
    """
    データローダーを作成

    Args:
        dataset_name: "shakespeare" または "wikitext2"
        seq_len: シーケンス長
        batch_size: バッチサイズ
        train_split: 訓練データの割合（shakespeareのみ）
        data_dir: データディレクトリ

    Returns:
        train_loader, val_loader, vocab_size, encode_fn, decode_fn
    """

    if dataset_name == "shakespeare":
        data, vocab_size, char_to_idx, idx_to_char = load_shakespeare_data(data_dir)

        # 訓練・検証分割
        split_idx = int(len(data) * train_split)
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        train_dataset = TextDataset(train_data, seq_len)
        val_dataset = TextDataset(val_data, seq_len)

        def encode_fn(text):
            return [char_to_idx.get(ch, 0) for ch in text]

        def decode_fn(indices):
            return "".join([idx_to_char.get(idx, "?") for idx in indices])

    elif dataset_name == "wikitext2":
        train_data, val_data, vocab_size, word_to_idx, idx_to_word = load_wikitext2_data(data_dir)

        train_dataset = TextDataset(train_data, seq_len)
        val_dataset = TextDataset(val_data, seq_len)

        def encode_fn(text):
            words = text.lower().split()
            return [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in words]

        def decode_fn(indices):
            return " ".join([idx_to_word.get(idx, "<UNK>") for idx in indices])

    else:
        raise ValueError("dataset_name must be 'shakespeare' or 'wikitext2'")

    # データローダーを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    logger.info(f"Created data loaders for {dataset_name}")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Vocab size: {vocab_size}")

    return train_loader, val_loader, vocab_size, encode_fn, decode_fn


def get_sample_batch(dataset_name, seq_len=128, batch_size=4):
    """サンプルバッチを取得（テスト用）"""
    train_loader, val_loader, vocab_size, encode_fn, decode_fn = create_data_loaders(
        dataset_name, seq_len, batch_size
    )

    # 最初のバッチを取得
    for batch_x, batch_y in train_loader:
        logger.info(f"Sample batch shape: {batch_x.shape}")

        # 最初の例を表示
        sample_text = decode_fn(batch_x[0].tolist())
        logger.info(f"Sample text: {sample_text[:100]}...")

        return batch_x, batch_y, vocab_size

    raise RuntimeError("No data available")


if __name__ == "__main__":
    # テスト用のコード
    logging.basicConfig(level=logging.INFO)

    print("=== Data Loader Test ===")

    # Shakespeare テスト
    try:
        print("\n1. Shakespeare Dataset Test:")
        batch_x, batch_y, vocab_size = get_sample_batch("shakespeare", seq_len=64, batch_size=2)
        print(f"Success! Vocab size: {vocab_size}")
    except Exception as e:
        print(f"Shakespeare test failed: {e}")

    # WikiText-2 テスト
    try:
        print("\n2. WikiText-2 Dataset Test:")
        batch_x, batch_y, vocab_size = get_sample_batch("wikitext2", seq_len=64, batch_size=2)
        print(f"Success! Vocab size: {vocab_size}")
    except Exception as e:
        print(f"WikiText-2 test failed: {e}")

    print("\nData loader test completed!")