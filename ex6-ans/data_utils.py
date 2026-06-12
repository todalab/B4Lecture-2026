"""
Data utilities for Transformer training
Handles Shakespeare and WikiText-2 datasets
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import requests
import zipfile
import pickle


class CharDataset(Dataset):
    """Character-level dataset for Shakespeare data"""
    
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class TokenDataset(Dataset):
    """Token-level dataset for WikiText-2 data"""
    
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def download_shakespeare():
    """Download and prepare Shakespeare dataset"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("shakespeare.txt"):
        print("Downloading Shakespeare dataset...")
        response = requests.get(url)
        with open("shakespeare.txt", "w") as f:
            f.write(response.text)
        print("Download complete!")
    
    with open("shakespeare.txt", "r") as f:
        text = f.read()
    
    # Create character mappings
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Encode the text
    data = [char_to_idx[ch] for ch in text]
    
    # Save mappings
    with open("char_mappings.pkl", "wb") as f:
        pickle.dump((char_to_idx, idx_to_char), f)
    
    return data, vocab_size, char_to_idx, idx_to_char


def download_wikitext2():
    """Download and prepare WikiText-2 dataset"""
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    
    if not os.path.exists("wikitext-2"):
        print("Downloading WikiText-2 dataset...")
        response = requests.get(url)
        with open("wikitext-2-v1.zip", "wb") as f:
            f.write(response.content)
        
        with zipfile.ZipFile("wikitext-2-v1.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        print("Download complete!")
    
    # Read train data
    with open("wikitext-2/wiki.train.tokens", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Simple tokenization (split by whitespace)
    tokens = text.split()
    
    # Create vocabulary
    vocab = sorted(list(set(tokens)))
    vocab_size = len(vocab)
    
    token_to_idx = {token: i for i, token in enumerate(vocab)}
    idx_to_token = {i: token for i, token in enumerate(vocab)}
    
    # Encode the tokens
    data = [token_to_idx[token] for token in tokens]
    
    # Save mappings
    with open("token_mappings.pkl", "wb") as f:
        pickle.dump((token_to_idx, idx_to_token), f)
    
    return data, vocab_size, token_to_idx, idx_to_token


def get_shakespeare_data(seq_len=128, train_split=0.9):
    """Get Shakespeare character-level data loaders"""
    data, vocab_size, char_to_idx, idx_to_char = download_shakespeare()
    
    # Train/validation split
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    train_dataset = CharDataset(train_data, seq_len)
    val_dataset = CharDataset(val_data, seq_len)
    
    return train_dataset, val_dataset, vocab_size, char_to_idx, idx_to_char


def get_wikitext2_data(seq_len=128, train_split=0.9):
    """Get WikiText-2 token-level data loaders"""
    data, vocab_size, token_to_idx, idx_to_token = download_wikitext2()
    
    # Train/validation split
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    train_dataset = TokenDataset(train_data, seq_len)
    val_dataset = TokenDataset(val_data, seq_len)
    
    return train_dataset, val_dataset, vocab_size, token_to_idx, idx_to_token


def create_data_loaders(dataset_name, seq_len=128, batch_size=32, train_split=0.9):
    """Create data loaders for specified dataset"""
    if dataset_name == "shakespeare":
        train_dataset, val_dataset, vocab_size, mapping1, mapping2 = get_shakespeare_data(seq_len, train_split)
    elif dataset_name == "wikitext2":
        train_dataset, val_dataset, vocab_size, mapping1, mapping2 = get_wikitext2_data(seq_len, train_split)
    else:
        raise ValueError("Dataset must be 'shakespeare' or 'wikitext2'")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, vocab_size, mapping1, mapping2


def decode_text(indices, mapping, dataset_name):
    """Decode indices back to text"""
    if dataset_name == "shakespeare":
        idx_to_char = mapping
        return "".join([idx_to_char[idx] for idx in indices])
    elif dataset_name == "wikitext2":
        idx_to_token = mapping
        return " ".join([idx_to_token[idx] for idx in indices])
    else:
        raise ValueError("Dataset must be 'shakespeare' or 'wikitext2'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download datasets")
    parser.add_argument("--test", action="store_true", help="Test data loading")
    args = parser.parse_args()
    
    if args.download:
        print("Downloading Shakespeare...")
        download_shakespeare()
        print("Downloading WikiText-2...")
        download_wikitext2()
    
    if args.test:
        print("Testing Shakespeare data loading...")
        train_loader, val_loader, vocab_size, char_to_idx, idx_to_char = create_data_loaders("shakespeare", batch_size=4)
        print(f"Vocab size: {vocab_size}")
        
        for batch_idx, (x, y) in enumerate(train_loader):
            print(f"Batch shape: {x.shape}")
            print(f"Sample text: {decode_text(x[0].tolist(), idx_to_char, 'shakespeare')[:100]}")
            break
        
        print("\nTesting WikiText-2 data loading...")
        train_loader, val_loader, vocab_size, token_to_idx, idx_to_token = create_data_loaders("wikitext2", batch_size=4)
        print(f"Vocab size: {vocab_size}")
        
        for batch_idx, (x, y) in enumerate(train_loader):
            print(f"Batch shape: {x.shape}")
            print(f"Sample text: {decode_text(x[0].tolist()[:20], idx_to_token, 'wikitext2')}")
            break