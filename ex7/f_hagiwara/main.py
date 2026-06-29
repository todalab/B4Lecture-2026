# -*- coding: utf-8 -*-
"""VAE の学習・可視化スクリプト。実装済み・変更不要。"""

import argparse
import os

from datetime import datetime
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from libs.Visualize import Visualize
from torch import optim
from torchvision import datasets, transforms
from VAE_coded import VAE


def seed_worker(worker_id):
    """DataLoader の各 worker プロセスにシードを設定する。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int = 42):
    """再現性のためにすべての乱数シードを固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnnの非決定的アルゴリズムを無効化（GPU使用時）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(batch_size: int, train_rate: float, seed: int = 42):
    """MNIST DataLoader を作成して返す。"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    full_train = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

    n_train = int(len(full_train) * train_rate)
    n_val = len(full_train) - n_train

    # 変更: random_split にも明示的に generator を渡す（グローバルシードへの暗黙依存を排除）
    split_g = torch.Generator()
    split_g.manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(
        full_train, [n_train, n_val], generator=split_g
    )

    train_g = torch.Generator()
    train_g.manual_seed(seed)

    # train_loader のみに generator を渡す（shuffle=True のときだけ意味があるため）
    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True,
        batch_size=batch_size, num_workers=2, pin_memory=True,
        worker_init_fn=seed_worker, generator=train_g
    )

    val_loader = torch.utils.data.DataLoader(
        val_set, shuffle=False,
        batch_size=batch_size, num_workers=2, pin_memory=True,
        worker_init_fn=seed_worker
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False,
        batch_size=batch_size, num_workers=2, pin_memory=True,
        worker_init_fn=seed_worker
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, device):
    """1 エポック分の訓練を行い、平均損失を返す。"""
    model.train()
    total = 0.0
    for x, _ in loader:
        x = x.to(device)
        lower_bound, _, _ = model(x)
        loss = -sum(lower_bound) / x.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, device):
    """検証データで損失を計算して返す。"""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            lower_bound, _, _ = model(x)
            total += (-sum(lower_bound) / x.size(0)).item()
    return total / len(loader)


def plot_loss(train_losses, val_losses, path):
    """Loss 推移グラフを保存する。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss  (= −ELBO)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Loss curve → {path}")


def main():
    parser = argparse.ArgumentParser(description="VAE training on MNIST")
    parser.add_argument("--z_dim", type=int, default=2, help="潜在変数の次元数")
    parser.add_argument("--h_dim", type=int, default=400, help="中間層の次元数")
    parser.add_argument("--drop_rate", type=float, default=0.2, help="Dropout 率")
    parser.add_argument("--lr", type=float, default=1e-3, help="学習率")
    parser.add_argument("--epochs", type=int, default=100, help="最大エポック数")
    parser.add_argument("--batch_size", type=int, default=256, help="バッチサイズ")
    parser.add_argument(
        "--train_rate", type=float, default=0.8, help="訓練データの割合"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping の patience"
    )
    args = parser.parse_args()

    # 変更: SEED定数を1箇所で管理（set_seed と get_data_loaders で同じ値を使うため）
    SEED = 42
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("./params", exist_ok=True)
    os.makedirs("./images", exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(
        args.batch_size, args.train_rate, seed=SEED
    )

    model = VAE(args.z_dim, args.h_dim, args.drop_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = (
        f"./params/model_z{args.z_dim}_"
        f"h{args.h_dim}_dr{args.drop_rate}_lr{args.lr}_{timestamp}.pth"
    )
    best_val_loss = float("inf")
    patience_count = 0
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[Epoch {epoch:3d}]  train: {train_loss:9.1f}  val: {val_loss:9.1f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    plot_loss(train_losses, val_losses, f"./images/LC_z{args.z_dim}_"
        f"h{args.h_dim}_dr{args.drop_rate}_lr{args.lr}_{timestamp}.png")

    # 可視化
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    vis = Visualize(args.z_dim, args.h_dim, test_loader, model, device)
    vis.createDirectories()
    vis.reconstruction()
    vis.latent_space()

    if args.z_dim == 2:
        vis.lattice_point()
        vis.walkthrough()
    else:
        print("z_dim != 2: lattice_point / walkthrough をスキップします。")
        print(
            "  ヒント: t-SNE や PCA で潜在空間を2次元に落として可視化してみよう（発展課題）。"
        )


if __name__ == "__main__":
    main()
