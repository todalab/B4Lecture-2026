# -*- coding: utf-8 -*-
"""VAE の学習・可視化スクリプト。実装済み・変更不要。"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from libs.Visualize import Visualize
from torch import flatten, optim
from torchvision import datasets, transforms
from VAE_skeleton import VAE


def get_data_loaders(batch_size: int, train_rate: float):
    """MNIST DataLoader を作成して返す。"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(flatten),
        ]
    )
    full_train = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

    n_train = int(len(full_train) * train_rate)
    n_val = len(full_train) - n_train
    train_set, val_set = torch.utils.data.random_split(full_train, [n_train, n_val])

    kwargs = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **kwargs)
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

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("./params", exist_ok=True)
    os.makedirs("./images", exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(
        args.batch_size, args.train_rate
    )

    model = VAE(args.z_dim, args.h_dim, args.drop_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model_path = f"./params/model_z{args.z_dim}_h{args.h_dim}.pth"
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

    plot_loss(train_losses, val_losses, "./images/loss_curve.png")

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
