"""
answer/01_scatter.py  ―  3-1 データの確認
各 CSV ファイルを読み込み，散布図を fig/ に保存する．

Usage:
    python answer/01_scatter.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

os.makedirs("fig", exist_ok=True)

datasets = ["data1", "data2", "data3"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, name in zip(axes, datasets):
    X = np.loadtxt(f"data/{name}.csv", delimiter=",")
    ax.scatter(X[:, 0], X[:, 1], s=15, alpha=0.5)
    ax.set_title(f"{name}  (N={len(X)})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")

plt.suptitle("scatter for data1~3", y=1.02)
plt.tight_layout()
plt.savefig("fig/all_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("保存: fig/all_scatter.png")

# 個別保存
for name in datasets:
    X = np.loadtxt(f"data/{name}.csv", delimiter=",")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X[:, 0], X[:, 1], s=15, alpha=0.5)
    ax.set_title(f"scatter for {name}  (N={len(X)})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"fig/{name}_scatter.png", dpi=150)
    plt.close()
    print(f"保存: fig/{name}_scatter.png")
