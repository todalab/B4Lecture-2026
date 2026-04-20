"""
answer/04_model_selection.py  ―  3-4 クラスター数の客観的な決定
K=1..8 で GMM を学習し，AIC・BIC をプロットして最適な K を選ぶ．

Usage:
    python answer/04_model_selection.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
from answer.gmm import GMM

os.makedirs("fig", exist_ok=True)

K_range = list(range(1, 9))

datasets = ["data1", "data2", "data3"]

fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4))

for ax, name in zip(axes, datasets):
    print(f"\n--- {name} ---")
    X = np.loadtxt(f"data/{name}.csv", delimiter=",")

    aics, bics = [], []
    for K in K_range:
        gmm = GMM(K=K, max_iter=300, tol=1e-6, random_state=0)
        gmm.fit(X, verbose=False)
        aics.append(gmm.aic(X))
        bics.append(gmm.bic(X))
        print(f"  K={K}: AIC={aics[-1]:.1f}, BIC={bics[-1]:.1f}")

    best_K_aic = K_range[int(np.argmin(aics))]
    best_K_bic = K_range[int(np.argmin(bics))]
    print(f"  AIC 最小: K={best_K_aic},  BIC 最小: K={best_K_bic}")

    # 個別ファイルに保存
    fig_i, ax_i = plt.subplots(figsize=(5, 4))
    ax_i.plot(K_range, aics, marker="o", label="AIC")
    ax_i.plot(K_range, bics, marker="s", label="BIC")
    ax_i.axvline(
        best_K_aic,
        color="C0",
        linestyle="--",
        alpha=0.7,
        label=f"AIC min K={best_K_aic}",
    )
    ax_i.axvline(
        best_K_bic,
        color="C1",
        linestyle="--",
        alpha=0.7,
        label=f"BIC min K={best_K_bic}",
    )
    ax_i.set_title(f"{name}  AIC BIC")
    ax_i.set_xlabel("number of cluster K")
    ax_i.set_ylabel("Information Criterion")
    ax_i.legend()
    ax_i.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"fig/{name}_aic_bic.png", dpi=150)
    plt.close(fig_i)
    print(f"  保存: fig/{name}_aic_bic.png")

    # まとめ図用
    ax.plot(K_range, aics, marker="o", label="AIC")
    ax.plot(K_range, bics, marker="s", label="BIC")
    ax.axvline(best_K_aic, color="C0", linestyle="--", alpha=0.6)
    ax.axvline(best_K_bic, color="C1", linestyle="--", alpha=0.6)
    ax.set_title(f"{name}\nAIC min K={best_K_aic}, BIC min K={best_K_bic}")
    ax.set_xlabel("K")
    ax.set_ylabel("Information Criterion")
    ax.legend()
    ax.grid(True, alpha=0.4)

plt.suptitle("decision of number of cluster with AIC BIC", y=1.02)
plt.tight_layout()
plt.savefig("fig/all_aic_bic.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n保存: fig/all_aic_bic.png")
