"""
answer/03_clustering.py  ―  3-3 データへの適用
各データに GMM を適用し，対数尤度の収束曲線とクラスタリング結果を fig/ に保存する．

Usage:
    python answer/03_clustering.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
from answer.gmm import GMM
from matplotlib.patches import Ellipse

os.makedirs("fig", exist_ok=True)


# ============================================================
# 可視化ユーティリティ
# ============================================================


def plot_ellipse(ax, mu, Sigma, color, n_std=(1, 2), alpha=0.25):
    """共分散行列から信頼楕円（1σ, 2σ）を描く．"""
    vals, vecs = np.linalg.eigh(Sigma)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    for n in n_std:
        width, height = 2 * n * np.sqrt(np.maximum(vals, 0))
        ell = Ellipse(
            xy=mu,
            width=width,
            height=height,
            angle=angle,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(ell)


def plot_gmm_result(X, gmm, title, filename):
    """クラスタリング結果（散布図 + 等高線 + 平均）を保存する．"""
    K = gmm.K
    labels = np.argmax(gmm.r_, axis=1)
    colors = plt.cm.tab10(np.linspace(0, 0.9, K))

    fig, ax = plt.subplots(figsize=(6, 6))
    for k in range(K):
        mask = labels == k
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=20,
            color=colors[k],
            alpha=0.6,
            label=f"Cluster {k + 1}",
        )
        plot_ellipse(ax, gmm.mu[k], gmm.Sigma[k], color=colors[k])
        ax.scatter(
            *gmm.mu[k],
            s=200,
            color=colors[k],
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
            marker="*",
        )

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"保存: {filename}")


def plot_log_likelihood(log_likelihoods, title, filename):
    """対数尤度の収束曲線を保存する．"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        range(1, len(log_likelihoods) + 1),
        log_likelihoods,
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    ax.set_title(title)
    ax.set_xlabel("iter cnt")
    ax.set_ylabel("log likelihood")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"保存: {filename}")


# ============================================================
# メイン
# ============================================================

# データごとに目安のクラスター数を指定
configs = {
    "data1": 3,  # 3クラスター
    "data2": 4,  # 4クラスター（2つが近接）
    "data3": 2,  # 2クラスター（細長い楕円形状）
}

for name, K in configs.items():
    print(f"\n--- {name} (K={K}) ---")
    X = np.loadtxt(f"data/{name}.csv", delimiter=",")

    gmm = GMM(K=K, max_iter=300, tol=1e-6, random_state=0)
    gmm.fit(X)

    plot_log_likelihood(
        gmm.log_likelihoods_,
        f"{name}  log likelihood (K={K})",
        f"fig/{name}_loglikelihood.png",
    )

    plot_gmm_result(
        X, gmm, f"{name}  GMM clustering (K={K})", f"fig/{name}_gmm_result.png"
    )
