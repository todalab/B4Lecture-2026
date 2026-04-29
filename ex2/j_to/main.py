"""P.C.A./L.D.A. visualisation."""

import subprocess
from math import sqrt
from os import path

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from numpy import array, column_stack, concatenate, mean, ones, outer
from numpy.linalg import eig, eigh, pinv
from pandas import read_csv

# Assume running on macOS. Choose system Japanese font
plt.rcParams["font.family"] = ["Hiragino Sans"]

FIG_DIR = "fig"


def covariance(X):
    """Compute the covariance matrix of mean-centered data `X`."""
    return (X.T @ X) / len(X)


def pca(filepath: str, is_3d=False):
    """Run P.C.A. on a CSV dataset and save scatter/PC-axis and reduced-dimension plots."""
    # Extract filename (without extension) as an identifier
    basename = path.basename(filepath)
    filename = path.splitext(basename)[0]

    # Read data (print out its scatter point graph)
    data = read_csv(filepath, header=None)
    X = array(data)  # shape: M by n
    n = data.shape[1]  # Dimension

    # Set up figure
    fig = figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d" if is_3d else None)

    # 元データの散布図
    if n <= 3:
        ax.grid(True)
        ax.axis("equal")
        ax.scatter(*X.T, label="Data", color="gray")
        ax.legend()
        fig.savefig(f"{FIG_DIR}/1_{filename}_scatter.png")

    # 平均中心化 Zero-out mean
    mu = mean(X, axis=0)
    X_centred = X - mu

    # 共分散行列 Calculate the covariance matrix
    Sigma = covariance(X_centred)

    # Calculate eigenvalues/vectors
    lambdas, us = eigh(Sigma)

    # Sort eigenpairs by eigenvalue
    order = lambdas.argsort()[::-1]
    lambdas_sorted = lambdas[order]
    us_sorted = us[:, order]

    # 各主成分の寄与率と累積寄与率を計算する
    total = sum(lambdas_sorted)
    ratios = lambdas_sorted / total
    accumulated_ratios = [0] + list(ratios.cumsum())

    # 主成分の軸を散布図上に重ねる
    for index, (lam, u, ratio) in enumerate(zip(lambdas_sorted, us_sorted.T, ratios)):
        factor = 2 * sqrt(lam)

        if n <= 3:
            ax.plot(
                *array([mu - factor * u, mu + factor * u]).T,
                "-",
                linewidth=2,
                label=f"PC-{index + 1} ({ratio * 100:.1f}%)",
            )

    # 主成分の軸を散布図上に重ねる
    if n <= 3:
        ax.set_title(f"P.C.A.: {n}次元データと主成分軸")
        ax.legend()
        fig.savefig(f"{FIG_DIR}/1_{filename}_pca_basis.png")

    # 2次元へ圧縮する
    if n > 2:
        U = us_sorted[:, :2]
        X_hat = X_centred @ U

        fig = figure(figsize=(8, 8))
        ax = fig.add_subplot()
        ax.grid(True)

        ax.axhline(0, color="black")
        ax.axvline(0, color="black")
        ax.scatter(*X_hat.T)

        pc_percentages = ratios * 100
        ax.set_xlabel(f"PC1 ({pc_percentages[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({pc_percentages[1]:.1f}%)")

        ax.set_title(f"P.C.A.: {basename} の2次元圧縮")
        fig.savefig(f"{FIG_DIR}/1_{filename}_pca_reduction.png")

    if n > 3:
        tau = 0.9
        k = next(
            (i + 1 for i, ratio in enumerate(accumulated_ratios[1:]) if ratio >= tau)
        )

        fig = figure(figsize=(12, 6))
        ax = fig.add_subplot()
        ax.grid(True)

        ax.plot(range(n + 1), accumulated_ratios, marker="o")
        ax.axvline(x=k, color="g", linestyle="--", label=f"k={k}")
        ax.axhline(y=tau, color="r", linestyle="--", label=f"{tau * 100:.1f}%")
        ax.legend()

        ax.set_xlabel("k")
        ax.set_ylabel("累積寄与率 (%)")
        ax.set_title("P.C.A.：累積寄与率")
        fig.savefig(f"{FIG_DIR}/1_{filename}_accumulated_ratio.png")


def lda_extra_metrics(labels, projections):
    """Compute precision/recall/F1 for L.D.A. projections."""
    labels = array(labels).astype(int)
    projections = array(projections)
    positive = labels == 0
    predicted = projections > 0

    tp = sum(predicted & positive)
    fp = sum(predicted & ~positive)
    fn = sum(~predicted & positive)

    assert tp + fp != 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    print("L.D.A. extra metrics")
    print(f"precision: {precision}, recall: {recall}, f1: {f1}")


def lda(filename: str):
    """Run 2-class L.D.A. on a CSV dataset, visualise the projection axis, and save plots."""
    data = read_csv(filename)

    # Read data
    x, y, label = data["x1"], data["x2"], data["label"]
    X = column_stack([x, y])
    labels = array(label)
    Xs = [X[labels == c] for c in range(2)]

    # Set up figure
    fig = figure(figsize=(8, 8))
    ax = fig.add_subplot()
    for c, X in enumerate(Xs):
        ax.scatter(*X.T, label=f"Class {c}")

    # クラスごとの平均ベクトルを計算する
    mu, Sigma = zip(*[(mean(X, axis=0), covariance(X)) for X in Xs])

    # クラス内分散行列とクラス間分散行列を計算する
    S_B = outer(mu[0] - mu[1], mu[0] - mu[1])
    S_W = sum(Sigma)

    # 一般化固有値問題を解き，L.D.A.の射影方向を求める
    lambdas, us = eig(pinv(S_W) @ S_B)
    eigpairs = sorted(zip(lambdas, us.T), key=lambda pair: -pair[0])

    # 元データの散布図にL.D.A.の射影軸を重ねて可視化する
    lam, u = eigpairs[0]
    factor = 2 * sqrt(lam)
    ax.plot(*array([-factor * u, factor * u]).T, "-", linewidth=2, label="L.D.A. Axis")
    ax.grid(True)
    ax.legend()

    ax.set_title("散布図とL.D.A.の射影軸")
    fig.savefig(f"{FIG_DIR}/2_lda_axis.png")

    # L.D.A.軸へ射影した1次元データを可視化する
    fig = figure(figsize=(12, 4))
    ax = fig.add_subplot()
    ax.grid(True)

    projections, labels = [], []
    correct = 0
    for class_index, class_samples in enumerate(Xs):
        projection = class_samples @ u
        correct += sum((projection > 0) == (class_index == 0))
        label = ones(len(class_samples)) * class_index

        projections.append(projection)
        labels.append(label)

        projected = column_stack([projection, label])
        ax.scatter(*projected.T, label=f"Class {class_index}")

    accuracy = correct / sum(len(X) for X in Xs)
    lda_extra_metrics(concatenate(labels), concatenate(projections))

    ax.set_yticks([0, 1])
    ax.set_title(f"L.D.A.: 1次元射影 (accuracy={accuracy:.3f})")
    ax.legend()
    fig.savefig(f"{FIG_DIR}/2_lda_projection.png")


def main():
    """Generate P.C.A./L.D.A. figures from the datasets into `fig/`."""
    subprocess.run(["mkdir", "-p", FIG_DIR])

    pca("data/pca_2d.csv")
    pca("data/pca_3d.csv", is_3d=True)
    pca("data/pca_100d.csv")
    lda("data/lda_2class.csv")


if __name__ == "__main__":
    main()
