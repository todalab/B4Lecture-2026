"""線形判別分析（LDA）の処理モジュール。"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pca_process import mean_centering


def calculate_class_means(X, y):
    """各クラスの平均ベクトルを計算する。

    Args:
        X (numpy.ndarray): データ行列。
        y (numpy.ndarray): ラベル配列。

    Returns:
        tuple: クラスの平均を格納した辞書とクラスの配列のタプル。
    """
    classes = np.unique(y)
    means = {}
    for c in classes:
        means[c] = np.mean(X[y == c], axis=0)
    return means, classes


def calculate_scatter_matrices(X, y, means, classes):
    """クラス内分散行列とクラス間分散行列を計算する。

    Args:
        X (numpy.ndarray): データ行列。
        y (numpy.ndarray): ラベル配列。
        means (dict): クラスごとの平均を格納した辞書。
        classes (numpy.ndarray): クラスの配列。

    Returns:
        tuple: クラス内分散行列とクラス間分散行列のタプル。
    """
    n_features = X.shape[1]
    S_W = np.zeros((n_features, n_features))

    # 総クラス内分散行列
    for c in classes:
        diff = X[y == c] - means[c]
        S_W += diff.T @ diff

    # クラス間分散行列
    mean_1 = means[classes[0]].reshape(-1, 1)
    mean_2 = means[classes[1]].reshape(-1, 1)
    mean_diff = mean_1 - mean_2
    S_B = mean_diff @ mean_diff.T

    return S_W, S_B


def solve_eigenproblem(S_W, S_B):
    """一般化固有値問題を解き、線形判別軸を求める。

    Args:
        S_W (numpy.ndarray): クラス内分散行列。
        S_B (numpy.ndarray): クラス間分散行列。

    Returns:
        tuple: ソートされた固有値と、対応する固有ベクトルのタプル。
    """
    S_W_inv = np.linalg.inv(S_W)
    eigenvalues, eigenvectors = np.linalg.eig(S_W_inv @ S_B)

    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    sorted_idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[sorted_idx], eigenvectors[:, sorted_idx]


def main():
    """2次元・2クラスデータに対してLDAの処理を実行するメイン関数。"""
    df = pd.read_csv("../data/lda_2class.csv")
    X = df[["x1", "x2"]].values
    y = df["label"].values

    # プロットの中心を合わせるためにデータを中心化
    global_mean, X_centered = mean_centering(X)

    means, classes = calculate_class_means(X, y)
    S_W, S_B = calculate_scatter_matrices(X, y, means, classes)
    eigenvalues, eigenvectors = solve_eigenproblem(S_W, S_B)

    # 最大固有値に対応する固有ベクトルを取得し、単位ベクトルに正規化
    lda_axis = eigenvectors[:, 0]
    lda_axis = lda_axis / np.linalg.norm(lda_axis)

    plt.figure(figsize=(8, 8))
    colors = ["blue", "orange"]
    for i, c in enumerate(classes):
        X_c = X[y == c]
        plt.scatter(
            X_c[:, 0], X_c[:, 1], alpha=0.5, color=colors[i], label=f"class {int(c)}"
        )

    # データの分布幅に合わせて直線の長さを計算
    projections = X_centered @ lda_axis
    start_point = global_mean + np.min(projections) * lda_axis
    end_point = global_mean + np.max(projections) * lda_axis

    plt.plot(
        [start_point[0], end_point[0]],
        [start_point[1], end_point[1]],
        color="red",
        linewidth=2,
        label="LDA Projection Axis",
    )

    plt.title("LDA: Original Data and Projection Axis")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # すべてのデータを判別軸（1次元）に射影する
    projected_1d = X_centered @ lda_axis

    # 射影後のデータをクラスごとに辞書にまとめる
    proj_by_class = {c: projected_1d[y == c] for c in classes}

    # クラスごとの射影データの平均値を求める
    mean_proj_0 = np.mean(proj_by_class[classes[0]])
    mean_proj_1 = np.mean(proj_by_class[classes[1]])

    # 2つのクラスの射影平均の中点を分類の閾値として設定する
    threshold = (mean_proj_0 + mean_proj_1) / 2

    # クラス0の平均がクラス1の平均より小さい場合と大きい場合で条件を分け、閾値を基準にラベルを予測
    if mean_proj_0 < mean_proj_1:
        y_pred = np.where(projected_1d < threshold, classes[0], classes[1])
    else:
        y_pred = np.where(projected_1d >= threshold, classes[0], classes[1])

    # 予測ラベルと実際のラベルを比較し、正解データ数の割合としてAccuracyを計算
    accuracy = np.sum(y_pred == y) / len(y)

    print("=== LDA 分類結果 ===")
    print(f"クラス {int(classes[0])} の射影後の平均: {mean_proj_0:.4f}")
    print(f"クラス {int(classes[1])} の射影後の平均: {mean_proj_1:.4f}")
    print(f"分類の閾値: {threshold:.3f}")
    print(f"Accuracy (正解率): {accuracy * 100:.2f}%")

    plt.figure(figsize=(10, 4))

    # クラスごとに高さを変えてプロット
    plt.scatter(
        proj_by_class[classes[0]],
        np.zeros_like(proj_by_class[classes[0]]),
        alpha=0.7,
        color=colors[0],
        label=f"class {int(classes[0])}",
    )
    plt.scatter(
        proj_by_class[classes[1]],
        np.ones_like(proj_by_class[classes[1]]),
        alpha=0.7,
        color=colors[1],
        label=f"class {int(classes[1])}",
    )

    # 閾値の垂直線を赤の点線で描画
    plt.axvline(
        x=threshold, color="red", linestyle="--", label=f"threshold={threshold:.3f}"
    )

    plt.title(f"LDA: One-dimension projection (accuracy={accuracy:.3f})")

    plt.ylim(-0.5, 1.5)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3, axis="x")

    plt.show()


if __name__ == "__main__":
    main()
