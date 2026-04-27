import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mean_centering(X):
    mean = np.mean(X, axis=0)
    x_centered = X - mean
    return mean, x_centered


def calculate_convariance(X_centered):
    # 共分散行列を計算
    M = X_centered.shape[0]
    return X_centered.T @ X_centered / M


def calculate_eigen(matrix):
    # 固有値分解
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # 固有値が大きい順にソート
    sorted_idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[sorted_idx], eigenvectors[:, sorted_idx]


def calculate_variance_ratio(eigenvalues):
    # 寄与率を計算
    total_variance = np.sum(eigenvalues)
    variance_ratio = eigenvalues / total_variance

    # 累積寄与率を計算
    comulative_ratio = np.cumsum(variance_ratio)

    return variance_ratio, comulative_ratio


def change_basis(eigenvectors, k, X_centered):
    U = eigenvectors[:, :k]
    return X_centered @ U


def process_pca_2d():
    df = pd.read_csv("../data/pca_2d.csv", header=None)
    X = df.values

    mean, X_centered = mean_centering(X)
    matrix = calculate_convariance(X_centered)
    eigenvalues, eigenvectors = calculate_eigen(matrix)
    variance_ratios, _ = calculate_variance_ratio(eigenvalues)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Original Data")

    colors = ["#d62728", "#2ca02c"]
    for i in range(2):
        # 単位ベクトルである固有ベクトルを取得
        u = eigenvectors[:, i]

        # 全データを主成分軸に射影し、分布の最小値と最大値を取得
        projections = X_centered @ u
        min_proj = np.min(projections)
        max_proj = np.max(projections)

        # データの分布の両端に合わせた線の始点と終点を計算
        start_point = mean + min_proj * u
        end_point = mean + max_proj * u

        label_text = f"PC{i + 1} ({variance_ratios[i] * 100:.1f}%)"

        # 計算した始点と終点を結ぶ直線を引く
        plt.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            color=colors[i],
            linewidth=3,
            label=label_text,
        )

    plt.title("PCA: Two-dimension")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.show()


def process_pca_3d():
    df = pd.read_csv("../data/pca_3d.csv", header=None)
    X = df.values

    mean, X_centered = mean_centering(X)
    matrix = calculate_convariance(X_centered)
    eigenvalues, eigenvectors = calculate_eigen(matrix)
    variance_ratios, _ = calculate_variance_ratio(eigenvalues)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Original Data")

    colors = ["#d62728", "#2ca02c"]
    for i in range(2):
        # 単位ベクトルである固有ベクトルを取得
        u = eigenvectors[:, i]

        # 全データを主成分軸に射影し、分布の最小値と最大値を取得
        projections = X_centered @ u
        min_proj = np.min(projections)
        max_proj = np.max(projections)

        # データの分布の両端に合わせた線の始点と終点を計算
        start_point = mean + min_proj * u
        end_point = mean + max_proj * u

        label_text = f"PC{i + 1} ({variance_ratios[i] * 100:.1f}%)"

        # 計算した始点と終点を結ぶ直線を引く
        plt.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            color=colors[i],
            linewidth=3,
            label=label_text,
        )

    plt.title("PCA: Two-dimension")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    print("=== 2次元データの処理 (pca_2d.csv) ===")
    process_pca_2d()
    print("\n=== 3次元データの処理 (pca_3d.csv) ===")

    print("\n=== 100次元データの処理 (pca_100d.csv) ===")
