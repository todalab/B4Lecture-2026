"""主成分分析（PCA）の処理モジュール."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mean_centering(X):
    """入力データの各特徴量の平均を引き、中心化を行う.

    Args:
        X (numpy.ndarray): 元のデータ行列.

    Returns:
        tuple: 平均値の配列と、中心化されたデータ行列を含むタプル.
    """
    mean = np.mean(X, axis=0)
    x_centered = X - mean
    return mean, x_centered


def calculate_covariance(X_centered):
    """中心化されたデータの共分散行列を計算する.

    Args:
        X_centered (numpy.ndarray): 中心化されたデータ行列.

    Returns:
        numpy.ndarray: 共分散行列.
    """
    # 共分散行列を計算
    M = X_centered.shape[0]
    return X_centered.T @ X_centered / M


def calculate_eigen(matrix):
    """与えられた行列の固有値分解を行い、結果を降順にソートする.

    Args:
        matrix (numpy.ndarray): 共分散行列.

    Returns:
        tuple: ソートされた固有値と、対応する固有ベクトルのタプル.
    """
    # 固有値分解
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # 固有値が大きい順にソート
    sorted_idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[sorted_idx], eigenvectors[:, sorted_idx]


def calculate_variance_ratio(eigenvalues):
    """固有値に基づいて寄与率および累積寄与率を計算する.

    Args:
        eigenvalues (numpy.ndarray): 固有値.

    Returns:
        tuple: 寄与率と累積寄与率のタプル.
    """
    # 寄与率を計算
    total_variance = np.sum(eigenvalues)
    variance_ratio = eigenvalues / total_variance

    # 累積寄与率を計算
    comulative_ratio = np.cumsum(variance_ratio)

    return variance_ratio, comulative_ratio


def change_basis(eigenvectors, k, X_centered):
    """上位k個の固有ベクトルで構成される新しい基底に、中心化されたデータを射影する.

    Args:
        eigenvectors (numpy.ndarray): 固有ベクトル.
        k (int): 保持する上位主成分の数.
        X_centered (numpy.ndarray): 中心化されたデータ行列.

    Returns:
        numpy.ndarray: 射影されたデータ行列.
    """
    U = eigenvectors[:, :k]
    return X_centered @ U


def process_pca_2d():
    """2次元データに対してPCAを適用し、主成分を可視化する."""
    df = pd.read_csv("../data/pca_2d.csv", header=None)
    X = df.values

    mean, X_centered = mean_centering(X)
    matrix = calculate_covariance(X_centered)
    eigenvalues, eigenvectors = calculate_eigen(matrix)
    variance_ratios, _ = calculate_variance_ratio(eigenvalues)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Original Data")

    colors = ["red", "green"]
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
    plt.xlabel(f"PC1 ({variance_ratios[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({variance_ratios[1] * 100:.1f}%)")
    plt.axis("equal")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.show()


def process_pca_3d():
    """3次元データに対してPCAを適用して3次元上の主成分軸を可視化し、2次元空間への射影を行う."""
    df = pd.read_csv("../data/pca_3d.csv", header=None)
    X = df.values

    mean, X_centered = mean_centering(X)
    matrix = calculate_covariance(X_centered)
    eigenvalues, eigenvectors = calculate_eigen(matrix)
    variance_ratios, _ = calculate_variance_ratio(eigenvalues)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5, label="Original Data")

    colors = ["red", "green", "purple"]
    for i in range(3):
        u = eigenvectors[:, i]

        # 3次元データを主成分軸に射影し、分布の最小値と最大値を取得
        projections = X_centered @ u
        min_proj = np.min(projections)
        max_proj = np.max(projections)

        # 3次元空間における線の始点と終点の座標を計算
        start_point = mean + min_proj * u
        end_point = mean + max_proj * u

        label_text = f"PC{i + 1} ({variance_ratios[i] * 100:.1f}%)"

        ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            [start_point[2], end_point[2]],
            color=colors[i],
            linewidth=3,
            label=label_text,
        )

    ax.set_title("PCA: Three-dimension")
    ax.set_xlabel(f"PC1 ({variance_ratios[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance_ratios[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({variance_ratios[2] * 100:.1f}%)")
    ax.legend(loc="upper left")
    plt.show()

    # PCAにより2次元へ圧縮し、2次元散布図として可視化
    X_projected = change_basis(eigenvectors, 2, X_centered)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.5)
    plt.title("PCA : Three-dimension to Two-dimension")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.show()


def process_pca_100d():
    """100次元データに対してPCAを適用し、2次元への射影を行うとともに累積寄与率を評価する."""
    df = pd.read_csv("../data/pca_100d.csv", header=None)
    X = df.values

    mean, X_centered = mean_centering(X)
    matrix = calculate_covariance(X_centered)
    eigenvalues, eigenvectors = calculate_eigen(matrix)
    variance_ratios, cumulative_ratios = calculate_variance_ratio(eigenvalues)

    X_projected = change_basis(eigenvectors, 2, X_centered)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.5)
    plt.title("PCA : 100-dimension to Two-dimension")
    plt.xlabel(f"PC1 ({variance_ratios[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({variance_ratios[1] * 100:.1f}%)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.show()

    Min_dimension = np.argmax(cumulative_ratios >= 0.90) + 1
    print(f"累積寄与率が90%以上となる最小の次元数: {Min_dimension}")

    # 累積寄与率を示す図の作成
    plt.figure(figsize=(8, 6))
    # 次元数は1から始まるため、x軸の範囲を調整
    dimensions = np.arange(1, len(cumulative_ratios) + 1)
    plt.plot(dimensions, cumulative_ratios, marker=".", linestyle="-")
    plt.axhline(y=0.90, color="red", linestyle="--", label="90% Threshold")
    plt.axvline(
        x=Min_dimension,
        color="green",
        linestyle="--",
        label=f"{Min_dimension}dimension",
    )

    plt.title("PCA 100D: Cumulative Variance Ratio")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def main():
    """各次元での処理と可視化を実行."""
    print("=== 2次元データの処理 (pca_2d.csv) ===")
    process_pca_2d()
    print("\n=== 3次元データの処理 (pca_3d.csv) ===")
    process_pca_3d()
    print("\n=== 100次元データの処理 (pca_100d.csv) ===")
    process_pca_100d()


if __name__ == "__main__":
    main()
