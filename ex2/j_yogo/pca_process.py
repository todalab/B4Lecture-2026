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
    return X_centered @ X_centered.T / M


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
    comulative_ratio = np.comsum(variance_ratio)

    return comulative_ratio


def change_basis(eigenvectors, k, X_centered):
    U = eigenvectors[:, :k]
    return X_centered @ U


def process_pca_2d():
    df = pd.read_csv("../data/pca_2d.csv", header=None)
    X = df.values

    mean, X_centered = mean_centering(X)
    matrix = calculate_convariance(X_centered)
    eigenvalues, eigenvectors = calculate_eigen(matrix)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Original Data")

    plt.show()

    print()


if __name__ == "__main__":
    print("=== 2次元データの処理 (pca_2d.csv) ===")
    process_pca_2d()
    print("\n=== 3次元データの処理 (pca_3d.csv) ===")

    print("\n=== 100次元データの処理 (pca_100d.csv) ===")
