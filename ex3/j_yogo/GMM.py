import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


# ---課題 3-1:データ確認---
def load_data(file_path):
    return np.loadtxt(file_path, delimiter=",")


def plot_data(X, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.show()


# ---課題 3-2:GMM---
class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-4, reg_cover=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_cover = reg_cover

    def fit(self, X):
        n_samples, n_features = X.shape
        # パラメータの初期化
        np.random.seed(42)  # 再現性のためのシード設定
        random_indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[random_indices]
        # 共分散行列の初期化（対角成分に微小な正の数を加えるp32より）
        self.covariances_ = np.array(
            [
                np.cov(X.T) + np.eye(n_features) * self.reg_cover
                for _ in range(self.n_components)
            ]
        )
        self.weights_ = np.ones(self.n_components) / self.n_components

        self.log_likelihood_ = []

        for i in range(self.max_iter):
            # Eステップ: 責任度の計算
            responsibilities = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                rv = multivariate_normal(self.means_[k], self.covariances_[k])
                responsibilities[:, k] = self.weights_[k] * rv.pdf(X)

            log_likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))
            self.log_likelihood_.append(log_likelihood)

            responsibilities = responsibilities / np.sum(
                responsibilities, axis=1, keepdims=True
            )
            # 収束判定
            if (
                i > 0
                and abs(self.log_likelihood_[-1] - self.log_likelihood_[-2]) < self.tol
            ):
                break

            N_k = np.sum(responsibilities, axis=0)

            for k in range(self.n_components):
                # 平均の更新
                self.means_[k] = (responsibilities[:, k] @ X) / N_k[k]

                # 共分散の更新
                diff = X - self.means_[k]
                self.covariances_[k] = (
                    (responsibilities[:, k] * diff.T) @ diff / N_k[k]
                ) + np.eye(n_features) * self.reg_cover

            # 重みの更新
            self.weights_[k] = N_k[k] / n_samples

    def predict(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            rv = multivariate_normal(self.means_[k], self.covariances_[k])
            responsibilities[:, k] = self.weights_[k] * rv.pdf(X)
        return np.argmax(responsibilities, axis=1)


def plot_gmm_results(X, gmm, title="GMM Clustering Results"):
    labels = gmm.predict(X)
    plt.figure(figsize=(8, 6))

    # 各クラスターを個別に描画して凡例に登録
    for k in range(gmm.n_components):
        plt.scatter(
            X[labels == k, 0],
            X[labels == k, 1],
            s=15,
            alpha=0.7,
            label=f"Cluster {k + 1}",
        )

    # 中心の描画
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c="red", marker="*", s=100)

    # 等高線の描画準備
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    X_grid, Y_grid = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )
    pos = np.dstack((X_grid, Y_grid))

    # 各ガウス分布の等高線を表示
    for k in range(gmm.n_components):
        rv = multivariate_normal(gmm.means_[k], gmm.covariances_[k])
        plt.contour(X_grid, Y_grid, rv.pdf(pos), levels=3, colors="black", alpha=0.5)

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def plot_convergence(log_likelihood, title="Log Likelihood Convergence"):
    plt.figure(figsize=(8, 6))
    plt.plot(log_likelihood, marker="o", markersize=4)
    plt.title(title)
    plt.xlabel("iter cnt")
    plt.ylabel("log likelihood")
    plt.grid(True)
    plt.show()


def main():
    filepaths = ["data1.csv", "data2.csv", "data3.csv"]
    for filepath in filepaths:
        if not os.path.exists("../data/" + filepath):
            print(f"File {filepath} not found.")
            continue
        X = load_data("../data/" + filepath)

        plot_data(X, f"scatter for {filepath}")

        n_components = 3
        gmm = GaussianMixtureModel(n_components=n_components)
        gmm.fit(X)
        plot_gmm_results(X, gmm, title=f"{filepath} GMM Clustering Results")
        plot_convergence(gmm.log_likelihood_, title=f"{filepath}Log Likelihood")


if __name__ == "__main__":
    main()
