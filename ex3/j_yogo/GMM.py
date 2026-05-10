import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


# ---課題 3-1:データ確認---
def load_data(file_path):
    """
    CSVファイルを読み込み、numpy配列として返す。

    Args:
        file_path (str): 読み込むデータのパス。
    Returns:
        np.ndarray: 読み込まれたデータ。
    """
    return np.loadtxt(file_path, delimiter=",")


def plot_data(X, title):
    """
    データの初期状態を散布図として表示する。

    Args:
        X (np.ndarray): 2次元データ。
        title (str): グラフのタイトル。
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.show()


# ---課題 3-2:GMM---
class GaussianMixtureModel:
    """
    EMアルゴリズムを用いた混合ガウスモデル（GMM）のクラス。
    """

    def __init__(self, n_components, max_iter=100, tol=1e-4, reg_cover=1e-6):
        """
        GMMの初期化。

        Args:
            n_components (int): クラスター数 (K)。
            max_iter (int): 最大反復回数。
            tol (float): 収束判定の閾値。
            reg_cover (float): 共分散行列の特異性回避用正則化パラメータ。
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_cover = reg_cover
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.log_likelihood_ = []

    def fit(self, X):
        """
        EMアルゴリズムを用いてモデルをデータに適合させる。

        Args:
            X (np.ndarray): 入力データ。
        """
        n_samples, n_features = X.shape
        # パラメータの初期化
        np.random.seed(0)  # 再現性のためのシード設定
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

            resp_sum = np.sum(responsibilities, axis=1)
            log_likelihood = np.sum(np.log(resp_sum))
            self.log_likelihood_.append(log_likelihood)

            responsibilities = responsibilities / resp_sum[:, np.newaxis]
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
                    (diff.T * responsibilities[:, k]) @ diff / N_k[k]
                ) + np.eye(n_features) * self.reg_cover

            # 重みの更新
            self.weights_ = N_k / n_samples

    def predict(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            rv = multivariate_normal(self.means_[k], self.covariances_[k])
            responsibilities[:, k] = self.weights_[k] * rv.pdf(X)
        return np.argmax(responsibilities, axis=1)


# ---課題 3-3: GMMの結果を可視化---
def plot_gmm_results(X, gmm, title="GMM Clustering Results"):
    labels = gmm.predict(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors

    # 等高線の描画準備
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    X_grid, Y_grid = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )
    pos = np.dstack((X_grid, Y_grid))

    # 各ガウス分布の塗りつぶし等高線を表示
    for k in range(gmm.n_components):
        rv = multivariate_normal(gmm.means_[k], gmm.covariances_[k])
        Z = rv.pdf(pos)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            f"cmap_{k}", ["white", colors[k]]
        )
        ax.contourf(X_grid, Y_grid, Z, levels=5, cmap=cmap, alpha=0.4)
        ax.contour(
            X_grid, Y_grid, Z, levels=5, colors=[colors[k]], alpha=0.6, linewidths=0.8
        )

    # 各クラスターを個別に描画して凡例に登録
    for k in range(gmm.n_components):
        ax.scatter(
            X[labels == k, 0],
            X[labels == k, 1],
            s=15,
            alpha=0.7,
            color=colors[k],
            label=f"Cluster {k + 1}",
        )

    # 中心の描画
    ax.scatter(
        gmm.means_[:, 0], gmm.means_[:, 1], c="black", marker="*", s=200, zorder=5
    )

    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.show()


def plot_convergence(log_likelihood, title="Log Likelihood Convergence"):
    plt.figure(figsize=(8, 6))
    plt.plot(log_likelihood, marker="o", markersize=4)
    plt.title(title)
    plt.xlabel("iter cnt")
    plt.ylabel("log likelihood")
    plt.grid(True)
    plt.show()


# ---課題 3-4: 情報力基準の計算 ---


def calculate_information_criteria(gmm, X):
    n_samples, n_features = X.shape
    log_likelihood = gmm.log_likelihood_[-1]

    n_weights = gmm.n_components - 1
    n_means = gmm.n_components * n_features
    n_covariances = gmm.n_components * n_features * (n_features + 1) / 2
    d = n_weights + n_means + n_covariances

    aic = -2 * log_likelihood + 2 * d
    bic = -2 * log_likelihood + d * np.log(n_samples)
    return aic, bic


def evaluate_optimal_clusters(X, max_k=8, title="AIC and BIC for Model Selection"):
    k_values = list(range(1, max_k + 1))
    aic_values = []
    bic_values = []

    for k in k_values:
        gmm = GaussianMixtureModel(n_components=k)
        gmm.fit(X)
        aic, bic = calculate_information_criteria(gmm, X)
        aic_values.append(aic)
        bic_values.append(bic)

        # ↓ 追加: 各Kのクラスタリング散布図を描画
        # plot_gmm_results(X, gmm, title=f"GMM Clustering (K={k})")

    # AIC/BICグラフ（既存のまま）
    plt.figure(figsize=(9, 6))
    plt.plot(
        k_values, aic_values, marker="o", label="AIC", color="#1f77b4", linewidth=2
    )
    plt.plot(
        k_values, bic_values, marker="s", label="BIC", color="#ff7f0e", linewidth=2
    )

    best_k_aic = k_values[np.argmin(aic_values)]
    best_k_bic = k_values[np.argmin(bic_values)]

    plt.axvline(
        x=best_k_aic,
        color="#1f77b4",
        linestyle="--",
        alpha=0.7,
        label=f"AIC min (K={best_k_aic})",
    )
    plt.axvline(
        x=best_k_bic,
        color="#ff7f0e",
        linestyle="--",
        alpha=0.7,
        label=f"BIC min (K={best_k_bic})",
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Information Criterion Value", fontsize=12)
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
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
        plot_convergence(gmm.log_likelihood_, title=f"{filepath} Log Likelihood")

        evaluate_optimal_clusters(X, max_k=8, title=f"{filepath} AIC BIC")


if __name__ == "__main__":
    main()
