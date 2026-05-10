"""
混合ガウスモデル（GMM）の実装モジュール。

EMアルゴリズムを使ってデータをクラスタリングし、AIC/BIC による最適クラスター数の選択をサポートする。
"""

import argparse
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
    2次元データを散布図として表示する。

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

    Attributes:
        n_components (int): クラスター数 K。
        max_iter (int): EM ループの最大反復回数。
        tol (float): 対数尤度の変化量がこの値を下回ったら収束とみなす。
        reg_covar (float): 共分散行列の対角成分に加える正則化項（特異性回避）。
        means_ (np.ndarray): 各成分の平均ベクトル。
        covariances_ (np.ndarray): 各成分の共分散行列。
        weights_ (np.ndarray): 各成分の混合重み。
        log_likelihood_ (list[float]): 反復ごとの対数尤度の履歴。
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

        初期化:
            - 平均: データからランダムにK点を選択。
            - 共分散: データ全体の共分散行列 + 正則化項（各成分共通）。
            - 重み: 一様分布 1/K。

        収束条件:
            対数尤度の変化量が tol 未満になった時点で終了。

        Args:
            X (np.ndarray): 入力データ。
        """
        n_samples, n_features = X.shape
        # パラメータの初期化
        np.random.seed(0)  # 再現性のためのシード設定
        random_indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[random_indices]
        # 共分散行列の初期化（対角成分に正則化項を加えて特異行列を回避）
        self.covariances_ = np.array(
            [
                np.cov(X.T) + np.eye(n_features) * self.reg_cover
                for _ in range(self.n_components)
            ]
        )
        self.weights_ = np.ones(self.n_components) / self.n_components

        self.log_likelihood_ = []

        # EMループ
        for i in range(self.max_iter):
            # Eステップ: 責任度 γ(z_nk) の計算
            # 各データ x_n が各クラスタ k に所属する負担率 γ(z_nk) を求める
            responsibilities = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                rv = multivariate_normal(self.means_[k], self.covariances_[k])
                # 分子
                responsibilities[:, k] = self.weights_[k] * rv.pdf(X)

            # 分母（全クラスタにわたる尤度の和）
            resp_sum = np.sum(responsibilities, axis=1)

            # 対数尤度
            log_likelihood = np.sum(np.log(resp_sum))
            self.log_likelihood_.append(log_likelihood)

            # 負担率を正規化
            responsibilities = responsibilities / resp_sum[:, np.newaxis]

            # 収束判定
            if (
                i > 0
                and abs(self.log_likelihood_[-1] - self.log_likelihood_[-2]) < self.tol
            ):
                break

            N_k = np.sum(responsibilities, axis=0)

            # Mステップ: パラメータの更新
            for k in range(self.n_components):
                # 平均の更新
                self.means_[k] = (responsibilities[:, k] @ X) / N_k[k]

                # 共分散の更新
                diff = X - self.means_[k]
                self.covariances_[k] = (
                    (diff.T * responsibilities[:, k]) @ diff / N_k[k]
                ) + np.eye(n_features) * self.reg_cover

            # 混合係数の更新
            self.weights_ = N_k / n_samples

    def predict(self, X):
        """
        各サンプルの所属クラスターを予測する。

        Args:
            X (np.ndarray): 観測データ。
        Returns:
            np.ndarray: クラスターラベル。
        """
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            rv = multivariate_normal(self.means_[k], self.covariances_[k])
            responsibilities[:, k] = self.weights_[k] * rv.pdf(X)
        return np.argmax(responsibilities, axis=1)


# ---課題 3-3: GMMの結果を可視化---
def plot_gmm_results(X, gmm, title="GMM Clustering Results"):
    """
    GMMのクラスタリング結果を散布図 + 等高線で可視化する。
    各成分のガウス分布を塗りつぶし等高線で示し、データ点をクラスターごとに色分けして描画する。

    Args:
        X (np.ndarray): 入力データ。shape。
        gmm (GaussianMixtureModel): 学習済みGMMインスタンス。
        title (str): グラフのタイトル。
    """
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

    # 各成分の平均の描画
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
    """
    EMアルゴリズムの対数尤度の収束曲線を描画する。

    Args:
        log_likelihood (list[float]): 反復ごとの対数尤度リスト。
        title (str): グラフのタイトル。
    """
    plt.figure(figsize=(8, 6))
    plt.plot(log_likelihood, marker="o", markersize=4)
    plt.title(title)
    plt.xlabel("iter cnt")
    plt.ylabel("log likelihood")
    plt.grid(True)
    plt.show()


# ---課題 3-4: 情報力基準の計算 ---


def calculate_information_criteria(gmm, X):
    """
    学習済みGMMのAICおよびBICを計算する。

    自由パラメータ数 d の内訳:
        - 混合係数 : K - 1
        - 平均     : K × D
        - 共分散   : K × D × (D + 1) / 2

    AIC（赤池情報量基準）: AIC = -2 ln(L) + 2d
    BIC（ベイズ情報量基準）: BIC = -2 ln(L) + d ln(N)

    Args:
        gmm (GaussianMixtureModel): 学習済みのGMMインスタンス。
        X (np.ndarray): 学習に用いたデータ。
    Returns:
        tuple[float, float]: (AIC, BIC) の値。
    """
    n_samples, n_features = X.shape
    log_likelihood = gmm.log_likelihood_[-1]

    # 自由パラメータ数の合計
    n_weights = gmm.n_components - 1
    n_means = gmm.n_components * n_features
    n_covariances = gmm.n_components * n_features * (n_features + 1) / 2
    d = n_weights + n_means + n_covariances

    aic = -2 * log_likelihood + 2 * d
    bic = -2 * log_likelihood + d * np.log(n_samples)
    return aic, bic


def evaluate_optimal_clusters(
    X, max_k=8, title="AIC and BIC for Model Selection", plot_scatter=False
):
    """
    K = 1 〜 max_k のGMMを学習し、AIC/BICで最適クラスター数を評価する。

    各KのAIC・BICを折れ線グラフで表示し、それぞれの最小値に縦線を引く。
    plot_scatter=True の場合は各KのGMMクラスタリング散布図も描画する。

    Args:
        X (np.ndarray): 入力データ。
        max_k (int): 評価するクラスター数の上限。
        title (str): AIC/BICグラフのタイトル。
        plot_scatter (bool): 各Kの散布図を描画するかどうか。デフォルト False。
    """
    k_values = list(range(1, max_k + 1))
    aic_values = []
    bic_values = []

    for k in k_values:
        gmm = GaussianMixtureModel(n_components=k)
        gmm.fit(X)
        aic, bic = calculate_information_criteria(gmm, X)
        aic_values.append(aic)
        bic_values.append(bic)

        # --scatterが指定されたら各Kのクラスタリング散布図を描画
        if plot_scatter:
            plot_gmm_results(X, gmm, title=f"GMM Clustering (K={k})")

    # AIC/BICグラフの描画
    plt.figure(figsize=(9, 6))
    plt.plot(k_values, aic_values, marker="o", label="AIC", color="blue", linewidth=2)
    plt.plot(k_values, bic_values, marker="s", label="BIC", color="orange", linewidth=2)

    best_k_aic = k_values[np.argmin(aic_values)]
    best_k_bic = k_values[np.argmin(bic_values)]

    plt.axvline(
        x=best_k_aic,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"AIC min (K={best_k_aic})",
    )
    plt.axvline(
        x=best_k_bic,
        color="orange",
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
    """
    コマンドライン引数を解析し、各CSVファイルに対してGMMを実行する。

    オプション:
        --scatter : K=1〜8の散布図のみ描画する。
        --files   : 処理するCSVファイル名のリスト（デフォルト: data1〜3.csv）。
        --k       : クラスター数K（--scatter非使用時のみ有効、デフォルト: 3）。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="K=1〜8の各クラスタリング散布図のみ表示する",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["data1.csv", "data2.csv", "data3.csv"],
        help="処理するCSVファイル名",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="クラスター数 K",
    )
    args = parser.parse_args()

    for filepath in args.files:
        if not os.path.exists("../data/" + filepath):
            print(f"File {filepath} not found.")
            continue
        X = load_data("../data/" + filepath)

        if args.scatter:
            # 散布図のみ：K=1〜8の散布図だけ描画
            for k in range(1, 9):
                gmm = GaussianMixtureModel(n_components=k)
                gmm.fit(X)
                plot_gmm_results(X, gmm, title=f"{filepath} GMM Clustering (K={k})")
        else:
            plot_data(X, f"scatter for {filepath}")

            n_components = args.k
            gmm = GaussianMixtureModel(n_components=n_components)
            gmm.fit(X)
            plot_gmm_results(X, gmm, title=f"{filepath} GMM Clustering Results")
            plot_convergence(gmm.log_likelihood_, title=f"{filepath} Log Likelihood")

            evaluate_optimal_clusters(X, max_k=8, title=f"{filepath} AIC BIC")


if __name__ == "__main__":
    main()
