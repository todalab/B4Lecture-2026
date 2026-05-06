# -*- coding: utf-8 -*-
"""課題3 GMM EMアルゴリズム."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "sans-serif"


class parameter:
    """EMアルゴリズムで学習された各ステップのパラメータを保存するクラス."""

    def __init__(self, mu: np.ndarray, Sigma: np.ndarray, pi: np.ndarray) -> None:
        """
        初期化 mu, Sigma, piを1次元配列として保存する.

        Args:
            mu (np.ndarray): 混合ガウス分布のセントロイドの集合(K,2)
            Sigma (np.ndarray): 混合ガウス分布の分散共分散行列(K,2,2)
            pi (np.ndarray): 混合ガウス分布の混合係数の集合(K)
        """
        self.mu = np.array([mu])
        self.Sigma = np.array([Sigma])
        self.pi = np.array([pi])

    def get_mu(self, index) -> np.ndarray:
        return self.mu[index]

    def get_Sigma(self, index) -> np.ndarray:
        return self.Sigma[index]

    def get_pi(self, index) -> np.ndarray:
        return self.pi[index]

    def append_mu(self, mu) -> None:
        self.mu = np.vstack([self.mu, [mu]])

    def append_Sigma(self, Sigma) -> None:
        self.Sigma = np.vstack([self.Sigma, [Sigma]])

    def append_pi(self, pi) -> None:
        self.pi = np.vstack([self.pi, [pi]])

    def __str__(self) -> str:
        return f"mu: {self.mu}\nSigma: {self.Sigma}\npi: {self.pi}\n"


def show_data_scatter(x1: pd.Series, x2: pd.Series, title: str) -> None:
    """
    2次元データの散布図を作成して保存する.

    Args:
        x1 (pd.Series): x1座標の1次元データ.
        x2 (pd.Series): x2座標の1次元データ.
        title (str): 図のタイトルおよび保存ファイル名に使う文字列.
    Returns:
        None: `output/{title}.png` に図を保存する.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, x2, label="data")
    ax.legend(loc="upper left")
    ax.set_title(title)
    ax.set_ylabel("x2")
    ax.set_xlabel("x1")
    plt.savefig(f"output/{title}.png")
    plt.close()


def gaussian(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    """
    ガウス関数

    Args:
        x (np.ndarray): 観測データの集合(N,2)
        mu (np.ndarray): ガウス分布のセントロイド(2)
        Sigma (np.ndarray): ガウス分布の分散共分散行列(2,2)

    Returns:
        float: 確率
    """
    # 分散共分散行列の行列式
    det = np.linalg.det(Sigma)
    # 分散共分散行列の逆行列（数値的安定性のため pinv を使用）
    inv = np.linalg.pinv(Sigma)
    n = x.ndim
    # det が非常に小さい場合の対応
    if abs(det) < 1e-15:
        det = 1e-15
    return np.exp(-np.diag((x - mu) @ inv @ (x - mu).T) / 2.0) / (
        np.sqrt((2 * np.pi) ** n * abs(det))
    )


def GMM(
    x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, pi: np.ndarray, K: int
) -> np.ndarray:
    """
    混合ガウス分布の式.

    Args:
        x (np.ndarray): 観測データの集合(N,2)
        mu (np.ndarray): GMMのセントロイドの集合(K,2)
        Sigma (np.ndarray): GMMの分散共分散行列(K,2,2)
        pi (np.ndarray): GMMの混合係数の集合(K)
        K (int): クラスタの総数

    Returns:
        np.ndarray: GMMから出力される確率(N)
    """
    return np.sum([pi[k] * gaussian(x, mu[k], Sigma[k]) for k in range(K)], axis=0)


def calculate_gamma(
    x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, pi: np.ndarray, k: int, K: int
) -> np.ndarray:
    """
    負担率を計算する.

    Args:
        x (np.ndarray): 観測データの集合(N,2)
        mu (np.ndarray): GMMのセントロイドの集合(K,2)
        Sigma (np.ndarray): GMMの分散共分散行列(K,2,2)
        pi (np.ndarray): GMMの混合係数の集合(K)
        k (int): 注目するクラスタの番号
        K (int): クラスタの総数

    Returns:
        np.ndarray: 各データのクラスタkの負担率(N)
    """
    normalization = GMM(x, mu, Sigma, pi, K)
    return pi[k] * gaussian(x, mu[k], Sigma[k]) / normalization


def e_step(
    x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, pi: np.ndarray, K: int
) -> np.ndarray:
    """
    EMアルゴリズムのEステップを実行する.

    Args:
        x (np.ndarray): 観測データの集合(N,2)
        mu (np.ndarray): GMMのセントロイドの集合(K,2)
        Sigma (np.ndarray): GMMの分散共分散行列(K,2,2)
        pi (np.ndarray): GMMの混合係数の集合(K)
        K (int): クラスタの総数

    Returns:
        np.ndarray: 各データの負担率(N,K)
    """
    gamma = calculate_gamma(x, mu, Sigma, pi, 0, K).reshape(-1, 1)
    for k in range(1, K):
        gamma = np.concatenate(
            [
                gamma,
                calculate_gamma(x, mu, Sigma, pi, k, K).reshape(-1, 1),
            ],
            1,
        )
    return gamma


def calculate_N_k(gamma: np.ndarray, K: int) -> np.ndarray:
    """
    各クラスタに所属するデータ数を求める.

    Args:
        gamma (np.ndarray): 各データの負担率(N,K)
        K (int): クラスタ数

    Returns:
        np.ndarray: 各クラスタに所属するデータ数(K)
    """
    return np.array([np.sum(gamma[:, k]) for k in range(K)])


def calculate_pi(N_k: np.ndarray, N: int) -> np.ndarray:
    """
    混合係数piを計算する.

    Args:
        N_k (np.ndarray): 各クラスタに所属するデータ数(K)
        N (int): データの総数

    Returns:
        np.ndarray: 各クラスタの混合係数pi_kの集合(K)
    """
    return N_k / N


def calculate_mu(
    x: np.ndarray, gamma: np.ndarray, N_k: np.ndarray, K: int
) -> np.ndarray:
    """
    各クラスタのセントロイドmuを計算する.

    Args:
        x (np.ndarray): 観測データの集合(N,2)
        gamma (np.ndarray): 各データの負担率(N,K)
        N_k (np.ndarray): 各クラスタに所属するデータ数(K)
        K (int): クラスタの総数

    Returns:
        np.ndarray: 各クラスタのセントロイドmuの集合(K,2)
    """
    tmp_mu = np.zeros((K, 2))

    for k in range(K):
        for i in range(len(x)):
            tmp_mu[k] += gamma[i, k] * x[i]
        tmp_mu[k] = tmp_mu[k] / N_k[k]
    return tmp_mu


def calculate_Sigma(
    x: np.ndarray, gamma: np.ndarray, mu: np.ndarray, N_k: np.ndarray, K: int
) -> np.ndarray:
    """
    各クラスタの分散共分散行列Sigmaを計算する.

    Args:
        x (np.ndarray): 観測データの集合(N,2)
        gamma (np.ndarray): 各データの負担率(N,K)
        mu (np.ndarray): 各クラスタのセントロイドの集合(K,2)
        N_k (np.ndarray): 各クラスタに所属するデータ数(K)
        K (int): クラスタの総数

    Returns:
        np.ndarray: 各クラスタの分散共分散行列Sigmaの集合
        (K,2,2)
    """
    tmp_Sigma = np.zeros((K, 2, 2))

    for k in range(K):
        tmp_Sigma[k] = np.zeros((2, 2))
        for i in range(len(x)):
            tmp = np.asanyarray(x[i] - mu[k])[:, np.newaxis]  # reshape(-1,1)
            tmp_Sigma[k] += gamma[i, k] * np.dot(tmp, tmp.T)
        tmp_Sigma[k] = tmp_Sigma[k] / N_k[k]
        # 対角成分に正則化項を加えて特異行列を回避
        tmp_Sigma[k] += 1e-2 * np.eye(2)
    return tmp_Sigma


def calculate_log_likelihood(
    x: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    pi: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    各データのGMMからの確率を計算し、対数を取って合計することで、モデルの対数尤度を計算する.

    Args:
        x (np.ndarray): 観測データの集合(N,2)
        mu (np.ndarray): GMMのセントロイドの集合(K,2)
        Sigma (np.ndarray): GMMの分散共分散行列(K,2,2)
        pi (np.ndarray): GMMの混合係数の集合(K)
        K (int): クラスタの総数

    Returns:
        np.ndarray: モデルの対数尤度
    """
    return np.sum(np.log(GMM(x, mu, Sigma, pi, K)))


def m_step(
    x: np.ndarray,
    gamma: np.ndarray,
    K: int,
) -> tuple:
    """
    EMアルゴリズムのMステップを実行する.

    Args:
        x (np.ndarray): 観測データの集合(N,2)
        gamma (np.ndarray): 各データの負担率(N,K)
        K (int): クラスタの総数

    Returns:
        tuple: 更新されたパラメータ (mu, Sigma, pi) とモデルの対数尤度
    1. mu: 各クラスタのセントロイドの集合(K,2)
    2. Sigma: 各クラスタの分散共分散行列の集合(K,2,2)
    3. pi: 各クラスタの混合係数の集合(K)
    4. log_likelihood: モデルの対数尤度
    """
    new_N_k = calculate_N_k(gamma, K)
    new_pi = calculate_pi(new_N_k, len(x))
    new_mu = calculate_mu(x, gamma, new_N_k, K)
    new_Sigma = calculate_Sigma(x, gamma, new_mu, new_N_k, K)
    log_likelihood = calculate_log_likelihood(x, new_mu, new_Sigma, new_pi, K)
    return (new_mu, new_Sigma, new_pi, log_likelihood)


def set_initial(x: np.ndarray, K: int) -> tuple:
    """
    GMMの初期パラメータを設定する.

    Args:
        x (np.ndarray): 観測データの集合(N,2)
        K (int): クラスタの総数
    Returns:
    tuple: 初期パラメータ (mu, Sigma, pi)
    1. mu: 各クラスタのセントロイドの集合(K,2)
    2. Sigma: 各クラスタの分散共分散行列の集合(K,2,2)
    3. pi: 各クラスタの混合係数の集合(K)
    """
    mu = np.random.rand(K, 2) * (np.max(x, axis=0) - np.min(x, axis=0)) + np.min(
        x, axis=0
    )
    Sigma = np.array([np.eye(2) for i in range(K)])
    pi = np.array([1 / K for i in range(K)])
    print("Initial parameters:\n", mu, "\n", Sigma, "\n", pi)
    return (mu, Sigma, pi)


def loop_em_algorithm(x: np.ndarray, K: int) -> tuple:
    """
    EMアルゴリズムを反復して実行する.

    Args:
        x (np.ndarray): 観測データの集合(N,2)
        K (int): クラスタの総数
    Returns:
        tuple: 学習されたパラメータの集合と対数尤度の履歴
    1. parameters: EMアルゴリズムで学習された各ステップのパラメータを保存するparameterクラスのインスタンス
    2. log_likelihoods: 各反復ステップでのモデルの対数尤度のリスト
    """
    # 初期化
    mu, Sigma, pi = set_initial(x, K)
    parameters = parameter(mu, Sigma, pi)
    log_likelihoods = np.array([calculate_log_likelihood(x, mu, Sigma, pi, K)])
    # 反復
    while True:
        gamma = e_step(x, parameters.mu[-1], parameters.Sigma[-1], parameters.pi[-1], K)
        new_mu, new_Sigma, new_pi, log_likelihood = m_step(x, gamma, K)
        parameters.append_mu(new_mu)
        parameters.append_Sigma(new_Sigma)
        parameters.append_pi(new_pi)
        log_likelihoods = np.append(log_likelihoods, [log_likelihood])
        if np.abs(log_likelihoods[-2] - log_likelihoods[-1]) < 0.01:
            break
    return (parameters, log_likelihoods)


def show_log_likelihood(log_likelihoods: np.ndarray, title: str) -> None:
    """
    EMアルゴリズムでのステップごとのモデルの対数尤度をグラフで出力する.

    Args:
        log_likelihoods (np.ndarray): 各反復ステップでのモデルの対数尤度のリスト
        title (str): 図のタイトルおよび保存ファイル名に使う文字列
    """
    print("Log Likelihoods:\n", log_likelihoods)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(log_likelihoods)), log_likelihoods, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log Likelihood")
    ax.grid(True)
    plt.savefig(f"output/{title}.png")
    plt.close()


def show_data_scatter_and_GMM(
    x: np.ndarray, parameters: parameter, K: int, title: str
) -> None:
    """
    GMMでクラスタリングした2次元データを、色分け、平均、等高線を重ねて表示する.

    Args:
        x (np.ndarray): 観測データの集合(N,2)
        parameters (parameter): EMアルゴリズムで学習されたパラメータ
        K (int): クラスタの総数
        title (str): 図のタイトルおよび保存ファイル名に使う文字列
    Returns:
        None: `output/{title}.png` に図を保存する.
    """
    # 最終的なパラメータを取得
    mu_final = parameters.mu[-1]
    Sigma_final = parameters.Sigma[-1]
    pi_final = parameters.pi[-1]
    print("Final parameters:\n", mu_final, "\n", Sigma_final, "\n", pi_final)

    # E-stepで負担率を計算
    gamma = e_step(x, mu_final, Sigma_final, pi_final, K)

    # 各データについて最大の負担率を持つクラスターを決定
    cluster_assignments = np.argmax(gamma, axis=1)

    # 色の定義
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    # 散布図を描画
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # クラスターごとにデータを色分けして描画
    for k in range(K):
        mask = cluster_assignments == k
        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            c=colors[k],
            label=f"Cluster {k}",
            alpha=0.6,
            s=50,
        )

    # 各ガウス分布の平均（セントロイド）を描画
    for k in range(K):
        ax.plot(
            mu_final[k, 0],
            mu_final[k, 1],
            marker="X",
            color=colors[k % len(colors)],
            markersize=15,
            markeredgecolor="black",
            markeredgewidth=2,
            label=f"Mean {k}",
        )

    # 等高線を描画するためのグリッドを作成
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 各ガウス分布の等高線（1σ, 2σ）を描画
    for k in range(K):
        # マハラノビス距離を計算
        diff = grid_points - mu_final[k]
        inv_Sigma = np.linalg.pinv(Sigma_final[k])
        mahal_dist_sq = np.sum(diff @ inv_Sigma * diff, axis=1)
        mahal_dist_sq = mahal_dist_sq.reshape(xx.shape)

        # 1σ と 2σ の等高線を描画（マハラノビス距離が 1 と 4）
        ax.contour(
            xx,
            yy,
            mahal_dist_sq,
            levels=[1, 4],
            colors=colors[k % len(colors)],
            linestyles=["solid", "dashed"],
            linewidths=2,
            alpha=0.6,
        )

    ax.set_title(title)
    ax.set_ylabel("x2")
    ax.set_xlabel("x1")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.savefig(f"output/{title}.png", dpi=150, bbox_inches="tight")
    plt.close()


def AIC(log_likelihood: float, num_parameters: int) -> float:
    """
    AICを計算する.

    Args:
        log_likelihood (float): モデルの対数尤度
        num_parameters (int): モデルのパラメータ数

    Returns:
        float: AICの値
    """
    return 2 * num_parameters - 2 * log_likelihood


def BIC(log_likelihood: float, num_parameters: int, num_samples: int) -> float:
    """
    BICを計算する.

    Args:
        log_likelihood (float): モデルの対数尤度
        num_parameters (int): モデルのパラメータ数
        num_samples (int): データのサンプル数

    Returns:
        float: BICの値
    """
    return np.log(num_samples) * num_parameters - 2 * log_likelihood


def AIC_BIC_show(log_likelihoods: list, K: np.ndarray, N: int, title: str) -> None:
    """
    AICとBICを計算してグラフに表示する.

    Args:
        log_likelihoods (list): 各Kに対する対数尤度のリスト
        K (np.ndarray): クラスタ数の配列
        N (int): サンプル数
        title (str): 図のタイトルおよび保存ファイル名に使う文字列
    """
    aic_values = np.array(
        [AIC(log_likelihoods[i], 6 * K[i] - 1) for i in range(len(K))]
    )
    bic_values = np.array(
        [BIC(log_likelihoods[i], 6 * K[i] - 1, N) for i in range(len(K))]
    )
    min_aic_index = np.argmin(aic_values)
    min_bic_index = np.argmin(bic_values)
    plt.figure(figsize=(10, 6))
    plt.plot(K, aic_values, marker="o", label="AIC")
    plt.plot(K, bic_values, marker="s", label="BIC")
    # 最小値のインデックスに縦線を引く
    plt.axvline(
        x=K[min_aic_index],
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"Min AIC at K={K[min_aic_index]}",
    )
    plt.axvline(
        x=K[min_bic_index],
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Min BIC at K={K[min_bic_index]}",
    )
    plt.xlabel("Number of Clusters")
    plt.ylabel("Information Criterion")
    plt.title("Model Selection using AIC and BIC")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"output/{title}.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """
    サンプルデータを読み込み、EMアルゴリズムを用いてGMMにフィットさせる.

    Returns:
        None: CSVファイルを読み込み、各解析結果を `output/` 以下に保存する.
    """
    # CSVファイルからデータを読み込む
    data1_raw = pd.read_csv("../data/data1.csv", header=None)
    data2_raw = pd.read_csv("../data/data2.csv", header=None)
    data3_raw = pd.read_csv("../data/data3.csv", header=None)

    # data1
    print("Start processing data1.")
    show_data_scatter(data1_raw[0], data1_raw[1], "data1 scatter plot")
    # データをnp.ndarray化
    data1 = data1_raw.to_numpy()
    # EMアルゴリズム実施
    final_parameters_data1 = []
    final_log_likelihoods_data1 = []
    for K1 in range(1, 9):
        parameters_data1, log_likelihoods_data1 = loop_em_algorithm(data1, K1)
        final_parameters_data1.append(
            [
                parameters_data1.mu[-1],
                parameters_data1.Sigma[-1],
                parameters_data1.pi[-1],
            ]
        )
        final_log_likelihoods_data1.append(log_likelihoods_data1[-1])
        show_log_likelihood(log_likelihoods_data1, f"data1 log likelihood (K={K1})")
        show_data_scatter_and_GMM(
            data1, parameters_data1, K1, f"data1 GMM clustering result (K={K1})"
        )
    AIC_BIC_show(
        final_log_likelihoods_data1, np.arange(1, 9), len(data1), "data1 AIC_BIC"
    )

    # data2
    print("Start processing data2.")
    show_data_scatter(data2_raw[0], data2_raw[1], "data2 scatter plot")
    data2 = data2_raw.to_numpy()
    # EMアルゴリズム実施
    final_parameters_data2 = []
    final_log_likelihoods_data2 = []
    for K2 in range(1, 9):
        parameters_data2, log_likelihoods_data2 = loop_em_algorithm(data2, K2)
        final_parameters_data2.append(
            [
                parameters_data2.mu[-1],
                parameters_data2.Sigma[-1],
                parameters_data2.pi[-1],
            ]
        )
        final_log_likelihoods_data2.append(log_likelihoods_data2[-1])
        show_log_likelihood(log_likelihoods_data2, f"data2 log likelihood (K={K2})")
        show_data_scatter_and_GMM(
            data2, parameters_data2, K2, f"data2 GMM clustering result (K={K2})"
        )
    AIC_BIC_show(
        final_log_likelihoods_data2, np.arange(1, 9), len(data2), "data2 AIC_BIC"
    )
    # data3
    print("Start processing data3.")
    show_data_scatter(data3_raw[0], data3_raw[1], "data3 scatter plot")
    data3 = data3_raw.to_numpy()
    # EMアルゴリズム実施
    final_parameters_data3 = []
    final_log_likelihoods_data3 = []
    for K3 in range(1, 9):
        parameters_data3, log_likelihoods_data3 = loop_em_algorithm(data3, K3)
        final_parameters_data3.append(
            [
                parameters_data3.mu[-1],
                parameters_data3.Sigma[-1],
                parameters_data3.pi[-1],
            ]
        )
        final_log_likelihoods_data3.append(log_likelihoods_data3[-1])
        show_log_likelihood(log_likelihoods_data3, f"data3 log likelihood (K={K3})")
        show_data_scatter_and_GMM(
            data3, parameters_data3, K3, f"data3 GMM clustering result (K={K3})"
        )
    AIC_BIC_show(
        final_log_likelihoods_data3, np.arange(1, 9), len(data3), "data3 AIC_BIC"
    )


if __name__ == "__main__":
    main()
