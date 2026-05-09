"""The answer of Ex03 by Hagiwara Futa."""

from pathlib import Path

import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import sys


def main():
    """Run main function."""
    gmm()


def gauss(x, mu, sigma):
    """Transform using a Gaussian function."""
    # 各データが各々のクラスタから出る確率密度
    D = x.shape[0]
    det = np.linalg.det(sigma)  # 共分散行列の行列式 |Σ|
    inv = np.linalg.inv(sigma)  # 共分散行列の逆行列
    return np.exp(-0.5 * (x - mu) @ inv @ (x - mu)) / np.sqrt((2 * np.pi) ** D * det)


def em_algorithm(data, K):
    """Run EM algorithm for Gaussian Mixture Model."""
    N = data.shape[0]

    # 初期値の設定
    np.random.seed(0)
    mu = data[np.random.choice(N, K, replace=False)]  # 各クラスの平均ベクトル
    sigma = np.array(
        [np.eye(2) for _ in range(K)]
    )  # 各クラスタに対する分散共分散行列 クラス内の分散とクラス間の共分散が並ぶ
    pi = np.ones(K) / K  # 混合係数 (各正規分布がGMMで占める割合)
    gamma = np.zeros((N, K))  # 負担率 (各データが各クラスタに属する確率)

    max_iter = 10000
    threshold = 1e-4
    like_old = 0.0
    likes = []

    # EMアルゴリズム
    for i in range(max_iter):
        # Eステップ (負担率の計算)
        # 分子
        for j in range(K):
            for n in range(N):
                gamma[n, j] = pi[j] * gauss(data[n], mu[j], sigma[j])
        # 正規化 (分母)
        # 行方向に足し合わせることで各データに対するクラスタの合計が得られる
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        # Mステップ (パラメータの更新)
        # クラスタkに割り当てられた実効的なデータ数Nkを更新
        Nk = np.sum(gamma, axis=0)

        # 平均ベクトルの更新
        # クラス毎に平均を取っている
        mu = (gamma.T @ data) / (Nk[:, np.newaxis] + 1e-12)

        # 共分散行列の更新
        # クラス毎の処理を行う
        for k in range(K):
            data_mu = data - mu[k]
            sigma[k] = (data_mu.T @ (gamma[:, k][:, None] * data_mu)) / (Nk[k] + 1e-12)
            # 正則化
            sigma[k] += 1e-6 * np.eye(2)
        # 混合係数の更新
        pi = Nk / N

        # 対数尤度の計算
        like = 0.0
        for n in range(N):
            tmp = 0.0
            for k in range(K):
                tmp += pi[k] * gauss(data[n], mu[k], sigma[k])
            like += np.log(tmp + 1e-12)

        likes.append(like)
        # 閾値以下になったら収束とみなす
        if abs(like - like_old) < threshold:
            print(f"K={K}:converged in the {i+1}th iteration.")
            break
        # 対数尤度の保存
        like_old = like

    return mu, sigma, pi, gamma, likes


def gmm_plot(data, K, likes, mu, sigma, gamma, name):
    """Plot results of GMM."""
    # 対数尤度の推移グラフ
    plotx = np.arange(1, len(likes) + 1)
    plt.xticks(np.arange(1, len(likes) + 1, 1))
    plt.xlabel("Iteration")
    plt.ylabel("Likelihood")
    plt.plot(plotx, likes, "-o", markersize=5)
    plt.grid(True)
    plt.tick_params(labelsize=7)
    plt.title(f"data{name} Log-Likelihood")
    plt.savefig(f"outputs/data{name}_likelihood.png")
    plt.close()

    # GMMによるソフトクラスタリング
    # 各データ点を，負担率が最大のクラスターで色分け
    # 最も高い負担率のクラスタをラベリング
    labels = np.argmax(gamma, axis=1)
    # 色分けした散布図を得る
    okabe_ito = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    ]
    plt.figure()
    plt.rcParams["axes.axisbelow"] = True
    # 散布図のプロット
    for k in range(K):
        mask = labels == k  # 適したもののみを抽出するブールインデックス
        plt.scatter(
            data[mask, 0],
            data[mask, 1],
            color=okabe_ito[k],
            s=20,
            label=f"Cluster {k+1}",
            zorder=1,
        )

    # 平均のプロット
    for k in range(K):
        plt.scatter(
            mu[k, 0],
            mu[k, 1],
            color=okabe_ito[k],
            marker="^",
            s=100,
            edgecolors="black",
            linewidths=0.8,
            zorder=2,
        )

    # 等高線 (1σ, 2σ) のプロット
    x = np.linspace(data[:, 0].min() - 1, data[:, 0].max() + 1, 100)
    y = np.linspace(data[:, 1].min() - 1, data[:, 1].max() + 1, 100)
    gx, gy = np.meshgrid(x, y)
    # 2次元リストを1次元へ変換
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)

    D = mu.shape[1]
    for k in range(K):
        # グリッド各点の確率密度を計算、元の2次元に戻す
        pdf = np.array([gauss(p, mu[k], sigma[k]) for p in grid]).reshape(gx.shape)
        # 1σ, 2σ に対応するガウス分布の確率密度値
        det = np.linalg.det(sigma[k])
        sigma1 = np.exp(-0.5 * 1**2) / np.sqrt((2 * np.pi) ** D * det)  # 1σ
        sigma2 = np.exp(-0.5 * 2**2) / np.sqrt((2 * np.pi) ** D * det)  # 2σ
        # 2σ〜1σ
        plt.contourf(
            gx,
            gy,
            pdf,
            levels=[sigma2, sigma1],
            colors=[okabe_ito[k]],
            alpha=0.15,
        )
        # 1σ以内
        plt.contourf(
            gx,
            gy,
            pdf,
            levels=[sigma1, pdf.max()],
            colors=[okabe_ito[k]],
            alpha=0.4,
        )
        # 境界線
        plt.contour(
            gx,
            gy,
            pdf,
            levels=[sigma2, sigma1],
            colors=[okabe_ito[k]],
            linestyles=["dashed", "solid"],
            linewidths=1.2,
        )

    plt.xlabel("Axis1")
    plt.ylabel("Axis2")
    plt.title(f"data{name} GMM Clustering (K={K})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"outputs/data{name}_gmm.png")
    plt.close()


def aic_bic(data, name):
    """Calculate AIC and BIC."""
    N, D = data.shape  # データ数・次元数

    aic_list = []
    bic_list = []
    k_list = list(range(1, 9))

    for K in k_list:
        # EMアルゴリズムでパラメータ推定
        mu, sigma, pi, gamma, likes = em_algorithm(data, K)

        # 収束後の対数尤度 (最後の値)
        log_likelihood = likes[-1]

        # GMMの自由パラメータ数を計算
        # 混合係数+平均+共分散
        n_params = (K - 1) + K * D + K * D * (D + 1) // 2

        # AIC(赤池情報量基準)
        # AIC = -2 × log_likelihood + 2 × パラメータ数
        aic = -2 * log_likelihood + 2 * n_params
        # BIC(ベイズ情報量基準)
        # BIC = -2 × log_likelihood + log(N) × パラメータ数
        # BICはデータ数Nが多いほどパラメータへのペナルティが大きくなる
        bic = -2 * log_likelihood + np.log(N) * n_params

        aic_list.append(aic)
        bic_list.append(bic)

    # AIC・BICをプロット
    fig, ax = plt.subplots()
    ax.plot(k_list, aic_list, "-o", label="AIC", color="tab:blue")
    best_aic = aic_list.index(min(aic_list)) + 1
    plt.axvline(
        x=best_aic,
        color="tab:blue",
        linestyle="--",
        label=f"AIC min (K={best_aic})",
    )
    ax.plot(k_list, bic_list, "-s", label="BIC", color="tab:orange")
    best_bic = bic_list.index(min(bic_list)) + 1
    plt.axvline(
        x=best_bic,
        color="tab:orange",
        linestyle="--",
        label=f"BIC min (K={best_bic})",
    )

    # AICラベル（左下）
    for k, aic in zip(k_list, aic_list):
        ax.annotate(
            f"{aic:.1f}",
            xy=(k, aic),
            xytext=(-4, -6),
            textcoords="offset points",
            fontsize=7,
            color="tab:blue",
            ha="right",
        )

    # BICラベル（右上）
    for k, bic in zip(k_list, bic_list):
        ax.annotate(
            f"{bic:.1f}",
            xy=(k, bic),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=7,
            color="tab:orange",
            ha="left",
        )

    plt.xlabel("K (クラスタ数)")
    plt.ylabel("Score")
    plt.title(f"data{name} AIC / BIC")
    plt.xticks(k_list)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"outputs/data{name}_aic_bic.png")
    plt.close()


def gmm():
    """Solve assignments."""
    # データ読み込み(Ex3-1)
    v1 = np.loadtxt("../data/data1.csv", delimiter=",")
    v1x = v1[:, 0]
    v1y = v1[:, 1]

    v2 = np.loadtxt("../data/data2.csv", delimiter=",")
    v2x = v2[:, 0]
    v2y = v2[:, 1]

    v3 = np.loadtxt("../data/data3.csv", delimiter=",")
    v3x = v3[:, 0]
    v3y = v3[:, 1]

    # プロット(Ex3-1)
    Path("outputs").mkdir(exist_ok=True)

    plt.figure()
    plt.rcParams["axes.axisbelow"] = True
    plt.scatter(v1x, v1y)
    plt.xlabel("Axis1")
    plt.ylabel("Axis2")
    plt.title("data1.csv")
    plt.grid(True)
    plt.savefig("outputs/data1_scat.png")
    plt.close()

    plt.figure()
    plt.scatter(v2x, v2y)
    plt.xlabel("Axis1")
    plt.ylabel("Axis2")
    plt.title("data2.csv")
    plt.grid(True)
    plt.savefig("outputs/data2_scat.png")
    plt.close()

    plt.figure()
    plt.scatter(v3x, v3y)
    plt.xlabel("Axis1")
    plt.ylabel("Axis2")
    plt.title("data3.csv")
    plt.grid(True)
    plt.savefig("outputs/data3_scat.png")
    plt.close()

    # GMM(Ex3-2)
    args = sys.argv
    mu1, sigma1, pi1, gamma1, likes1 = em_algorithm(v1, int(args[1]))
    mu2, sigma2, pi2, gamma2, likes2 = em_algorithm(v2, int(args[2]))
    mu3, sigma3, pi3, gamma3, likes3 = em_algorithm(v3, int(args[3]))

    # GMM結果プロット(Ex3-3)
    gmm_plot(v1, int(args[1]), likes1, mu1, sigma1, gamma1, 1)
    gmm_plot(v2, int(args[2]), likes2, mu2, sigma2, gamma2, 2)
    gmm_plot(v3, int(args[3]), likes3, mu3, sigma3, gamma3, 3)

    # AIC,BIC(Ex3-4)
    aic_bic(v1, 1)
    aic_bic(v2, 2)
    aic_bic(v3, 3)


if __name__ == "__main__":
    main()
