"""The answer of Ex03 by Hagiwara Futa."""

from pathlib import Path

import japanize_matplotlib  # noqa
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sys

# # fmt: off
# from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
#                              precision_recall_curve, roc_curve)

# # fmt: on


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


def gmm():
    """Solve assignments."""
    # データ読み込み(Ex3-1)
    v1 = np.loadtxt("../data/data1.csv", delimiter=",")
    v1x = v1[:, 0]
    v1y = v1[:, 1]
    N_v1 = v1x.size

    v2 = np.loadtxt("../data/data2.csv", delimiter=",")
    v2x = v2[:, 0]
    v2y = v2[:, 1]
    N_v2 = v2x.size

    v3 = np.loadtxt("../data/data3.csv", delimiter=",")
    v3x = v3[:, 0]
    v3y = v3[:, 1]
    N_v3 = v3x.size

    # プロット
    plt.figure()
    plt.scatter(v1x, v1y)
    plt.xlabel("Axis1")
    plt.ylabel("Axis2")
    plt.title("data1.csv")
    plt.grid(True)
    Path("outputs").mkdir(exist_ok=True)
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

    # GMMの実装(Ex3-2)
    # 初期値の設定
    args = sys.argv
    K = int(args[1])
    np.random.seed(0)
    mu = v1[np.random.choice(N_v1, K, replace=False)]  # 各クラスの平均ベクトル
    sigma = np.array(
        [np.eye(2) for _ in range(K)]
    )  # 各クラスタに対する分散共分散行列 クラス内の分散とクラス間の共分散が並ぶ
    pi = np.ones(K) / K  # 混合係数 (各正規分布がGMMで占める割合)
    gamma = np.zeros((N_v1, K))  # 負担率 (各データが各クラスタに属する確率)
    threshold = 0.01  # 閾値
    like_old = 0.0

    # EMアルゴリズム
    for i in range(10000):
        # Eステップ (負担率の計算)
        # 分子
        for j in range(K):
            for n in range(N_v1):
                gamma[n, j] = pi[j] * gauss(v1[n], mu[j], sigma[j])
        # 正規化 (分母)
        # 行方向に足し合わせることで各データに対するクラスタの合計が得られる
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        # Mステップ (パラメータの更新)
        # クラスタkに割り当てられた実効的なデータ数Nkを更新
        Nk = np.sum(gamma, axis=0)

        # 平均ベクトルの更新
        # クラス毎に平均を取っている
        mu = (gamma.T @ v1) / (Nk[:, np.newaxis] + 1e-12)

        # 共分散行列の更新
        # クラス毎の処理を行う
        for k in range(K):
            v1_mu = v1 - mu[k]
            sigma[k] = (v1_mu.T @ (gamma[:, k][:, None] * v1_mu)) / (Nk[k] + 1e-12)
            # 正則化
            sigma[k] += 1e-6 * np.eye(2)
        # 混合係数の更新
        pi = Nk / N_v1

        # 対数尤度の計算
        like = 0.0
        for n in range(N_v1):
            tmp = 0.0
            for k in range(K):
                tmp += pi[k] * gauss(v1[n], mu[k], sigma[k])
            like += np.log(tmp + 1e-12)
        # 閾値以下になったら収束とみなす
        if abs(like - like_old) < threshold:
            print("converged in the", i, "th iteration.")
            break
        # 対数尤度の保存
        like_old = like


if __name__ == "__main__":
    main()
