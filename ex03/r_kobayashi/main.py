"""散布図、等高線、尤度の折れ線グラフ、およびパラメータを出力."""

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

try:
    arg = sys.argv[1]
except IndexError:
    print("引数が指定されていません。1, 2, 3 のいずれかを指定してください。")
    sys.exit(1)

if arg == "1":
    txt_dir = "../../ex3/data/data1.csv"
    csv_name = "gamma_data1"
    dat_name = "params_data1"
    fig_name = "plot_data1"
    fig_gap_name = "log_likelihood_gap_data1"
    k = 3

elif arg == "2":
    txt_dir = "../../ex3/data/data2.csv"
    csv_name = "gamma_data2"
    dat_name = "params_data2"
    fig_name = "plot_data2"
    fig_gap_name = "log_likelihood_gap_data2"
    k = 4

elif arg == "3":
    txt_dir = "../../ex3/data/data3.csv"
    csv_name = "gamma_data3"
    dat_name = "params_data3"
    fig_name = "plot_data3"
    fig_gap_name = "log_likelihood_gap_data3"
    k = 2

else:
    print(f"不正な引数です: {arg}。1, 2, 3 のいずれかを指定してください。")
    sys.exit(1)


with open(txt_dir) as f:
    reader = csv.reader(f)
    l = [row for row in reader]
    for i in range(len(l)):
        for j in range(len(l[i])):
            l[i][j] = float(l[i][j])
l = np.array(l)


def calc(x, mu, sigma_inv, sigma_det):
    """D次元のガウス分布の計算.

    Args:
        x (ndarray of shape (D,)): 元データ
        mu (ndarray of shape (D,)): 分布の平均ベクトル
        sigma_inv (ndarray of shape (D,D)): 共分散行列の逆行列
        sigma_det (float): 共分散行列の行列式

    Returns:
        float: xにおけるガウス分布の確率密度値

    """
    D = x.shape[0]
    exp = -0.5 * (x - mu).T @ sigma_inv.T @ (x - mu)
    denomin = np.sqrt(sigma_det) * (np.sqrt(2 * np.pi) ** D)
    return np.exp(exp) / denomin


def gauss(X, mu, sigma):
    """各データにおけるガウス分布の確率密度のリスト化.

    Args:
        x (ndarray of shape (N,D)): 元データ
        mu (ndarray of shape (D,)): 分布の平均ベクトル
        sigma (ndarray of shape (D,D)): 共分散行列

    Returns:
        ndarray of shape (N,): 各データにおけるガウス分布の確率密度のリスト
    """
    output = np.array([])
    eps = np.spacing(1)
    Eps = eps * np.eye(sigma.shape[0])
    sigma_inv = la.inv(sigma)
    sigma_det = la.det(sigma)
    N = X.shape[0]
    for i in range(N):
        output = np.append(output, calc(X[i], mu, sigma_inv, sigma_det))
    return output


def mix_gauss(X, Mu, Sigma, Pi):
    """混合ガウスモデルの導出.

    Args:
        x (ndarray of shape (N,D)): 元データ
        mu (ndarray of shape (k,D)): 分布の平均ベクトル
        sigma (ndarray of shape (k,D,D)): 共分散行列
        Pi (ndarray of shape (k,)): 混合係数

    Returns:
        output (ndarray of shape (K, N): 各クラスタの確率密度
        out_sum (ndarray of shape (1, N)): 混合分布の確率密度総和

    """
    k = len(Mu)
    output = np.array([Pi[i] * gauss(X, Mu[i], Sigma[i]) for i in range(k)])
    return output, np.sum(output, 0)[None, :]


def setInitial(X, k):
    """初期化.

    Args:
        X (ndarray of shape (N,D)): 元データ
        k (int): 想定クラスタ数

    Returns:
        Mu (ndarray of shape (k,D)): 分布の平均ベクトル初期値
        Sigma (ndarray of shape (k,D,D)): 共分散行列初期値(単位行列の集合)
        Pi (ndarray of shape (k,)): 混合係数初期値

    """
    D = X.shape[1]  # 列数取得
    Mu = np.random.randn(k, D)  # 平均ベクトル初期化
    Sigma = np.array([np.eye(D) for i in range(k)])  # 共分散初期化
    Pi = np.array([1 / k for i in range(k)])  # 混合重み初期化
    return Mu, Sigma, Pi


def log_likelihood(X, Mu, Sigma, Pi):
    """尤度の対数化.

    Args:
        X (ndarray of shape (N,D)): 元データ
        Mu (ndarray of shape (k,D)): 分布の平均ベクトル
        Sigma (ndarray of shape (k,D,D)): 共分散行列
        Pi (ndarray of shape (k,)): 混合係数

    Returns:
        float: 対数化された尤度の合計
    """
    K = Mu.shape[0]
    D = X.shape[1]
    N = X.shape[0]
    _, out_sum = mix_gauss(X, Mu, Sigma, Pi)
    logs = np.array([np.log(out_sum[0][n]) for n in range(N)])
    return np.sum(logs)


def EM(X, k, Mu, Sigma, Pi, thr):
    """EMアルゴリズムの実行.

    Args:
        X (ndarray of shape (N,D)): 元データ
        k (int): 想定クラスタ数
        Mu (ndarray of shape (k,D)): 分布の平均ベクトル
        Sigma (ndarray of shape (k,D,D)): 共分散行列
        Pi (ndarray of shape (k,)): 混合係数
        thr (float): 閾値

    Returns:
        n_iter (int): 収束までに要した EM の反復回数
        log_list (ndarray of shape (T,)): 対数尤度の一覧(Tは反復回数)数
        Mu (ndarray of shape (k, D)): 分布の平均ベクトル
        Sigma (ndarray of shape (k,D,D)): 共分散行列
        Pi (ndarray of shape (k,)): 混合係数
        gamma (ndarray of shape (k, N)): 負担率

    """
    K = Mu.shape[0]
    D = X.shape[1]
    N = X.shape[0]
    log_list = np.array([])
    log_list = np.append(log_list, log_likelihood(X, Mu, Sigma, Pi))
    count = 0
    while True:
        # Eステップ: パラメータを固定し各データが各クラスタに所属する期待値𝛾(𝑧_𝑛𝑘)を求める
        out_com, out_sum = mix_gauss(X, Mu, Sigma, Pi)
        gamma = out_com / out_sum

        # Mステップ: 負担率𝛾(𝑧_𝑛𝑘)を固定してパラメータ（𝜇, 𝛴, 𝜋）を更新

        # 1. クラスタkに割り当てられた実効的なデータ数𝑁_𝑘
        N_k = np.sum(gamma, 1)[:, None]

        # 2. 平均ベクトル𝜇_𝑘の更新
        Mu = (gamma @ X) / N_k

        # 3. 共分散行列Σ_𝑘の更新
        sigma_list = np.zeros((N, K, D, D))
        for k in range(K):
            for n in range(N):
                sigma_com = (
                    gamma[k][n] * (X[n] - Mu[k])[:, None] @ (X[n] - Mu[k])[None, :]
                )
                sigma_list[n][k] = sigma_com
        Sigma = np.sum(sigma_list, 0) / N_k[:, None]

        # 4. 混合係数𝜋_𝑘の更新
        Pi = N_k / N

        # 更新したパラメータの記録
        log_list = np.append(log_list, log_likelihood(X, Mu, Sigma, Pi))

        # 対数尤度の判定 (閾値以下で終了)
        if np.abs(log_list[count] - log_list[count + 1]) < thr:
            return count + 1, log_list, Mu, Sigma, Pi, gamma
        else:
            print(
                "Previous log-likelihood gap:"
                + str(np.abs(log_list[count] - log_list[count + 1]))
            )
            count += 1


thr = 0.01
# k = 2
Mu, Sigma, Pi = setInitial(l, k)
n_iter, log_list, Mu, Sigma, Pi, gamma = EM(l, k, Mu, Sigma, Pi, thr)
print("Iteration:" + str(n_iter))
print("log-likelihood:" + str(log_list))

# params = np.array([Mu.ravel(), Sigma.ravel(), Pi.ravel()])
# params = [Mu.ravel(), Sigma.ravel(), Pi.ravel()]
params = []

# Mu（平均）
for i in range(k):
    params.append([f"Mu_{i + 1}"] + Mu[i].tolist())

# Sigma（共分散）
for i in range(k):
    params.append([f"Sigma_{i + 1}"] + Sigma[i].ravel().tolist())

# Pi（混合比）
params.append(["Pi"] + Pi.tolist())

index = np.argmax(gamma, 0)

cm = plt.get_cmap("tab10")
fig = plt.figure()
ax = fig.add_subplot(111)
N = l.shape[0]

for n in range(N):
    ax.plot([l[n][0]], [l[n][1]], "o", color=cm(index[n]), ms=1.5)
# ax.view_init(elev=30, azim=45)


# 各ガウス分布の平均をプロット
plt.scatter(Mu[:, 0], Mu[:, 1], s=100, c="red", edgecolors="black")
# 各ガウス分布の等高線をプロット
x = np.linspace(l[:, 0].min(), l[:, 0].max(), 100)
y = np.linspace(l[:, 1].min(), l[:, 1].max(), 100)

X_grid, Y_grid = np.meshgrid(x, y)
grid = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

_, Z = mix_gauss(grid, Mu, Sigma, Pi)
Z = Z.reshape(X_grid.shape)
plt.contour(
    X_grid,
    Y_grid,
    Z,
    levels=20,
    linewidths=1,
    colors="green",
    linestyles="dashed",
    alpha=0.5,
)
plt.savefig(fig_name)
plt.show()


with open(csv_name, "w") as file:
    writer = csv.writer(file, lineterminator="\n")
    writer.writerows(gamma.T)

with open(dat_name, "w") as file:
    writer = csv.writer(file, lineterminator="\n\n")
    writer.writerows(params)

log_gap_list = log_list[1:] - log_list[:-1]
ax = plt.gca()
ax.set_yscale("log")
plt.title("log-likelihood-gap")
plt.xlabel("X", fontsize=18)
plt.ylabel("Y", fontsize=18)
plt.grid(which="both")
plt.hlines(
    [1e-2], 0, len(log_gap_list) - 1, color="blue", linestyles="dashed", linewidth=1
)
plt.plot(log_gap_list)
plt.savefig(fig_gap_name)
plt.show()
print("log-likelihood:" + str(log_gap_list))
