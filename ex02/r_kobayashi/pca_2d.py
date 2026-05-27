"""_summary_.

Perform principal component analysis on the two-dimensional.

"""

import matplotlib.pyplot as plt
import numpy as np

path_2d = "../../ex2/data/pca_2d.csv"

with open(path_2d) as f:
    sample1 = np.loadtxt(path_2d, delimiter=",")

# print(sample1.shape)
# print(sample1.shape[1])

x_input = sample1[:, 0]
y_input = sample1[:, 1]

# print(x_input.shape)
# print(y_input.shape)

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.scatter(x_input, y_input, color="black", s=10)


def pca(X, n_components=2):
    """_summary_.

    Args:
        X: Original data
        n_components: Number of eigenvectors to extract

    Returns:
        T: Dimensionally compressed array
    """
    mu = X.mean(axis=0)  # 元データの平均点（2次元）
    X = X - mu  # 平均中心化
    cov = np.cov(X, rowvar=False)  # 共分散行列の導出
    l, v = np.linalg.eig(cov)  # 固有値l, 固有ベクトルvの導出
    l_index = np.argsort(l)[::-1]  # 固有値の大きい順にソート

    print(f"l = {l}")
    print(f"v = {v}")
    evr_list = l / np.sum(l)  # 寄与率の導出
    print(f"evr: {evr_list}")

    cev = sum(evr_list[:n_components])
    print(f"cev: {cev} (n = {n_components})")  # n_components個の累積寄与率の導出

    v_ = v[:, l_index]
    components = v_[:, :n_components]  # n_components個の固有ベクトルを取得
    print(f"components = {components}")
    v1 = components[:, 0]  # 第1主成分
    v2 = components[:, 1]  # 第2主成分
    t = np.linspace(-2, 2, 100)  # 定義域
    line_v1 = np.outer(t, v1) + mu
    line_v2 = np.outer(t, v2) + mu
    ax.plot(line_v1[:, 0], line_v1[:, 1], color="red")
    ax.plot(line_v2[:, 0], line_v2[:, 1], color="green")
    ax.set_aspect("equal", adjustable="box")
    plt.savefig("pca_2d.png")
    T = np.dot(X, components)  # 固有ベクトルをかけて次元圧縮
    return T


if __name__ == "__main__":
    T = pca(sample1)
    # print(T)
