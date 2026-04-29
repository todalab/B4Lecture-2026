"""_summary_

Perform principal component analysis on the three-dimensional input data, plot the results in three-dimensional space, and then reduce the dimensions to two-dimensional space.

"""

import matplotlib.pyplot as plt
import numpy as np

path_3d = "../../ex2/data/pca_3d.csv"

with open(path_3d) as f:
    sample1 = np.loadtxt(path_3d, delimiter=",")

# print(sample1.shape)
# print(sample1.shape[1])

x_input = sample1[:, 0]
y_input = sample1[:, 1]
z_input = sample1[:, 2]

# print(x_input.shape)
# print(y_input.shape)

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.scatter(x_input, y_input, color="black", s=10)


def pca(X, n_components=3):
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
    v3 = components[:, 2]  # 第3主成分
    t = np.linspace(-2, 2, 100)  # 定義域
    line_v1 = np.outer(t, v1) + mu
    line_v2 = np.outer(t, v2) + mu
    line_v3 = np.outer(t, v3) + mu
    T = np.dot(X, components)  # 固有ベクトルをかけて次元圧縮
    print(f"T.shape={T.shape}")

    fig = plt.figure()
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.scatter(x_input, y_input, z_input, color="black", s=10)
    ax3.plot(line_v1[:, 0], line_v1[:, 1], line_v1[:, 2], color="red")
    ax3.plot(line_v2[:, 0], line_v2[:, 1], line_v2[:, 2], color="green")
    ax3.plot(line_v3[:, 0], line_v3[:, 1], line_v3[:, 2], color="blue")
    ax3.set_aspect("equal", adjustable="box")

    x2_input = T[:, 0]
    y2_input = T[:, 1]
    # fig = plt.figure()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(x2_input, y2_input, color="black", s=10)
    ax2.plot(line_v1[:, 0], line_v1[:, 1], color="red")
    ax2.plot(line_v2[:, 0], line_v2[:, 1], color="green")
    ax2.set_aspect("equal", adjustable="box")
    plt.savefig("pca_3d.png")
    return T


if __name__ == "__main__":
    T = pca(sample1)
    # print(T)
