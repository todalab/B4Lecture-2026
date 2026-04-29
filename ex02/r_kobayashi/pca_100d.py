"""_summary_

Perform principal component analysis on the 100-dimensional input data, plot the results on a two-dimensional plane, and output a table showing the changes in cumulative contribution rates.

"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path_100d = "../../ex2/data/pca_100d.csv"

with open(path_100d) as f:
    sample1 = np.loadtxt(path_100d, delimiter=",")


def pca(X, n_components=2):
    mu = X.mean(axis=0)  # 元データの平均点（2次元）
    X = X - mu  # 平均中心化
    cov = np.cov(X, rowvar=False)  # 共分散行列の導出
    l, v = np.linalg.eig(cov)  # 固有値l, 固有ベクトルvの導出
    l_index = np.argsort(l)[::-1]  # 固有値の大きい順にソート

    # print(f"l = {l}")
    # print(f"v = {v}")
    evr_list = l / np.sum(l)  # 寄与率の導出
    # print(f"evr: {evr_list}")

    cev_list = list(itertools.accumulate(evr_list))
    cev_list_round = [float(round(cev_list[n], 5)) for n in range(len(cev_list))]

    index = []
    for i, value in enumerate(cev_list_round):
        index.append(i + 1)
        print(f"{i + 1}: {value}")
        if value >= 0.9:
            print(f"{value} >= 0.9, so {i + 1} is minimal dimension")
            cev_dict_result = dict(zip(index, cev_list_round[: i + 1]))
            print(cev_dict_result)
            df = pd.DataFrame.from_dict(
                cev_dict_result, orient="index", columns=["value"]
            )
            index_col = df.index.tolist()
            values = df["value"].tolist()

            cell_text = list(zip(index_col, values))
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.axis("off")
            ax.axis("tight")
            ax.table(
                cellText=cell_text,
                colLabels=["dim", "value"],
                bbox=[0, 0, 1, 1],
            )
            plt.savefig("pca_100d_cev.png")
            break

    # cev = sum(evr_list[:n_components])
    # print(f"cev: {cev} (n = {n_components})")  # n_components個の累積寄与率の導出

    v_ = v[:, l_index]
    components = v_[:, :n_components]  # n_components個の固有ベクトルを取得
    # print(f"components = {components}")
    v1 = components[:, 0]  # 第1主成分
    v2 = components[:, 1]  # 第2主成分
    T = np.dot(X, components)  # 固有ベクトルをかけて次元圧縮
    print(f"T.shape={T.shape}")

    x2_input = T[:, 0]
    y2_input = T[:, 1]
    t = np.linspace(-20, 20, 100)  # 定義域
    line_v1 = np.outer(t, v1) + mu
    line_v2 = np.outer(t, v2) + mu
    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.scatter(x2_input, y2_input, color="black", s=10)
    ax2.plot(line_v1[:, 0], line_v1[:, 1], color="red")
    ax2.plot(line_v2[:, 0], line_v2[:, 1], color="green")
    ax2.set_aspect("equal", adjustable="box")
    plt.savefig("pca_100d.png")
    return T


if __name__ == "__main__":
    T = pca(sample1)
    # print(T)
