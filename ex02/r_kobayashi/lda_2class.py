"""_summary_

Output the scatter plot and LDA axes from lda_2class.csv onto a two-dimensional plane.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path_2class = "../../ex2/data/lda_2class.csv"

with open(path_2class) as f:
    sample1 = np.loadtxt(path_2class, delimiter=",", skiprows=1)

# 散布図用の点の座標抽出
x1_input = sample1[:, 0]
x2_input = sample1[:, 1]
label_input = sample1[:, 2]

df = pd.read_csv(path_2class)
df_0 = df[df["label"] <= 0.5]
df_1 = df[df["label"] > 0.5]

df_0 = df_0.drop("label", axis=1).to_numpy()
df_1 = df_1.drop("label", axis=1).to_numpy()
df_0_mean = np.mean(df_0, axis=0)
df_1_mean = np.mean(df_1, axis=0)

print(f"df_0_mean = {df_0_mean}")
print(f"df_1_mean = {df_1_mean}")

SB = np.outer((df_0_mean - df_1_mean), (df_0_mean - df_1_mean))

# print(np.stack([df_0_mean, df_0_mean], 1).shape)
# print(type(df_0), type(df_0_mean))
# print(df_0.shape, df_0_mean.shape)
# print((df_0 - df_0_mean).shape)
SW = (df_0 - df_0_mean).T @ (df_0 - df_0_mean) + (df_1 - df_1_mean).T @ (
    df_1 - df_1_mean
)

print(f"SB = \n{SB}")
print(f"SW = \n{SW}")
# print(type(SB), type(SW))
# print(SB.shape, SW.shape)

# 固有値の導出
w, v = np.linalg.eigh(np.linalg.pinv(SW) @ SB)

# 全体平均の導出
mu = (df_0_mean + df_1_mean) / 2

t = np.linspace(-5, 5, 100)
line_v1 = np.outer(t, v[:, -1]) + mu

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# 散布図
ax.scatter(x1_input, x2_input, c=label_input, label="data")

# LDA の判別軸
ax.plot(line_v1[:, 0], line_v1[:, 1], color="red", label="LDA axis")

# タイトル・ラベル・凡例・グリッド
ax.set_title("LDA: Discriminant Axis")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
ax.grid(True)

ax.set_aspect("equal", adjustable="box")
plt.savefig("lda_2class.png")


df_nolabel = df.drop("label", axis=1).to_numpy()


def acc(df_0, df_1, threshold):
    """_summary_

    Args:
        df_0 : x0
        df_1 : x1
        threshold : threshold

    Returns:
        acc : accuracy
    """
    acc = (
        np.count_nonzero((np.dot(df_0, v[:, -1]) < threshold))
        + np.count_nonzero((np.dot(df_1, v[:, -1]) >= threshold))
    ) / df.shape[0]
    return acc


print(f"accuracy (threshold = 0) = {acc(df_0=df_0, df_1=df_1, threshold=0)}")
