import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "sans-serif"


# CSVファイルからデータを読み込む
sample2d_1 = pd.read_csv("../data/sample2d_1.csv")
sample2d_2 = pd.read_csv("../data/sample2d_2.csv")
sample3d = pd.read_csv("../data/sample3d.csv")
sample_logistic = pd.read_csv("../data/sample_logistic.csv")

# データを可視化
# sample2d_1
print(sample2d_1.describe())
sample2d_1_xlabel, sample2d_1_ylable = sample2d_1.columns

plt.scatter(sample2d_1["x"], sample2d_1["y"], label="data")
plt.legend(loc="upper left")
plt.title("sample2d_1 scatter plot")
plt.ylabel(sample2d_1_ylable)
plt.xlabel(sample2d_1_xlabel)
plt.show()

# sample2d_2
print(sample2d_2.describe())
sample2d_2_xlabel, sample2d_2_ylable = sample2d_2.columns

plt.scatter(sample2d_2["x"], sample2d_2["y"], label="data")
plt.legend(loc="upper left")
plt.title("sample2d_2 scatter plot")
plt.ylabel(sample2d_2_ylable)
plt.xlabel(sample2d_2_xlabel)
plt.show()

# sample3d
print(sample3d.describe())
sample3d_xlabel, sample3d_ylable, sample3d_zlable = sample3d.columns

fig_sample3d = plt.figure()
ax_sample3d = fig_sample3d.add_subplot(111, projection="3d")
ax_sample3d.scatter(
    sample3d[sample3d_xlabel],
    sample3d[sample3d_ylable],
    sample3d[sample3d_zlable],
    label="data",
)
ax_sample3d.legend(loc="upper left")
ax_sample3d.set_title("sample3d scatter plot")
ax_sample3d.set_zlabel("z")
ax_sample3d.set_ylabel("y")
ax_sample3d.set_xlabel("x")
plt.show()

# 重みの推定
# sample2d_1 線形回帰（単回帰）を用いる
sample2d_1_ndarray = sample2d_1.to_numpy()
sample2d_1_num_data = sample2d_1_ndarray.shape[0]
sample2d_1_design_mat = np.append(
    sample2d_1_ndarray[:, 0:1], np.ones((sample2d_1_num_data, 1)), axis=1
)  # 計画行列 X　sample2d_1_ndarray[:, 0:1]のように0:1としないとベクトルになってしまう
sample2d_1_design_mat_T = sample2d_1_design_mat.T  # 計画行列の転置 X^T
sample2d_1_meseared_vec = sample2d_1_ndarray[:, 1]  # 実測値のベクトル y

sample2d_1_weight_vec = (
    np.linalg.inv(sample2d_1_design_mat_T @ sample2d_1_design_mat)
    @ sample2d_1_design_mat_T
    @ sample2d_1_meseared_vec
)  # 重みのベクトル w = (X^T X)^{-1} X^T y
print("sample2d_1 weight vector:", sample2d_1_weight_vec)

# グラフの描画
fig, ax = plt.subplots()
ax.scatter(sample2d_1["x"], sample2d_1["y"], label="data")
x = np.linspace(sample2d_1["x"].min(), sample2d_1["x"].max(), 100)
y = sample2d_1_weight_vec[0] * x + sample2d_1_weight_vec[1]
ax.plot(x, y, color="red", label="fitted line")
ax.legend(loc="upper left")
ax.set_title("sample2d_1 linear regression")
ax.set_ylabel(sample2d_1_ylable)
ax.set_xlabel(sample2d_1_xlabel)
plt.show()
plt.savefig("output/sample2d_1_linear_regression.png")


# sample2d_2 3次非線形回帰（単回帰）を用いる
degree = 3
sample2d_2_ndarray = sample2d_2.to_numpy()
sample2d_2_num_data = sample2d_2_ndarray.shape[0]
sample2d_2_design_mat = np.append(
    sample2d_2_ndarray[:, 0:1], sample2d_2_ndarray[:, 0:1] ** 2, axis=1
)  # x^2の列を追加
for i in range(3, degree + 1):
    sample2d_2_design_mat = np.append(
        sample2d_2_design_mat, sample2d_2_design_mat[:, 0:1] ** i, axis=1
    )  # x^iの列を追加
sample2d_2_design_mat = np.append(
    sample2d_2_design_mat, np.ones((sample2d_2_num_data, 1)), axis=1
)  # 定数項の列を追加
sample2d_2_design_mat_T = sample2d_2_design_mat.T  # 計画行列の転置 X^T
sample2d_2_meseared_vec = sample2d_2_ndarray[:, 1]  # 実測値のベクトル y

sample2d_2_weight_vec = (
    np.linalg.inv(sample2d_2_design_mat_T @ sample2d_2_design_mat)
    @ sample2d_2_design_mat_T
    @ sample2d_2_meseared_vec
)  # 重みのベクトル w = (X^T X)^{-1} X^T y
print("sample2d_2 weight vector:", sample2d_2_weight_vec)

# グラフの描画
fig, ax = plt.subplots()
ax.scatter(sample2d_2["x"], sample2d_2["y"], label="data")
x = np.linspace(sample2d_2["x"].min(), sample2d_2["x"].max(), 100)
y = (
    sample2d_2_weight_vec[0] * x
    + sample2d_2_weight_vec[1] * x**2
    + sample2d_2_weight_vec[2] * x**3
    + sample2d_2_weight_vec[3]
)  # 3次の重回帰の式
ax.plot(x, y, color="red", label="fitted curve")
ax.legend(loc="upper left")
ax.set_title("sample2d_2 cubic regression")
ax.set_ylabel(sample2d_2_ylable)
ax.set_xlabel(sample2d_2_xlabel)
plt.show()
plt.savefig("output/sample2d_2_cubic_regression.png")


# sample3d 2次非線形回帰（重回帰）を用いる
degree = 2
sample3d_ndarray = sample3d.to_numpy()
sample3d_num_data = sample3d_ndarray.shape[0]
sample3d_design_mat = np.append(
    sample3d_ndarray[:, 0:2], sample3d_ndarray[:, 0:2] ** 2, axis=1
)  # x^2, y^2の列を追加
sample3d_design_mat = np.append(
    sample3d_design_mat, sample3d_ndarray[:, 0:1] * sample3d_ndarray[:, 1:2], axis=1
)  # x*yの列を追加
sample3d_design_mat = np.append(
    sample3d_design_mat, np.ones((sample3d_num_data, 1)), axis=1
)  # 定数項の列を追加
sample3d_design_mat_T = sample3d_design_mat.T  # 計画行列の転置 X^T
sample3d_meseared_vec = sample3d_ndarray[:, 2]  # 実測値のベクトル z
sample3d_weight_vec = (
    np.linalg.inv(sample3d_design_mat_T @ sample3d_design_mat)
    @ sample3d_design_mat_T
    @ sample3d_meseared_vec
)  # 重みのベクトル w = (X^T X)^{-1} X^T y
print("sample3d weight vector:", sample3d_weight_vec)

# グラフの描画
fig_sample3d = plt.figure()
ax_sample3d = fig_sample3d.add_subplot(111, projection="3d")
ax_sample3d.scatter(
    sample3d[sample3d_xlabel],
    sample3d[sample3d_ylable],
    sample3d[sample3d_zlable],
    label="data",
)
x = np.linspace(sample3d[sample3d_xlabel].min(), sample3d[sample3d_xlabel].max(), 100)
y = np.linspace(sample3d[sample3d_ylable].min(), sample3d[sample3d_ylable].max(), 100)
X, Y = np.meshgrid(x, y)
Z = (
    sample3d_weight_vec[0] * X
    + sample3d_weight_vec[1] * Y
    + sample3d_weight_vec[2] * X**2
    + sample3d_weight_vec[3] * Y**2
    + sample3d_weight_vec[4] * X * Y
    + sample3d_weight_vec[5]
)  # 2次の重回帰の式
ax_sample3d.plot_surface(X, Y, Z, color="red", alpha=0.5, label="fitted surface")
ax_sample3d.legend(loc="upper left")
ax_sample3d.set_title("sample3d quadratic regression")
ax_sample3d.set_zlabel(sample3d_zlable)
ax_sample3d.set_ylabel(sample3d_ylable)
ax_sample3d.set_xlabel(sample3d_xlabel)
plt.show()
plt.savefig("output/sample3d_quadratic_regression.png")
