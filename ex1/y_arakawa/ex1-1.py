import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'sans-serif'


# CSVファイルからデータを読み込む
sample2d_1 = pd.read_csv("../data/sample2d_1.csv")
sample2d_2 = pd.read_csv("../data/sample2d_2.csv")
sample3d = pd.read_csv("../data/sample3d.csv")
sample_logistic = pd.read_csv("../data/sample_logistic.csv")

# データを可視化
## sample2d_1
print(sample2d_1.describe())
sample2d_1_xlabel, sample2d_1_ylable = sample2d_1.columns

plt.scatter(sample2d_1["x"], sample2d_1["y"], label = "data")
plt.legend(loc='upper left')
plt.title("sample2d_1 scatter plot")
plt.ylabel(sample2d_1_ylable)
plt.xlabel(sample2d_1_xlabel)
plt.show()

## sample2d_2
print(sample2d_2.describe())
sample2d_2_xlabel, sample2d_2_ylable = sample2d_2.columns

plt.scatter(sample2d_2["x"], sample2d_2["y"], label = "data")
plt.legend(loc='upper left')
plt.title("sample2d_2 scatter plot")
plt.ylabel(sample2d_2_ylable)
plt.xlabel(sample2d_2_xlabel)
plt.show()

## sample3d
print(sample3d.describe())
sample3d_xlabel, sample3d_ylable, sample3d_zlable = sample3d.columns

fig_sample3d = plt.figure()
ax_sample3d = fig_sample3d.add_subplot(111, projection = "3d")
ax_sample3d.scatter(sample3d[sample3d_xlabel], sample3d[sample3d_ylable],sample3d[sample3d_zlable], label = "data")
ax_sample3d.legend(loc="upper left")
ax_sample3d.set_title("sample3d scatter plot")
ax_sample3d.set_zlabel("z")
ax_sample3d.set_ylabel("y")
ax_sample3d.set_xlabel("x")
plt.show()

