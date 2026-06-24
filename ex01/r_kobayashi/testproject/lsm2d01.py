# 2次元配列その1

import matplotlib.pyplot as plt
import numpy as np

path_2d_1 = "../../../ex1/data/sample2d_1.csv"

with open(path_2d_1) as f:
    sample2_1 = np.loadtxt(path_2d_1, delimiter=",", skiprows=1)

x = sample2_1[:, 0]
y = sample2_1[:, 1]

X = np.c_[x, np.ones(x.shape[0])]  # Xの行列の端に[1,1,1,...,1]^Tを加える。
A = np.dot(X.T, X)
w = np.linalg.pinv(A) @ X.T @ y
print(w)

print("回帰曲線: " + str(round(w[0], 6)) + "x+" + str(round(w[1], 6)))

plt.scatter(x, y, color="black", s=10)
plt.plot(
    [x.min(), x.max()], [w[0] * x.min() + w[1], w[0] * x.max() + w[1]], color="green"
)
# plt.show()
plt.savefig("2d_01.png")
print("2d_01.pngを出力しました。")
