# 2次元配列その2

import matplotlib.pyplot as plt
import numpy as np

path_2d_2 = "../../../ex1/data/sample2d_2.csv"

with open(path_2d_2) as f:
    sample2_2 = np.loadtxt(path_2d_2, delimiter=",", skiprows=1)

x = sample2_2[:, 0]
y = sample2_2[:, 1]


# 一次式以外の多項式への回帰を行えるようにした
def pol_reg(x_input, y_input, deg):
    l = []
    for x in x_input:
        tmp = []
        for j in range(0, deg + 1):
            tmp.append(x**j)
        l.append(tmp)
    # l = [1, x, x^2,...]

    X = np.array(l, dtype=float)
    y = np.array([[y] for y in y_input])
    w = ((np.linalg.inv(X.T @ X)) @ X.T) @ y

    x_axis = np.linspace(x_input.min(), x_input.max())
    y_axis = []
    for z in x_axis:
        val = 0
        for i in range(len(w)):
            val += w[i][0] * z**i
        y_axis.append(val)
    plt.scatter(x_input, y_input, color="black", s=10)
    plt.plot(x_axis, y_axis)
    # plt.show()
    plt.savefig("2d_02.png")

    result = "回帰曲線: y = "
    for index, value in enumerate(w):
        result += str(round(value[0], 4)) + "x^" + str(index) + " + "
    result = result[:-3]
    print(result)
    print("2d_02.pngを出力しました。")


if __name__ == "__main__":
    pol_reg(x, y, 3)
