import matplotlib.pyplot as plt
import numpy as np


# ---課題 3-1:データ確認---
def load_data(file_path):
    return np.loadtxt(file_path, delimiter=",")


def plot_data(X, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.show()


def main():
    # データの読み込み
    X = load_data("../data/data1.csv")

    # データのプロット
    plot_data(X, "scatter for data1")


if __name__ == "__main__":
    main()
