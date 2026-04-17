def main():
    import numpy as np
    import matplotlib.pyplot as plt

    # csvファイル読み込み
    a = np.loadtxt('data/sample2d_1.csv', delimiter=',', skiprows=1)
    print(a)

    # データのスライシング
    x = a[:, 0]
    y = a[:, 1]

    # プロット
    plt.figure()
    plt.scatter(x, y)   # 点で表示

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sample2d_1")
    plt.savefig("sample2d_1.png")


if __name__ == "__main__":
    main()
