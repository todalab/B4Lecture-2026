def main():
    import numpy as np
    import matplotlib.pyplot as plt

# 2d_1
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

# 2d_2
    # csvファイル読み込み
    a = np.loadtxt('data/sample2d_2.csv', delimiter=',', skiprows=1)
    print(a)

    # データのスライシング
    x = a[:, 0]
    y = a[:, 1]

    # プロット
    plt.figure()
    plt.scatter(x, y)   # 点で表示
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sample2d_2")
    plt.savefig("sample2d_2.png")

# 3d
    # csvファイル読み込み
    a = np.loadtxt('data/sample3d.csv', delimiter=',', skiprows=1)
    print(a)

    # データのスライシング
    x = a[:, 0]
    y = a[:, 1]
    z = a[:, 2]

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("sample3d")
    plt.savefig("sample3d.png")

if __name__ == "__main__":
    main()
