def main():
    import numpy as np
    import matplotlib.pyplot as plt

# 2d_1
    # csvファイル読み込み
    a1 = np.loadtxt('data/sample2d_1.csv', delimiter=',', skiprows=1)

    # データのスライシング
    x = a1[:, 0]
    y = a1[:, 1]

    # 図より線形が適切と判断
    # 正規方程式により推定
    ones = np.ones(x.size)
    ones_row = ones.reshape(1,-1)
    x_row = x.reshape(1, -1)
    x_ex = np.concatenate([x_row.T, ones_row.T],1)
    x_T = x_ex.T
    w1 = np.linalg.inv(x_T@x_ex)@x_T@y
    print(w1)

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(x, y)   # 点で表示
    ax.axline((0, w1[1]), slope=w1[0], color='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sample2d_1")
    plt.savefig("sample2d_1.png")

# 2d_2
    # csvファイル読み込み
    a2 = np.loadtxt('data/sample2d_2.csv', delimiter=',', skiprows=1)

    # データのスライシング
    x = a2[:, 0]
    y = a2[:, 1]

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(x, y)   # 点で表示
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sample2d_2")
    plt.savefig("sample2d_2.png")

# 3d
    # csvファイル読み込み
    a3 = np.loadtxt('data/sample3d.csv', delimiter=',', skiprows=1)

    # データのスライシング
    x = a3[:, 0]
    y = a3[:, 1]
    z = a3[:, 2]

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("sample3d")
    plt.savefig("sample3d.png")

if __name__ == "__main__":
    main()
