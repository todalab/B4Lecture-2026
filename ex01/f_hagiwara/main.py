def main():
    import numpy as np
    import matplotlib.pyplot as plt

# 2d_1
    # csvファイル読み込み
    a1 = np.loadtxt('data/sample2d_1.csv', delimiter=',', skiprows=1)

    # データのスライシング
    x = a1[:, 0]
    y = a1[:, 1]

    # 散布図より線形が適切と判断
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

    # 散布図より3次式が適切と判断
    # 正規方程式により推定
    ones = np.ones(x.size)
    ones_row = ones.reshape(1,-1)
    x_row = x.reshape(1, -1)
    x_row2 = x_row*x_row
    x_row3 = x_row*x_row2
    x_ex = np.concatenate([x_row3.T, x_row2.T, x_row.T, ones_row.T],1)
    x_T = x_ex.T
    w2 = np.linalg.inv(x_T@x_ex)@x_T@y
    print(w2)

# TODO: 他とのコードの整合性を整える
# TODO: 4次式,5次式にした際の変化を確認する
    # プロット
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(x, y)   # 点で表示
    x = np.linspace(0, 10, 100)
    y = w2[0]*x**3 + w2[1]*x**2 + w2[2]*x + w2[3]
    plt.plot(x, y, color='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sample2d_2(3D)")
    plt.savefig("sample2d_2.png")

# 3d
    # csvファイル読み込み
    a3 = np.loadtxt('data/sample3d.csv', delimiter=',', skiprows=1)

    # データのスライシング
    x = a3[:, 0]
    y = a3[:, 1]
    z = a3[:, 2]

    # 散布図より曲面(zをx,yで表現)が適切と判断
    # 先程3Dで行ったものと同様、項ごとにパラメータを設定
    # 中央に向かって深くなっている様子が見受けられるためそれぞれ2次とする
    # 正規方程式により推定
    ones = np.ones(x.size)
    ones_row = ones.reshape(1,-1)
    x_row = x.reshape(1, -1)
    x_row2 = x_row*x_row
    y_row = y.reshape(1, -1)
    y_row2 = y_row*y_row
    xy_ex = np.concatenate([x_row2.T, x_row.T, y_row2.T, y_row.T, ones_row.T],1)
    xy_T = xy_ex.T
    w3 = np.linalg.inv(xy_T@xy_ex)@xy_T@z
    print(w3)

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(x, y, z, s=10)
    x = np.arange(-5, 5, 0.05)
    y = np.arange(-5, 5, 0.05)
    x, y = np.meshgrid(x, y)
    z = w3[0]*x**2 + w3[1]*x + w3[2]*y**2 + w3[3]*y + w3[4]
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='inferno', alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("sample3d")
    plt.savefig("sample3d.png", dpi=300)

if __name__ == "__main__":
    main()
