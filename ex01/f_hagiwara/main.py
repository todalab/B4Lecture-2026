import numpy as np
import matplotlib.pyplot as plt

def main():
    h11()
    h12()

def normal_eq(X,y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def h11():
### 課題1-1 ###
# 2d_1
    # csvファイル読み込み
    a1 = np.loadtxt('data/sample2d_1.csv', delimiter=',', skiprows=1)

    # データのスライシング
    x1 = a1[:, 0]
    y1 = a1[:, 1]

    # 散布図より線形が適切と判断
    # 正規方程式により推定

    # 行列の拡大
    x1_ex = np.column_stack([x1, np.ones_like(x1)]) #column_stackは1次行列の組み合わせなら2次行列に変換する
    w1 = normal_eq(x1_ex,y1)
    print("w1:" + w1)   # パラメータ確認

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(x1, y1)
    ax.axline((0, w1[1]), slope=w1[0], color='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sample2d_1")
    plt.savefig("sample2d_1.png")

# 2d_2
    # csvファイル読み込み
    a2 = np.loadtxt('data/sample2d_2.csv', delimiter=',', skiprows=1)

    # データのスライシング
    x2 = a2[:, 0]
    y2 = a2[:, 1]

    # 散布図より3次式が適切と判断
    # 正規方程式により推定
    x2_ex = np.column_stack([x2**3, x2**2, x2, np.ones_like(x2)])
    w2 = normal_eq(x2_ex,y2)
    print("w2:" + w2)

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(x2, y2)   # 点で表示
    x = np.linspace(0, 10, 100)
    y = w2[0] * x**3 + w2[1] * x**2 + w2[2] * x + w2[3]
    plt.plot(x, y, color='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sample2d_2(3D)")
    plt.savefig("sample2d_2.png")

# 3d
    # csvファイル読み込み
    a3 = np.loadtxt('data/sample3d.csv', delimiter=',', skiprows=1)

    # データのスライシング
    x3 = a3[:, 0]
    y3 = a3[:, 1]
    z3 = a3[:, 2]

    # 散布図より曲面(zをx,yで表現)が適切と判断
    # 先程3Dで行ったものと同様、項ごとにパラメータを設定
    # 中央に向かって深くなっている様子が見受けられるためそれぞれ2次とする
    # 正規方程式により推定
    xy_ex = np.column_stack([x3**2, x3, y3**2, y3, np.ones_like(x3)])
    w3 = normal_eq(xy_ex,z3)
    print("w3:" + w3)

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(x, y, z, s=10)

    x = np.arange(-5, 5, 0.05)
    y = np.arange(-5, 5, 0.05)
    x, y = np.meshgrid(x, y)
    z = w3[0] * x**2 + w3[1] * x + w3[2] * y**2 + w3[3] * y + w3[4]
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='inferno', alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("sample3d")
    plt.savefig("sample3d.png", dpi=150)

def h12():
### 課題1-2 ###
    learn_rate = 0.05
    w = np.zeros(2)
    print(w)

    # csvファイル読み込み
    a = np.loadtxt('data/sample_logistic.csv', delimiter=',', skiprows=1)

    # データのスライシング
    x = a[:, 0:2]
    y = a[:, 2]
    N = y.size

    loss_list = []
    likeli_list = []
    acc_list = []

    # 今回は大丈夫そうだが勾配消失の対応をすべきか
    for i in range(300):
        sigm = 1 / (1 + np.exp(-x @ w))
        likeli = np.sum(y * np.log(sigm) + (1 - y) * np.log(1 - sigm))
        loss = -likeli / N
        grad = ((sigm - y) @ x) / N
        w = w - learn_rate * grad
        acc = np.count_nonzero(np.abs(sigm-y) < 0.5) / N

        likeli_list.append(likeli)
        loss_list.append(loss)
        acc_list.append(acc)

    fig = plt.figure(figsize=(6,8))
    plotx = np.arange(1, 301)

    ax1 = fig.add_subplot(311)
    plt.grid()
    plt.ylabel("Likelihood")
    plt.yticks(np.arange(-200,-100,20))
    plt.plot(plotx,likeli_list,color='red')

    ax2 = fig.add_subplot(312)
    x = np.linspace(0, 300, 300)
    plt.grid()
    plt.ylabel("Loss")
    plt.plot(plotx,loss_list, color='blue')

    ax3 = fig.add_subplot(313)
    x = np.linspace(0, 300, 300)
    plt.grid()
    plt.ylabel("Accuracy")
    plt.plot(plotx,acc_list, color='green')

    plt.suptitle("sample_logistic(η=0.05)", fontsize=20)
    plt.savefig("samplelog.png", dpi=300)

if __name__ == "__main__":
    main()
