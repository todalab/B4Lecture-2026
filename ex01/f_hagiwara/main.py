# 発展内容のためのライブラリscikit-learnもimport
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import auc, confusion_matrix, roc_curve


def main():
    h11()
    h12()


def normal_eq(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def h11():
    ### 課題1-1 ###
    # 2d_1
    # csvファイル読み込み
    a1 = np.loadtxt("data/sample2d_1.csv", delimiter=",", skiprows=1)

    # データのスライシング
    x1 = a1[:, 0]
    y1 = a1[:, 1]

    # 散布図より線形が適切と判断
    # 正規方程式により推定

    # 行列の拡大
    x1_ex = np.column_stack(
        [x1, np.ones_like(x1)]
    )  # column_stackは1次行列の組み合わせなら2次行列に変換する
    w1 = normal_eq(x1_ex, y1)
    print("w1:", w1)  # パラメータ確認

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(x1, y1)
    ax.axline((0, w1[1]), slope=w1[0], color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sample2d_1")
    plt.savefig("sample2d_1.png")

    # 過学習の検証
    x1_ex2 = np.column_stack([x1**2, x1, np.ones_like(x1)])
    x1_ex3 = np.column_stack([x1**3, x1**2, x1, np.ones_like(x1)])
    x1_ex4 = np.column_stack([x1**4, x1**3, x1**2, x1, np.ones_like(x1)])
    x1_ex5 = np.column_stack([x1**5, x1**4, x1**3, x1**2, x1, np.ones_like(x1)])

    w12 = normal_eq(x1_ex2, y1)
    w13 = normal_eq(x1_ex3, y1)
    w14 = normal_eq(x1_ex4, y1)
    w15 = normal_eq(x1_ex5, y1)

    x = np.linspace(-10, 10, 100)
    y11 = w1[0] * x + w1[1]
    y12 = w12[0] * x**2 + w12[1] * x + w12[2]
    y13 = w13[0] * x**3 + w13[1] * x**2 + w13[2] * x + w13[3]
    y14 = w14[0] * x**4 + w14[1] * x**3 + w14[2] * x**2 + w14[3] * x + w14[4]
    y15 = (
        w15[0] * x**5
        + w15[1] * x**4
        + w15[2] * x**3
        + w15[3] * x**2
        + w15[4] * x
        + w15[5]
    )

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(x1, y1)
    plt.plot(x, y11, label="1st", color="black")
    plt.plot(x, y12, label="2nd", color="red")
    plt.plot(x, y13, label="3rd", color="blue")
    plt.plot(x, y14, label="4th", color="green")
    plt.plot(x, y15, label="5th", color="orange")
    plt.legend()

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sample2d_1(compare:wide)")
    plt.savefig("sample2d_1_compare_w.png")

    # Ridge回帰
    ridge = Ridge(alpha=1.0, fit_intercept=False)
    ridge.fit(x1_ex5, y1)
    y15_ridge = ridge.predict(
        np.column_stack([x**5, x**4, x**3, x**2, x, np.ones_like(x)])
    )

    # Lasso
    lasso = Lasso(alpha=0.1, fit_intercept=False, max_iter=2000)
    lasso.fit(x1_ex5, y1)
    y15_lasso = lasso.predict(
        np.column_stack([x**5, x**4, x**3, x**2, x, np.ones_like(x)])
    )

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(x1, y1)
    plt.plot(x, y11, label="1st", color="black")
    plt.plot(x, y12, label="2nd", color="red")
    plt.plot(x, y13, label="3rd", color="blue")
    plt.plot(x, y14, label="4th", color="green")
    plt.plot(x, y15, label="5th", color="orange")
    plt.plot(x, y15_ridge, label="Ridge", linestyle="--", color="orange")
    plt.plot(x, y15_lasso, label="Lasso", linestyle=":", color="orange")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sample2d_1_lr")
    plt.savefig("sample2d_1_lr.png")

    # 2d_2
    # csvファイル読み込み
    a2 = np.loadtxt("data/sample2d_2.csv", delimiter=",", skiprows=1)

    # データのスライシング
    x2 = a2[:, 0]
    y2 = a2[:, 1]

    # 散布図より3次式が適切と判断
    # 正規方程式により推定
    x2_ex = np.column_stack([x2**3, x2**2, x2, np.ones_like(x2)])
    w2 = normal_eq(x2_ex, y2)
    print("w2:", w2)

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(x2, y2)  # 点で表示
    x = np.linspace(0, 10, 100)
    y = w2[0] * x**3 + w2[1] * x**2 + w2[2] * x + w2[3]
    plt.plot(x, y, color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sample2d_2(3D)")
    plt.savefig("sample2d_2.png")

    # 3d
    # csvファイル読み込み
    a3 = np.loadtxt("data/sample3d.csv", delimiter=",", skiprows=1)

    # データのスライシング
    x3 = a3[:, 0]
    y3 = a3[:, 1]
    z3 = a3[:, 2]

    # 散布図より曲面(zをx,yで表現)が適切と判断
    # 先程3Dで行ったものと同様、項ごとにパラメータを設定
    # 中央に向かって深くなっている様子が見受けられるためそれぞれ2次とする
    # 正規方程式により推定
    xy_ex = np.column_stack([x3**2, x3, y3**2, y3, np.ones_like(x3)])
    w3 = normal_eq(xy_ex, z3)
    print("w3:", w3)

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x3, y3, z3, s=10)

    x = np.arange(-5, 5, 0.05)
    y = np.arange(-5, 5, 0.05)
    x, y = np.meshgrid(x, y)
    z = w3[0] * x**2 + w3[1] * x + w3[2] * y**2 + w3[3] * y + w3[4]
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap="inferno", alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("sample3d")
    plt.savefig("sample3d.png", dpi=150)


def h12():
    ### 課題1-2 ###
    learn_rate = 0.5
    w = np.zeros(2)

    # csvファイル読み込み
    a = np.loadtxt("data/sample_logistic.csv", delimiter=",", skiprows=1)

    # データのスライシング
    x = a[:, 0:2]
    y = a[:, 2]
    N = y.size

    # 値格納用のリスト宣言
    loss_list = []
    likeli_list = []
    acc_list = []
    auc_list = []

    # 勾配降下法
    for i in range(300):
        sigm = 1 / (1 + np.exp(-x @ w))

        likeli = np.sum(y * np.log(sigm) + (1 - y) * np.log(1 - sigm))  # 尤度
        loss = -likeli / N  # 損失
        grad = ((sigm - y) @ x) / N  # 勾配
        w = w - learn_rate * grad  # パラメータ更新
        acc = np.count_nonzero(np.abs(sigm - y) < 0.5) / N  # 閾値は0.5とした
        fpr, tpr, thresholds = roc_curve(y, sigm)
        auc_score = auc(fpr, tpr)

        likeli_list.append(likeli)
        loss_list.append(loss)
        acc_list.append(acc)
        auc_list.append(auc_score)

    # ROC-AUCの値取得
    sigm = 1 / (1 + np.exp(-x @ w))
    fpr, tpr, thresholds = roc_curve(y, sigm)
    auc_score = auc(fpr, tpr)
    y_pred = (sigm >= 0.5).astype(int)
    mat = confusion_matrix(y, y_pred)

    # プロット
    fig = plt.figure(figsize=(6, 11))
    plotx = np.arange(1, 301)

    # 尤度
    ax1 = fig.add_subplot(411)
    plt.grid()
    plt.ylabel("Likelihood")
    plt.yticks(np.arange(-200, -100, 20))
    plt.plot(plotx, likeli_list, color="red")

    # 損失
    ax2 = fig.add_subplot(412)
    x = np.linspace(0, 300, 300)
    plt.grid()
    plt.ylabel("Loss")
    plt.plot(plotx, loss_list, color="blue")

    # 正解率
    ax3 = fig.add_subplot(413)
    x = np.linspace(0, 300, 300)
    plt.grid()
    plt.ylabel("Accuracy")
    plt.plot(plotx, acc_list, color="green")

    # ROC-AUC
    ax4 = fig.add_subplot(414)
    x = np.linspace(0, 300, 300)
    plt.grid()
    ax4.set_ylabel("ROC-AUC")
    ax4.set_xlabel("Epochs")
    ax4.plot(plotx, auc_list, color="orange")

    plt.suptitle("sample_logistic(η=0.5)", fontsize=20)
    plt.savefig("samplelog.png", dpi=300)

    # ROC曲線
    plt.figure()
    plt.plot(fpr, tpr, marker="o")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.suptitle("ROC Curve", fontsize=20)
    plt.savefig("roc_curve.png", dpi=300)


if __name__ == "__main__":
    main()
