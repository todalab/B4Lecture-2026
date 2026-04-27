"""The answer of Ex02 by Hagiwara Futa."""

import japanize_matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def main():
    """Run main function."""
    pca()
    lda()


def pca():
    """Solve Principal Component Analysis(ex2-1)."""

    # データ読み込み
    val2d = np.loadtxt("../data/pca_2d.csv", delimiter=",")
    val2d_x = val2d[:, 0]
    val2d_y = val2d[:, 1]
    M2d = val2d_x.size

    val3d = np.loadtxt("../data/pca_3d.csv", delimiter=",")
    val3d_x = val3d[:, 0]
    val3d_y = val3d[:, 1]
    val3d_z = val3d[:, 2]
    M3d = val3d_x.size

    # 散布図取得
    # pca_2d
    plt.scatter(val2d_x, val2d_y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("pca_2d")
    plt.grid(True)
    plt.savefig("outputs/pca_2d_scat.png")

    # pca_3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(val3d_x, val3d_y, val3d_z, s=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("pca_3d")
    plt.grid(True)
    plt.savefig("outputs/pca_3d_scat.png", dpi=150)

    # pca_3dを回してみる
    def rotate(angle):
        ax.view_init(azim=angle)

    pca_3d_gif = animation.FuncAnimation(
        fig, rotate, frames=np.arange(0, 360, 2), interval=100
    )

    # GIF保存
    pca_3d_gif.save("outputs/pca_3d_scat_rot.gif", writer="pillow")
    plt.close(fig)

    # 平均中心化
    val2d_mean = np.mean(val2d, axis=0)
    val2d_mc = val2d - val2d_mean

    val3d_mean = np.mean(val3d, axis=0)
    val3d_mc = val3d - val3d_mean

    val2d_x_mc = val2d_mc[:, 0]
    val2d_y_mc = val2d_mc[:, 1]

    val3d_x_mc = val3d_mc[:, 0]
    val3d_y_mc = val3d_mc[:, 1]
    val3d_z_mc = val3d_mc[:, 2]

    # 共分散行列計算
    Sxy2d = np.sum(val2d_x_mc * val2d_y_mc)
    CovMat2d = (
        np.array(
            [
                [np.sum(val2d_x_mc**2), Sxy2d],
                [Sxy2d, np.sum(val2d_y_mc**2)],
            ]
        )
        / M2d
    )

    Sxy = np.sum(val3d_x_mc * val3d_y_mc)
    Sxz = np.sum(val3d_x_mc * val3d_z_mc)
    Syz = np.sum(val3d_y_mc * val3d_z_mc)

    CovMat3d = (
        np.array(
            [
                [np.sum(val3d_x_mc**2), Sxy, Sxz],
                [Sxy, np.sum(val3d_y_mc**2), Syz],
                [Sxz, Syz, np.sum(val3d_z_mc**2)],
            ]
        )
        / M3d
    )

    # 固有値問題・主成分ソート
    eigval_2d, eigvec_2d = np.linalg.eigh(CovMat2d)
    idx = np.argsort(eigval_2d)[::-1]
    eigval_2d = eigval_2d[idx]
    eigvec_2d = eigvec_2d[:, idx]

    eigval_3d, eigvec_3d = np.linalg.eigh(CovMat3d)
    idx = np.argsort(eigval_3d)[::-1]
    eigval_3d = eigval_3d[idx]
    eigvec_3d = eigvec_3d[:, idx]

    # 寄与率
    contri_2d = eigval_2d / np.sum(eigval_2d)
    contri_3d = eigval_3d / np.sum(eigval_3d)

    # 主成分を重ねた散布図取得（可視化）
    # pca_2d
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(val2d_x, val2d_y)

    v1 = eigvec_2d[:, 0]
    v2 = eigvec_2d[:, 1]

    ax.axline(
        val2d_mean,
        val2d_mean + v1,
        color="red",
        label=f"1st ({contri_2d[0]*100:.1f}%)",
    )

    ax.axline(
        val2d_mean,
        val2d_mean + v2,
        color="orange",
        label=f"2nd ({contri_2d[1]*100:.1f}%)",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("主成分の軸を示したpca_2d")
    ax.legend()
    ax.grid(True)
    plt.savefig("outputs/pca_2d.png")
    plt.close(fig)

    # pca_3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(val3d_x, val3d_y, val3d_z, s=10)

    t = np.linspace(-2, 2, 100)
    v1 = eigvec_3d[:, 0]
    v2 = eigvec_3d[:, 1]
    v3 = eigvec_3d[:, 2]

    ax.plot(
        val3d_mean[0] + t * v1[0],
        val3d_mean[1] + t * v1[1],
        val3d_mean[2] + t * v1[2],
        color="red",
        label=f"1st ({contri_3d[0]*100:.1f}%)",
    )

    ax.plot(
        val3d_mean[0] + t * v2[0],
        val3d_mean[1] + t * v2[1],
        val3d_mean[2] + t * v2[2],
        color="orange",
        label=f"2nd ({contri_3d[1]*100:.1f}%)",
    )

    ax.plot(
        val3d_mean[0] + t * v3[0],
        val3d_mean[1] + t * v3[1],
        val3d_mean[2] + t * v3[2],
        color="green",
        label=f"3rd ({contri_3d[2]*100:.1f}%)",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("主成分の軸を示したpca_3d")
    ax.legend()
    ax.grid(True)
    plt.savefig("outputs/pca_3d.png", dpi=150)

    pca_3d_gif = animation.FuncAnimation(
        fig, rotate, frames=np.arange(0, 360, 2), interval=100
    )

    # GIF保存
    pca_3d_gif.save("outputs/pca_3d_rot.gif", writer="pillow")
    plt.close(fig)

    # pca_3dに対するPCAの実行（2次元への圧縮）及びその可視化
    # どの要素が不要かを自動で処理するのは100次元の時に行うとして、ここでは3番目の要素（z）が不要と分かるためこれを削減する
    W3d = eigvec_3d[:, :2]
    val3d_pca = val3d_mc @ W3d

    val3d_pca1 = val3d_pca[:, 0]
    val3d_pca2 = val3d_pca[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(val3d_pca1, val3d_pca2)
    plt.xlabel("data1")
    plt.ylabel("data2")
    plt.title("pca_3d compressed by PCA")
    plt.grid(True)
    plt.savefig("outputs/pca_3d_comp.png")

    # pca_100d
    val100d = np.loadtxt("../data/pca_100d.csv", delimiter=",")  # データ読み込み
    val100d_mc = val100d - np.mean(val100d, axis=0)  # 平均中心化
    CovMat100d = (val100d_mc.T @ val100d_mc) / val100d.shape[0]  # 共分散行列
    eigval_100d, eigvec_100d = np.linalg.eigh(CovMat100d)  # 固有値・固有ベクトル
    idx = np.argsort(eigval_100d)[::-1]  # 最大固有値取得
    eigval_100d = eigval_100d[idx]
    eigvec_100d = eigvec_100d[:, idx]

    W100d = eigvec_100d[:, :2]
    val100d_pca = val100d_mc @ W100d  # 転置@転置が必要だったため逆に
    val100d_pca1 = val100d_pca[:, 0]
    val100d_pca2 = val100d_pca[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(val100d_pca1, val100d_pca2)
    plt.xlabel("data1")
    plt.ylabel("data2")
    plt.title("pca_100d compressed by PCA")
    plt.grid(True)
    plt.savefig("outputs/pca_100d_comp.png")

    # 寄与率の計算
    # 累積寄与率が0.9以上となる最小の次元数を得る
    eigval_100d_sum = np.sum(eigval_100d)
    ContriRate_100d = eigval_100d / eigval_100d_sum
    cevlist = []
    cev = 0.0
    k = 0
    k90 = 0

    while k < ContriRate_100d.size:
        cev += ContriRate_100d[k]
        cevlist.append(cev)
        if cev >= 0.9 and k90 == 0:
            k90 = k + 1  # kは0始まりなのでk+1が次元数
        k += 1

    print("次元数:", k90)

    # 累積寄与率プロット
    fig = plt.figure()
    ax = fig.add_subplot()
    x = np.arange(1, ContriRate_100d.size + 1)
    plt.plot(x, cevlist, marker=".", markersize=5, label="累積寄与率")
    ax.axhline(0.9, color="red", linestyle="--", label="90%")
    plt.title("累積寄与率")
    plt.grid(True)
    plt.savefig("outputs/pca_100d_ContriRate.png")

    # 【発展】標準化
    val2d_std = np.std(val2d, axis=0)
    val2d_std_data = (val2d - val2d_mean) / val2d_std

    val3d_std = np.std(val3d, axis=0)
    val3d_std_data = (val3d - val3d_mean) / val3d_std

    # 共分散行列計算
    CovMat2d = (val2d_std_data.T @ val2d_std_data) / M2d
    CovMat3d = (val3d_std_data.T @ val3d_std_data) / M3d

    # 固有値問題
    eigval_2d, eigvec_2d = np.linalg.eigh(CovMat2d)
    idx = np.argsort(eigval_2d)[::-1]
    eigval_2d = eigval_2d[idx]
    eigvec_2d = eigvec_2d[:, idx]

    eigval_3d, eigvec_3d = np.linalg.eigh(CovMat3d)
    idx = np.argsort(eigval_3d)[::-1]
    eigval_3d = eigval_3d[idx]
    eigvec_3d = eigvec_3d[:, idx]

    # 寄与率
    contri_2d = eigval_2d / np.sum(eigval_2d)
    contri_3d = eigval_3d / np.sum(eigval_3d)

    # 可視化（2D）
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(val2d_std_data[:, 0], val2d_std_data[:, 1])

    v1 = eigvec_2d[:, 0]
    v2 = eigvec_2d[:, 1]

    ax.axline(
        [0, 0],
        v1,
        color="red",
        label=f"1st ({contri_2d[0]*100:.1f}%)",
    )

    ax.axline(
        [0, 0],
        v2,
        color="orange",
        label=f"2nd ({contri_2d[1]*100:.1f}%)",
    )

    ax.set_title("主成分の軸を示したpca_2d（標準化）")
    ax.legend()
    ax.grid(True)
    plt.savefig("outputs/pca_2d_std.png")
    plt.close(fig)

    # 可視化（3D）
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        val3d_std_data[:, 0],
        val3d_std_data[:, 1],
        val3d_std_data[:, 2],
        s=10,
    )

    v1 = eigvec_3d[:, 0]
    v2 = eigvec_3d[:, 1]
    v3 = eigvec_3d[:, 2]

    t = np.linspace(-2, 2, 100)

    ax.plot(
        val3d_mean[0] + t * v1[0],
        val3d_mean[1] + t * v1[1],
        val3d_mean[2] + t * v1[2],
        color="red",
        label=f"1st ({contri_3d[0]*100:.1f}%)",
    )

    ax.plot(
        val3d_mean[0] + t * v2[0],
        val3d_mean[1] + t * v2[1],
        val3d_mean[2] + t * v2[2],
        color="orange",
        label=f"2nd ({contri_3d[1]*100:.1f}%)",
    )

    ax.plot(
        val3d_mean[0] + t * v3[0],
        val3d_mean[1] + t * v3[1],
        val3d_mean[2] + t * v3[2],
        color="green",
        label=f"3rd ({contri_3d[2]*100:.1f}%)",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("主成分の軸を示したpca_3d（標準化）")
    ax.legend()
    ax.grid(True)
    plt.savefig("outputs/pca_3d_std.png")
    plt.close(fig)

    pca_3d_gif = animation.FuncAnimation(
        fig, rotate, frames=np.arange(0, 360, 2), interval=100
    )

    # GIF保存
    pca_3d_gif.save("outputs/pca_3d_std_rot.gif", writer="pillow")
    plt.close(fig)


def lda():
    """Solve Linear Discriminant Analysis(ex2-2)."""

    valLDA = np.loadtxt("../data/lda_2class.csv", delimiter=",", skiprows=1)
    valLDA_data = valLDA[:, :2]
    valLDA_label = valLDA[:, 2]
    mean_all = np.mean(valLDA_data, axis=0)

    valLDA_data0 = valLDA_data[valLDA_label == 0]
    valLDA_data1 = valLDA_data[valLDA_label == 1]

    # クラス毎の平均ベクトル
    mean0 = np.mean(valLDA_data0, axis=0)
    mean1 = np.mean(valLDA_data1, axis=0)

    # クラス内分散
    valLDA_data0_mc = valLDA_data0 - mean0
    valLDA_data1_mc = valLDA_data1 - mean1

    Sw = valLDA_data0_mc.T @ valLDA_data0_mc + valLDA_data1_mc.T @ valLDA_data1_mc
    print("Sw:\n", Sw)

    # クラス間分散
    mean_dif = mean0 - mean1
    Sb = np.outer(mean_dif, mean_dif)
    print("Sb:\n", Sb)

    # 一般化固有値問題
    # -λIを流用するためにSwの逆行列を掛ける
    eigval_LDA, eigvec_LDA = np.linalg.eig(
        np.linalg.inv(Sw) @ (Sb)
    )  # 固有値・固有ベクトル
    idx = np.argsort(eigval_LDA)[::-1]  # 最大固有値取得
    # eigval_LDA = eigval_LDA[idx]
    eigvec_LDA = eigvec_LDA[:, idx]
    WLDA = eigvec_LDA[:, 0]

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot()

    # 散布図
    ax.scatter(valLDA_data0[:, 0], valLDA_data0[:, 1], label="class 0")
    ax.scatter(valLDA_data1[:, 0], valLDA_data1[:, 1], label="class 1")

    # LDA軸（直線）
    t = np.linspace(-5, 5, 100)

    ax.plot(
        mean_all[0] + t * WLDA[0],
        mean_all[1] + t * WLDA[1],
        color="red",
        label="LDA射影軸",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("LDA射影軸を重ねた散布図")
    ax.legend()
    ax.grid(True)
    plt.savefig("outputs/lda_2d.png")
    plt.close(fig)

    # LDA軸へ射影した1次元データ
    valLDA_1d = valLDA_data @ WLDA
    valLDA_1d_0 = valLDA_1d[valLDA_label == 0]
    valLDA_1d_1 = valLDA_1d[valLDA_label == 1]

    # 1次元データの可視化
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.set_size_inches(6, 2)
    ax.set_ylim(-0.05, 0.15)
    ax.scatter(valLDA_1d_0, np.zeros_like(valLDA_1d_0), label="class 0", s=10)
    ax.scatter(valLDA_1d_1, np.ones_like(valLDA_1d_1) * 0.1, label="class 1", s=10)
    ax.tick_params(left=False, right=False)
    plt.yticks(color="None")
    ax.set_title("LDA 軸へ射影した1次元データ")
    ax.legend()
    plt.savefig("outputs/lda_1d.png")
    plt.close(fig)

    # 【発展】複数評価指標による比較
    # 分類1：0を閾値とする
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    threshold1 = 0

    y_true = valLDA_label.astype(int)

    # 散布図からより大きいほうがclass0
    y_pred1 = (valLDA_1d < threshold1).astype(int)

    TP = np.sum((y_pred1 == 1) & (y_true == 1))
    FP = np.sum((y_pred1 == 1) & (y_true == 0))
    FN = np.sum((y_pred1 == 0) & (y_true == 1))
    TN = np.sum((y_pred1 == 0) & (y_true == 0))

    acc1 = (TP + TN) / (TP + TN + FP + FN)
    print("閾値を0としたときのacc=", acc1)

    # 分類2：平均の中点を動的に閾値とする
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    score0 = valLDA_1d[y_true == 0]
    score0_mean = np.mean(score0)
    score1 = valLDA_1d[y_true == 1]
    score1_mean = np.mean(score1)

    threshold2 = (score0_mean + score1_mean) / 2

    # 散布図からより大きいほうがclass0
    y_pred2 = (valLDA_1d < threshold2).astype(int)

    TP = np.sum((y_pred2 == 1) & (y_true == 1))
    FP = np.sum((y_pred2 == 1) & (y_true == 0))
    FN = np.sum((y_pred2 == 0) & (y_true == 1))
    TN = np.sum((y_pred2 == 0) & (y_true == 0))

    acc2 = (TP + TN) / (TP + TN + FP + FN)
    print("閾値を平均の中点としたときのacc=", acc2)

    mat = confusion_matrix(y_true, y_pred2)
    print(mat)

    # ROC-AUC
    valLDA_auc = -valLDA_1d

    fpr, tpr, thresholds = roc_curve(y_true, valLDA_auc)
    roc_auc_score = auc(fpr, tpr)

    print("ROC-AUC:", roc_auc_score)

    # ROCプロット
    plt.figure()
    plt.plot(fpr, tpr, marker=".", markersize=8)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.suptitle("ROC Curve", fontsize=20)
    plt.savefig("outputs/lda_roc.png")
    plt.close(fig)

    # PR-AUC
    pre, rec, thresholds = precision_recall_curve(y_true, valLDA_auc)
    pr_score = average_precision_score(y_true, valLDA_auc)
    print("PR-AUC(ave):", pr_score)

    # PRプロット
    plt.figure()
    plt.plot(rec, pre, marker=".", markersize=8)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.suptitle("PR Curve", fontsize=20)
    plt.savefig("outputs/lda_pr.png")
    plt.close()

    # 分類3：平均同士の間を総当たり（各指標）
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    acc = 0
    acc_th = 0
    pre = 0
    pre_th = 0
    tpr = 0
    tpr_th = 0
    fpr = 1
    fpr_th = 0

    i = score1_mean
    while i < score0_mean:
        # 散布図からより大きいほうがclass0
        y_pred3 = (valLDA_1d < i).astype(int)

        TP = np.sum((y_pred3 == 1) & (y_true == 1))
        FP = np.sum((y_pred3 == 1) & (y_true == 0))
        FN = np.sum((y_pred3 == 0) & (y_true == 1))
        TN = np.sum((y_pred3 == 0) & (y_true == 0))

        acc3 = (TP + TN) / (TP + TN + FP + FN)
        pre3 = TP / (TP + FP)
        tpr3 = TP / (TP + FN)
        fpr3 = FP / (TN + FP)
        if acc < acc3:
            acc = acc3
            acc_th = i
        if pre < pre3:
            pre = pre3
            pre_th = i
        if tpr < tpr3:
            tpr = tpr3
            tpr_th = i
        if fpr > fpr3:
            fpr = fpr3
            fpr_th = i

        i += 0.0001

    print("ACC:", acc, "閾値:", acc_th)
    print("PRE:", pre, "閾値:", pre_th)
    print("TPR:", tpr, "閾値:", tpr_th)
    print("FPR:", fpr, "閾値:", fpr_th)


if __name__ == "__main__":
    main()
