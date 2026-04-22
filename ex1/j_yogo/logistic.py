import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sigmoid(x):
    """シグモイド関数.

    Args:
        x (np.ndarray): 入力

    Returns:
        np.ndarray: シグモイド関数の出力
    """
    return 1 / (1 + np.exp(-x))


def main():
    """勾配降下法を用いたロジスティック回帰モデルの学習と評価、可視化を実行する."""
    df = pd.read_csv("../data/sample_logistic.csv")
    x1, x2 = df["x1"], df["x2"]
    y = df["y"]
    X = np.vstack([np.ones(len(x1)), x1, x2]).T

    # パラメータ
    lr = 1
    iterations = 300
    w_log = np.zeros(X.shape[1])

    history = {"loss": [], "log_likelihood": [], "accuracy": []}

    # 勾配降下法
    for _ in range(iterations):
        y_pred = sigmoid(X @ w_log)

        # 損失、対数尤度、正解率の算出
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        log_likelihood = np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        acc = np.mean((y_pred >= 0.5).astype(int) == y)

        history["loss"].append(loss)
        history["log_likelihood"].append(log_likelihood)
        history["accuracy"].append(acc)

        # 勾配の計算と重みの更新
        grad = X.T @ (y_pred - y) / len(y)
        w_log -= lr * grad

    # 学習曲線の可視化
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    fig.suptitle(f"Learning Rate ={lr}")
    axes[0].plot(history["loss"], color="red")
    axes[0].set_ylabel("Loss")

    axes[1].plot(history["log_likelihood"], color="blue")
    axes[1].set_ylabel("Log_likelihood")

    axes[2].plot(history["accuracy"], color="green")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
