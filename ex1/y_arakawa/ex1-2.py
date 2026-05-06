# -*- coding: utf-8 -*-
"""1-2 ロジスティック回帰（勾配降下法）."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "sans-serif"

# CSVファイルからデータを読み込む
sample = pd.read_csv("../data/sample_logistic.csv")

# データを可視化
fig, ax = plt.subplots()
sample_y0 = sample[sample["y"] == 0]
sample_y1 = sample[sample["y"] == 1]

ax.scatter(sample_y0["x1"], sample_y0["x2"], color="blue", label="y=0")
ax.scatter(sample_y1["x1"], sample_y1["x2"], color="red", label="y=1")
ax.legend(loc="upper left")
ax.set_title("sample_logistic scatter plot")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.savefig("output/sample_logistic.png")
plt.show()

# 勾配降下法によるロジスティック回帰
X = sample.iloc[:, 0:2].to_numpy()
y = sample.iloc[:, 2:3].to_numpy()

# 重みとバイアスの初期化
w = np.array([[0.1], [0.2]])
b = 0.0

# 学習率
learning_rate = 1


# シグモイド関数
def sigmoid(x):
    """Summary.

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1 / (1 + np.exp(-x))


# 損失関数（勾配）
def logLossGrad(n, w, b, X, y):
    """Summary.

    Args:
        n (_type_): _description_
        w (_type_): _description_
        b (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    pred = sigmoid(X @ w + b)
    diff = pred - y
    grad_w = (X.T @ diff) / n
    grad_b = np.sum(diff) / n
    return grad_w, grad_b


# 対数尤度
def logLikelihood(n, w, b, X, y):
    """Summary.

    Args:
        n (_type_): _description_
        w (_type_): _description_
        b (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    pred = sigmoid(X @ w + b)
    pred = np.clip(pred, 1e-12, 1 - 1e-12)
    return np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))


# 対数損失
def logLoss(n, w, b, X, y):
    """Summary.

    Args:
        n (_type_): _description_
        w (_type_): _description_
        b (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    return -logLikelihood(n, w, b, X, y) / n


# 分類精度
def categorize_accuracy(n, w, b, X, y):
    """Summary.

    Args:
        n (_type_): _description_
        w (_type_): _description_
        b (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    pred_label = (sigmoid(X @ w + b) >= 0.5).astype(int)
    return np.sum(pred_label == y) / n


# 学習
n = X.shape[0]
w_history, loss_history = [w.copy().T], [logLoss(n, w, b, X, y)]
logLikelihood_history = [logLikelihood(n, w, b, X, y)]
categorize_accuracy_history = [categorize_accuracy(n, w, b, X, y)]

for i in range(1000000):
    grad_w, grad_b = logLossGrad(n, w, b, X, y)
    w_next = w - learning_rate * grad_w
    b_next = b - learning_rate * grad_b
    loss_t = logLoss(n, w_next, b_next, X, y)

    # 収束判定
    if loss_t >= loss_history[-1]:
        break

    w, b = w_next, b_next
    w_history.append(w.copy().T)
    loss_history.append(loss_t)
    logLikelihood_history.append(logLikelihood(n, w, b, X, y))
    categorize_accuracy_history.append(categorize_accuracy(n, w, b, X, y))

print(f"final loss: {loss_history[-1]:.6f}")
print(f"accuracy: {categorize_accuracy(n, w, b, X, y):.4f}")

# 学習過程の可視化
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(loss_history)
ax[0].set_title("Loss")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Loss")
ax[1].plot(logLikelihood_history)
ax[1].set_title("Log-Likelihood")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Log-Likelihood")
ax[2].plot(categorize_accuracy_history)
ax[2].set_title("Categorize Accuracy")
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Accuracy")
plt.tight_layout()
plt.savefig("output/logistic_learning_process.png")
plt.show()
