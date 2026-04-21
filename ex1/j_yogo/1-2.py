import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from 1-1 import normal_equation

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    df = pd.read_csv("../data/sample_logistics.csv", header=0)
    x1, x2=df["x1"], df["x2"]
    y = df["y"]

    X = np.vstack([np.ones(len(x1)), x1, x2]).T

    #パラメータ
    lr = 0.1
    iterations = 300
    w_log = np.zeros(X.shape[1])
    eps = 1e-15

    history = {'loss': [], 'log-likelihood': [], 'accuracy': []}

    # 勾配降下法
    for _ in range(iterations):
        y_pred = sigmoid(X @ w_log)
        
        # 指標の計算
        loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
        log = np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
        acc = np.mean((y >= 0.5).astype(int) == y)
        
        history['loss'].append(loss)
        history['log-likelihood'].append(log)
        history['accuracy'].append(acc)
        
        # 勾配の計算と重みの更新
        grad = X.T @ (y_pred - y) / len(y)
        w_log -= lr * grad

if __name__ == "__main__":
    main()