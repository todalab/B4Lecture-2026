import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def main():
    # sample2d_1.csv の処理
    df1 = pd.read_csv("sample2d_1.csv", header=0)
    x1, y1=df1["x"], df1["y"]
    X1 = np.vstack([np.ones(len(x1)), x1]).T
    w1 = normal_equation(X1, y1)

    # sample2d_1.csv の可視化
    x1_line = np.linspace(x1.min(), x1.max(), 100)
    y1_line = w1[0] + w1[1] * x1_line
    plt.figure()
    plt.scatter(x1, y1, label='Data')
    plt.plot(x1_line, y1_line, color='red', label='Fit')
    plt.legend()
    plt.show()

    # sample2d_2.csv の処理
    df2 = pd.read_csv("sample2d_2.csv", header=0)
    x2, y2=df2["x"], df2["y"]
    X2 = np.vstack([np.ones(len(x2)), x2, x2**2, x2**3]).T
    w2 = normal_equation(X2, y2)

    # sample2d_2.csv の可視化
    x2_line = np.linspace(x2.min(), x2.max(), 100)
    X2_line = np.vstack([np.ones(len(x2_line)), x2_line, x2_line**2, x2_line**3]).T
    y2_line = X2_line @ w2
    plt.figure()
    plt.scatter(x2, y2, label='Data')
    plt.plot(x2_line, y2_line, color='red', label='Fit')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()