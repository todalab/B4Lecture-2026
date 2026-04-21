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
    
    # sample3d.csv の処理
    df3 = pd.read_csv('sample3d.csv')
    x3 = df3['x']
    y3 = df3['y']
    z3 = df3['z']
    X3 = np.vstack([np.ones(len(x3)), x3, y3, x3**2, y3**2, x3*y3]).T
    w3 = normal_equation(X3, z3)

    # sample3d の可視化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x3, y3, z3, label='Data')

    x_grid, y_grid = np.meshgrid(np.linspace(x3.min(), x3.max(), 100),
                                np.linspace(y3.min(), y3.max(), 100))
    X3_grid = np.vstack([np.ones(x_grid.size), x_grid.ravel(), y_grid.ravel(),
                        x_grid.ravel()**2, y_grid.ravel()**2, x_grid.ravel() * y_grid.ravel()]).T
    z_grid = (X3_grid @ w3).reshape(x_grid.shape)

    ax.plot_surface(x_grid, y_grid, z_grid, color='red')
    plt.show()

if __name__ == "__main__":
    main()