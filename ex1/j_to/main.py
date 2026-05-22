"""線形回帰・多項式回帰・3次元回帰・ロジスティック回帰（勾配降下法）の実装."""

from inspect import signature
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy import column_stack, linspace, meshgrid, ones_like
from numpy.linalg import inv, norm
from pandas import read_csv


def regression(
    fig: Figure, index: int, total: int, filename: str, title: str, poly: Callable
):
    """回帰分析を行い、結果をプロットする."""
    # Read data from file
    data = read_csv(filename)

    # Convert to vectors (inputs, target)
    has_z = True if len(signature(poly).parameters) == 2 else False
    columns = "xy" + ("z" if has_z else "")
    *inputs, target = (data[key].to_numpy() for key in columns)

    # Stack all-1 vector for bias
    def augment(inputs):
        return column_stack((*poly(*inputs), ones_like(inputs[0])))

    X = augment(inputs)

    # Calculate coëfficients
    coeffs = inv(X.T @ X) @ X.T @ target

    # Subplot
    ax = fig.add_subplot(1, total, index, projection="3d" if has_z else None)
    ax.set_title(title)

    # Display scatters points
    ax.scatter(*inputs, target, label="Data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if has_z:
        ax.set_zlabel("z")

    # Display approximation
    spaces = [linspace(axis.min(), axis.max()) for axis in inputs]
    if has_z:
        spaces = [space.flatten() for space in meshgrid(*spaces)]
    plot = ax.plot_trisurf if has_z else ax.plot
    plot(
        *spaces, augment(spaces) @ coeffs, color="red", alpha=0.5, label="Approximation"
    )

    # Display labels
    ax.legend()


def sigmoid(x):
    """シグモイド関数."""
    return 1 / (1 + np.exp(-x))


def gradient_descent(filename: str):
    """勾配降下法でロジスティック回帰を行う."""
    # Read data from file
    data = read_csv(filename)

    x1 = data["x1"]
    x2 = data["x2"]
    y = data["y"]
    n = x1.shape[0]

    # Stack all-1 vector for bias
    X = column_stack((x1, x2, np.ones_like(x1)))
    w = np.random.rand(3)  # weights and bias

    losses = []
    likelihoods = []
    accuracies = []

    iteration = 0
    eta = 0.5
    while True:
        yh = sigmoid(X @ w)
        gradient_sum = X.T @ (yh - y)
        w_ = w - eta / n * gradient_sum

        # Metrics
        likelihood = np.sum(y * np.log(yh) + (1 - y) * np.log(1 - yh))
        loss = -likelihood / n

        correct_count = np.sum(np.abs(yh - y) < 0.5)
        accuracy = correct_count / n

        losses.append(loss)
        likelihoods.append(likelihood)
        accuracies.append(accuracy)

        iteration += 1
        if norm(w - w_) < 1e-5:
            break

        w = w_

    iterations = range(iteration)

    # Subplots
    fig, axes = plt.subplots(3, 1, figsize=[12, 12], sharex=True)
    loss_ax, likelihood_ax, accuracy_ax = axes
    for ax in axes:
        ax.grid(True, alpha=0.5)

    loss_ax.plot(iterations, losses, color="red")
    loss_ax.set_ylabel("Loss")

    likelihood_ax.plot(iterations, likelihoods)
    likelihood_ax.set_ylabel("Log Likeliness")

    accuracy_ax.plot(iterations, accuracies, color="green")
    accuracy_ax.set_ylabel("Accuracy")
    accuracy_ax.set_xlabel("Iteration")

    fig.tight_layout()
    fig.savefig("2_gradient_descent_metrics.png")


def main():
    """メイン関数."""
    fig = plt.figure(figsize=[24, 6])  # Size in inches

    regression(fig, 1, 3, "data/sample2d_1.csv", "Linear 2D", lambda x: (x,))
    regression(
        fig, 2, 3, "data/sample2d_2.csv", "Polynomial 2D", lambda x: (x**3, x**2, x)
    )
    regression(
        fig,
        3,
        3,
        "data/sample3d.csv",
        "Polynomial 3D",
        lambda x, y: (x**2, y**2, x * y, x, y),
    )

    fig.tight_layout()
    fig.savefig("1_regression.png")

    gradient_descent("data/sample_logistic.csv")


if __name__ == "__main__":
    main()
