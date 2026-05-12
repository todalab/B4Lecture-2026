"""Provide model evaluation criteria (A.I.C. and B.I.C.) to determine the optimal number of clusters for a dataset."""

import os
from matplotlib.pyplot import (
    figure,
    plot,
    title,
    xlabel,
    ylabel,
    legend,
    grid,
    savefig,
    axvline,
)
from numpy import array, log
from numpy.typing import NDArray
from pandas import read_csv

from gmm import GMM, FIG_DIR


def n_params(K: int, D: int) -> int:
    """Calculate the total number of free parameters."""
    return int(K * D + K * D * (D + 1) / 2 + K - 1)


def AIC(gmm: GMM, X: NDArray) -> float:
    """Compute the A.I.C."""
    _, D = X.shape
    L = gmm.log_likelihoods[-1]
    return -2 * L + 2 * n_params(gmm.K, D)


def BIC(gmm: GMM, X: NDArray) -> float:
    """Compute the B.I.C."""
    N, D = X.shape
    L = gmm.log_likelihoods[-1]
    return -2 * L + n_params(gmm.K, D) * log(N)


def evaluate(filename: str, max_K: int = 8):
    """Evaluate and plots A.I.C. and B.I.C. scores across a range of cluster counts to identify the optimal K."""
    basename = os.path.basename(filename)
    data = read_csv(filename, header=None)
    X = array(data)

    criteria = {
        AIC: [],
        BIC: [],
    }

    K_range = range(1, max_K + 1)
    for K in K_range:
        gmm = GMM(K)
        gmm.train(X)
        for f, criterion in criteria.items():
            criterion.append(f(gmm, X))

    figure(figsize=(8, 6))

    for f, criterion in criteria.items():
        name = f.__name__
        min_K = K_range[array(criterion).argmin()]

        line = plot(K_range, criterion, marker="o", label=name)

        # Highlight the min K
        axvline(
            min_K,
            color=line[0].get_color(),
            linestyle=":",
            label=f"min {name} (K = {min_K})",
        )

    title(f"AIC and BIC ({basename})")
    xlabel("Number of Clusters (K)")
    ylabel("Information Criterion")
    legend()
    grid(True)

    out_filename = f'AIC_BIC_{basename.replace(".csv", ".png")}'
    savefig(os.path.join(FIG_DIR, out_filename))
