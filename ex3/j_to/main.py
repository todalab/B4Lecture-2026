"""Main execution script that runs Gaussian Mixture Model clustering and criteria evaluation on sample datasets."""

import os
from numpy import array
from pandas import read_csv

from gmm import GMM, FIG_DIR
from criteria import evaluate


def apply_GMM(filename: str, K: int):
    """Trains a GMM with a specific number of clusters on a dataset and outputs the clustering visualization."""
    data = read_csv(filename, header=None)
    X = array(data)

    gmm = GMM(K)
    gmm.train(X)

    gmm.plot(X, filename)


def main():
    """Orchestrates the GMM training and criteria evaluation processes for multiple data files."""
    os.makedirs(FIG_DIR, exist_ok=True)

    for filename, k in [
        ("data/data1.csv", 3),
        ("data/data2.csv", 4),
        ("data/data3.csv", 2),
    ]:
        apply_GMM(filename, k)
        evaluate(filename)


if __name__ == "__main__":
    main()
