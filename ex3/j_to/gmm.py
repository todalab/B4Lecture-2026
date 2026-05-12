"""Implement the Gaussian Mixture Model (GMM) and Expectation-Maximization (EM) algorithm."""

import os
from matplotlib.pyplot import (
    figure,
    plot,
    scatter,
    subplot,
    gca,
    savefig,
    title,
    xlabel,
    ylabel,
    legend,
    grid,
)
from matplotlib.patches import Ellipse
from numpy import (
    angle,
    array,
    diag,
    eye,
    inf,
    log,
    ones,
    exp,
    zeros,
    sqrt,
)
from numpy.typing import NDArray
from numpy.linalg import eigh
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

FIG_DIR = "fig"


class GMM:
    def __init__(self, K: int):
        """Initialise the GMM instance with the specified number of clusters K."""
        self.K: int = K

        self.phi: NDArray
        self.mu: NDArray
        self.Sigma: NDArray

        self.log_likelihoods: list[float] = []

        self.w: NDArray

    def initialise(self, X):
        """Initialise the cluster means using a K-means++ strategy, along with uniform weights and identity covariances."""
        _, D = X.shape

        # Assume homogeneous portion of data for each clustering
        self.phi = ones(self.K) / self.K

        mu = [zeros(D)]
        for _ in range(self.K - 1):
            distance_2 = array([((X - mu) ** 2).sum(axis=1) for mu in mu]).min(axis=0)
            mu.append(X[distance_2.argmax()])
        self.mu = array(mu)

        self.Sigma = array([eye(D) for _ in range(self.K)])

    def e_step(self, X):
        """Perform the Expectation step."""
        all_log_likelihood_prior = array(
            [
                log(self.phi[k])
                + multivariate_normal.logpdf(X, mean=self.mu[k], cov=self.Sigma[k])
                for k in range(self.K)
            ]
        )

        all_log_evidence = array(logsumexp(all_log_likelihood_prior, axis=0))
        log_likelihood = all_log_evidence.sum()
        w = exp(all_log_likelihood_prior - all_log_evidence)
        return w, log_likelihood

    def m_step(self, X, w):
        """Perform the Maximization step."""
        N, D = X.shape
        w_sum_for_data = w.sum(axis=1)
        self.phi = w_sum_for_data / N
        self.mu = array([w[k] @ X / w_sum_for_data[k] for k in range(self.K)])
        self.Sigma = array(
            [
                (X - self.mu[k]).T @ diag(w[k]) @ (X - self.mu[k]) / w_sum_for_data[k]
                + 1e-6 * eye(D)
                for k in range(self.K)
            ]
        )

    def train(self, X, tolerance=1e-4):
        """Train the GMM on the dataset using the EM algorithm until the log-likelihood converges."""
        self.initialise(X)
        log_likelihoods = [-inf]
        while True:
            w, log_likelihood = self.e_step(X)
            if log_likelihood - log_likelihoods[-1] < tolerance:
                self.w = w
                break
            log_likelihoods.append(log_likelihood)

            self.m_step(X, w)

        self.log_likelihoods = log_likelihoods[1:]  # Skip -inf

    def predict(self, X):
        """Predict the most likely cluster assignment for each data point based on the trained model."""
        w, _ = self.e_step(X)
        return w.argmax(axis=0)

    def plot(self, X, filename):
        """Plot the clustered data, Gaussian ellipses, and log-likelihood convergence."""
        basename = os.path.basename(filename)

        # Plot original unclustered scatter points
        figure(figsize=(8, 6))
        title(f"{basename}")
        xlabel("x_0")
        ylabel("x_1")
        scatter(*X.T, alpha=3 / 4)

        scatter_out = f'scatter_{basename.replace(".csv", ".png")}'
        savefig(os.path.join(FIG_DIR, scatter_out))

        # Plot GMM clustering results
        figure(figsize=(15, 6))

        subplot(1, 2, 1)
        title(f"GMM Clustering (K={self.K})")
        xlabel("x_0")
        ylabel("x_1")

        z = self.predict(X)
        for k in range(self.K):
            scatter(*X[z == k].T, c=f"C{k}", label=f"Cluster {k}", alpha=3 / 4)

        for k in range(self.K):
            lam, u = eigh(self.Sigma[k])
            main_axis = u.T[0]
            arg = angle(complex(*main_axis), deg=True)
            width, height = 2 * sqrt(lam)

            for sigma in [1, 2]:
                gca().add_patch(
                    Ellipse(
                        xy=self.mu[k],
                        width=width * sigma,
                        height=height * sigma,
                        angle=arg,
                        facecolor=f"C{k}",
                        edgecolor="black",
                        alpha=1 / 4,
                    )
                )

        scatter(
            *self.mu.T,
            c=[f"C{k}" for k in range(self.K)],
            marker="*",
            edgecolors="black",
            label="mu",
        )
        legend()

        # Plot Log-Likelihood Convergence
        subplot(1, 2, 2)
        title("Log-Likelihood Convergence")
        xlabel("Iter")
        ylabel("Log-Likelihood")
        plot(self.log_likelihoods, marker="o")
        grid(True)

        out_filename = f'GMM_EM_{basename.replace(".csv", ".png")}'
        savefig(os.path.join(FIG_DIR, out_filename))
