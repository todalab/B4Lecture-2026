"""Implement the Gaussian Mixture Model (G.M.M.) and Expectation-Maximization (EM) algorithm."""

import os

from matplotlib.patches import Ellipse
from matplotlib.pyplot import (figure, gca, grid, legend, plot, savefig,
                               scatter, subplot, title, xlabel, ylabel)
from numpy import angle, array, diag, exp, eye, inf, log, ones, sqrt, zeros
from numpy.linalg import eigh
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

FIG_DIR = "fig"


class GMM:
    """Gaussian Mixture Model (G.M.M.) class for clustering."""

    def __init__(self, K: int):
        """Initialise the G.M.M. instance with the specified number of clusters K."""
        self.K: int = K

        self.phi: NDArray
        self.mu: NDArray
        self.Sigma: NDArray

        self.log_likelihoods: list[float] = []

        self.w: NDArray

    def initialise(self, X):
        """Initialise using K-means++, uniform weights, and identity covariances."""
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
        """Perform the Maximisation step."""
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
        """Train the G.M.M. using the EM algorithm until the log-likelihood converges."""
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
        """Predict the most likely cluster assignment for each data point."""
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

        # Plot G.M.M. clustering results
        figure(figsize=(15, 6))

        subplot(1, 2, 1)
        title(f"G.M.M. Clustering (K={self.K})")
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
