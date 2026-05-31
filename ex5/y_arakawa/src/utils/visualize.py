"""Visualization helpers for spectrogram reconstructions."""

import matplotlib.pyplot as plt
import torch


def plot_recon_pair(gt: torch.Tensor, recon: torch.Tensor) -> plt.Figure:
    """Plot ground truth and reconstruction side by side.

    Parameters
    ----------
    gt : torch.Tensor
        Ground-truth mel spectrogram (n_mels, frames).
    recon : torch.Tensor
        Reconstructed mel spectrogram (n_mels, frames).

    Returns
    -------
    figure : plt.Figure
        Matplotlib figure with two panels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    ax_gt, ax_recon = axes
    ax_gt.imshow(gt.numpy(), origin="lower", aspect="auto")
    ax_gt.set_title("gt")
    ax_gt.set_xlabel("time")
    ax_gt.set_ylabel("mel")
    ax_recon.imshow(recon.numpy(), origin="lower", aspect="auto")
    ax_recon.set_title("recon")
    ax_recon.set_xlabel("time")
    ax_recon.set_ylabel("mel")
    fig.tight_layout()
    return fig
