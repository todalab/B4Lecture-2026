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
    gt_np = gt.numpy()
    recon_np = recon.numpy()
    vmin = min(float(gt_np.min()), float(recon_np.min()))
    vmax = max(float(gt_np.max()), float(recon_np.max()))

    ax_gt.imshow(gt_np, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax_gt.set_title("gt")
    ax_gt.set_xlabel("time")
    ax_gt.set_ylabel("mel")
    ax_recon.imshow(recon_np, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax_recon.set_title("recon")
    ax_recon.set_xlabel("time")
    ax_recon.set_ylabel("mel")
    fig.tight_layout()
    return fig
