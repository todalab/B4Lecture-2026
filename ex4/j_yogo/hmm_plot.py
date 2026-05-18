from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from process_data import Array


def plot_confusion_matrix(
    cm: Array,
    title: str,
    ax: plt.Axes,
    cmap: str = "Blues",
) -> None:
    k = cm.shape[0]
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    labels = [f"m{i}" for i in range(k)]
    ax.set(
        xticks=np.arange(k),
        yticks=np.arange(k),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    thresh = cm.max() / 2.0
    for i in range(k):
        for j in range(k):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )


def plot_comparison(
    results: dict[str, dict],
    save_path: str | Path | None = None,
) -> plt.Figure:
    dnames = list(results.keys())
    fwd_accs = [results[d]["fwd_acc"] for d in dnames]
    vtb_accs = [results[d]["vtb_acc"] for d in dnames]
    fwd_times = [results[d]["fwd_time"] for d in dnames]
    vtb_times = [results[d]["vtb_time"] for d in dnames]

    x, w = np.arange(len(dnames)), 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Forward vs Viterbi: Accuracy & Computation Time",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    ax.bar(x - w / 2, fwd_accs, w, label="Forward", color="steelblue")
    ax.bar(x + w / 2, vtb_accs, w, label="Viterbi", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(dnames, rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy Comparison")
    ax.legend()
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    for xi, (fa, va) in enumerate(zip(fwd_accs, vtb_accs)):
        ax.text(xi - w / 2, fa + 0.01, f"{fa:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, va + 0.01, f"{va:.3f}", ha="center", fontsize=8)

    ax = axes[1]
    ax.bar(x - w / 2, fwd_times, w, label="Forward", color="steelblue")
    ax.bar(x + w / 2, vtb_times, w, label="Viterbi", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(dnames, rotation=15)
    ax.set_ylabel("Time (s)")
    ax.set_title("Computation Time Comparison")
    ax.legend()
    max_t = max(fwd_times + vtb_times)
    for xi, (ft, vt) in enumerate(zip(fwd_times, vtb_times)):
        ax.text(xi - w / 2, ft + max_t * 0.01, f"{ft:.3f}s", ha="center", fontsize=8)
        ax.text(xi + w / 2, vt + max_t * 0.01, f"{vt:.3f}s", ha="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_state_sequence(
    state_seq: list[int],
    obs_seq: list[int],
    title: str,
    ax: plt.Axes,
) -> None:
    T = len(state_seq)
    ax.step(
        range(T),
        state_seq,
        where="mid",
        color="steelblue",
        linewidth=2,
        label="State (Viterbi)",
    )
    ax.scatter(range(T), obs_seq, color="orange", zorder=5, s=60, label="Observation")
    ax.set_xlabel("Time step")
    ax.set_ylabel("State / Symbol index")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
