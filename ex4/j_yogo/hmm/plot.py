from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from hmm.process_data import Array


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
