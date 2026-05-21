"""H.M.M. evaluation and normalised plotting helpers."""

import os
import pickle
from time import perf_counter

from matplotlib.pyplot import (
    colorbar,
    figure,
    imshow,
    savefig,
    subplot,
    suptitle,
    text,
    tight_layout,
    title,
    xlabel,
    xticks,
    ylabel,
    yticks,
)
from sklearn.metrics import confusion_matrix

from hmm import forward, viterbi

FIG_DIR = "fig"


def evaluate(data_path):
    """Evaluate H.M.M. models for a labelled dataset."""
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    answer_models = data["answer_models"]
    outputs = data["output"]
    pi = data["models"]["PI"]
    a = data["models"]["A"]
    b = data["models"]["B"]

    K = len(pi)
    labels = list(range(K))
    model_params = list(zip(pi, a, b))

    def predict(outputs, f):
        """Return H.M.M. predictions with timing."""
        start = perf_counter()
        predictions = [
            max(range(K), key=lambda k: f(o, *model_params[k])) for o in outputs
        ]
        return predictions, perf_counter() - start

    def accuracy(true, prediction):
        """Return H.M.M. accuracy normalised by length."""
        return sum(a == b for a, b in zip(true, prediction)) / len(true)

    basename = os.path.basename(data_path).split(".")[0]

    figure(figsize=(12, 6))
    for index, f in enumerate((forward, viterbi), start=1):
        pred, elapsed = predict(outputs, f)
        matrix = confusion_matrix(answer_models, pred, labels=labels)
        acc = accuracy(answer_models, pred)
        plot(matrix, f.__name__, labels, index, acc, elapsed)
    suptitle(f"H.M.M. Model Estimation – {basename}")

    tight_layout()
    out_path = os.path.join(FIG_DIR, f"{basename}_hmm_estimation.png")
    savefig(out_path)


def plot(matrix, plot_title, labels, index, acc, elapsed):
    """Plot H.M.M. confusion matrix with normalised text colour."""
    subplot(1, 2, index)

    im = imshow(matrix, cmap="Blues")
    title(f"{plot_title}\nAcc={acc:.3f} Time={elapsed:.4f}s")
    xlabel("Predicted model")
    ylabel("True model")
    xticks(range(len(labels)), labels)
    yticks(range(len(labels)), labels)

    colorbar(im)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = im.norm(matrix[i, j])
            color = "white" if value > 0.5 else "black"
            text(j, i, matrix[i, j], ha="center", va="center", color=color)
