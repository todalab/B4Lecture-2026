from __future__ import annotations

import numpy as np
from process_data import Array


def confusion_matrix(true_labels: Array, pred_labels: Array, k: int) -> Array:
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(cm: Array) -> float:
    return float(np.trace(cm) / cm.sum())


def evaluate(
    true_labels: Array,
    pred_labels: Array,
    k: int,
) -> tuple[Array, float]:
    cm = confusion_matrix(true_labels, pred_labels, k)
    return cm, accuracy(cm)
