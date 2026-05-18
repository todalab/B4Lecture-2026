from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

Array = np.ndarray
Sequence = list[int]

EPS = 1e-300


def convert_to_log_params(
    PI: Array,
    A: Array,
    B: Array,
) -> tuple[Array, Array, Array]:
    log_PI = np.log(PI[:, 0] + EPS)
    log_A = np.log(A + EPS)
    log_B = np.log(B + EPS)
    return log_PI, log_A, log_B


def load_dataset(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_hmm_params(data: dict) -> tuple:
    models = data["models"]
    PI_all = models["PI"]
    A_all = models["A"]
    B_all = models["B"]
    outputs = data["output"]
    true_labels = np.array(data["answer_models"])
    k = PI_all.shape[0]
    return outputs, PI_all, A_all, B_all, true_labels, k
