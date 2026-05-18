from __future__ import annotations

import numpy as np
from hmm.process_data import EPS, Array, Sequence, convert_to_log_params


def _forward_log(
    log_PI: Array,
    A: Array,
    log_B: Array,
    O: Sequence,
) -> tuple[float, Array]:
    prob_PI = np.exp(log_PI)
    prob_B = np.exp(log_B)
    T = len(O)
    l = A.shape[0]
    alpha = np.zeros((T, l))

    alpha[0] = prob_PI * prob_B[:, O[0]]

    for t in range(T - 1):
        alpha[t + 1] = (alpha[t] @ A) * prob_B[:, O[t + 1]]

    log_likelihood = np.log(alpha[-1].sum() + EPS)
    return log_likelihood, alpha


def forward(
    PI: Array,
    A: Array,
    B: Array,
    O: Sequence,
) -> tuple[float, Array]:
    log_PI, log_A, log_B = convert_to_log_params(PI, A, B)
    return _forward_log(log_PI, A, log_B, O)
