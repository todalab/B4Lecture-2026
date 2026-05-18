from __future__ import annotations

from typing import Literal

import numpy as np
from forward import _forward_log
from process_data import Array, Sequence, convert_to_log_params


def score_sequences(
    outputs: list[Sequence],
    PI_all: Array,
    A_all: Array,
    B_all: Array,
    method: Literal["forward", "viterbi"] = "forward",
) -> tuple[Array, list | None]:
    k = PI_all.shape[0]
    p = len(outputs)
    scores = np.zeros((p, k))
    best_paths = [[None] * k for _ in range(p)] if method == "viterbi" else None

    log_params = [
        convert_to_log_params(PI_all[m], A_all[m], B_all[m]) for m in range(k)
    ]

    for seq_idx, O in enumerate(outputs):
        for m_idx in range(k):
            log_PI, log_A, log_B = log_params[m_idx]
            A = A_all[m_idx]

            score, _ = _forward_log(log_PI, A, log_B, O)

            scores[seq_idx, m_idx] = score

    predictions = np.argmax(scores, axis=1)
    return predictions, scores, best_paths
