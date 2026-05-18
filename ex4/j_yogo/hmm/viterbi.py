from __future__ import annotations

import numpy as np
from hmm.process_data import Array, Sequence, convert_to_log_params


def _viterbi_log(
    log_PI: Array,
    log_A: Array,
    log_B: Array,
    O: Sequence,
) -> tuple[float, list[int], Array]:
    l = log_A.shape[0]
    T = len(O)

    delta = np.zeros((T, l))
    psi = np.zeros((T, l), dtype=int)

    delta[0] = log_PI + log_B[:, O[0]]

    for t in range(T - 1):
        trans = delta[t][:, None] + log_A
        psi[t + 1] = np.argmax(trans, axis=0)
        delta[t + 1] = np.max(trans, axis=0) + log_B[:, O[t + 1]]

    log_prob = float(np.max(delta[-1]))
    best_last = int(np.argmax(delta[-1]))

    best_path = [best_last]
    for t in range(T - 1, 0, -1):
        best_path.append(int(psi[t][best_path[-1]]))
    best_path.reverse()

    return log_prob, best_path, delta


def viterbi(
    PI: Array,
    A: Array,
    B: Array,
    O: Sequence,
) -> tuple[float, list[int], Array]:
    log_PI, log_A, log_B = convert_to_log_params(PI, A, B)
    return _viterbi_log(log_PI, log_A, log_B, O)
