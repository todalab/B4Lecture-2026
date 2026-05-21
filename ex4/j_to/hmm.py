"""H.M.M. scoring utilities."""

from numpy import zeros


def forward(o, pi, a, b):
    """Return H.M.M. forward likelihood."""
    t = len(o)
    n = a.shape[0]

    alpha = zeros((t, n))
    alpha[0] = pi[:, 0] * b[:, o[0]]

    for i in range(t - 1):
        alpha[i + 1] = (alpha[i] @ a) * b[:, o[i + 1]]

    return alpha[-1].sum()


def viterbi(o, pi, a, b):
    """Return H.M.M. Viterbi likelihood."""
    t = len(o)
    n = a.shape[0]

    delta = zeros((t, n))
    delta[0] = pi[:, 0] * b[:, o[0]]

    for i in range(t - 1):
        for j in range(n):
            val = delta[i] * a[:, j]
            delta[i + 1, j] = val.max() * b[j, o[i + 1]]

    return delta[-1].max()
