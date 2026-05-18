"""
Forward アルゴリズムの実装.

Forward アルゴリズムは、与えられた観測系列の対数尤度を計算.
"""

from __future__ import annotations

import numpy as np
from hmm.process_data import EPS, Array, Sequence, convert_to_log_params


def _forward_log(
    log_PI: Array,
    A: Array,
    log_B: Array,
    O: Sequence,
) -> tuple[float, Array]:
    """
    対数スケールで Forward アルゴリズムを実行します.

    対数領域での計算により、アンダーフローを防ぐ.

    Args:
        log_PI (Array): 初期状態確率（対数領域）
                       shape: (l,), log_PI[i] = log(π_i)
        A (Array): 状態遷移確率行列（通常領域）
                  shape: (l, l), A[i, j] = P(X_{t+1}=j | X_t=i)
        log_B (Array): 出力確率行列（対数領域）
                      shape: (l, n_symbols)
                      log_B[i, o] = log(b_i(o))
        O (Sequence): 観測系列
                     shape: (T,), O[t] = t 番目の観測シンボル（0～n_symbols-1）

    Returns:
        tuple[float, Array]: (対数尤度, フォワード確率行列)
            - 対数尤度: float, log P(O | λ)
            - フォワード確率: shape (T, l), 各時刻・状態でのフォワード確率
    """
    # 対数領域から通常領域へ変換（出力確率のみ）
    prob_PI = np.exp(log_PI)
    prob_B = np.exp(log_B)
    T = len(O)  # 観測系列の長さ
    l = A.shape[0]  # 状態数

    # フォワード変数
    alpha = np.zeros((T, l))

    # t=0 での初期化: alpha[0, i] = π_i * b_i(O_0)
    alpha[0] = prob_PI * prob_B[:, O[0]]

    # t=1～T-1 での再帰的な更新
    for t in range(T - 1):
        # 前時刻のフォワード確率と遷移確率の積を計算
        alpha[t + 1] = (alpha[t] @ A) * prob_B[:, O[t + 1]]

    # 最終時刻での確率をすべて加算して尤度を計算
    log_likelihood = np.log(alpha[-1].sum() + EPS)
    return log_likelihood, alpha


def forward(
    PI: Array,
    A: Array,
    B: Array,
    O: Sequence,
) -> tuple[float, Array]:
    """
    Forward アルゴリズムを実行.

    通常領域のHMMパラメータを対数領域に変換した後、
    _forward_log を実行します.

    Args:
        PI (Array): 初期状態確率（通常領域）
                   shape: (l,), PI[i] = π_i
        A (Array): 状態遷移確率行列
                  shape: (l, l), A[i, j] = P(X_{t+1}=j | X_t=i)
        B (Array): 出力確率行列
                  shape: (l, n_symbols), B[i, o] = b_i(o)
        O (Sequence): 観測系列
                     shape: (T,), O[t] = t 番目の観測シンボル

    Returns:
        tuple[float, Array]: (対数尤度, フォワード確率行列)
            - 対数尤度: float
            - フォワード確率: shape (T, l)
    """
    # HMMパラメータを対数領域に変換
    log_PI, log_A, log_B = convert_to_log_params(PI, A, B)
    # 対数領域での Forward アルゴリズムを実行
    return _forward_log(log_PI, A, log_B, O)
