"""
Viterbi アルゴリズムの実装

Viterbi アルゴリズムは、与えられた観測系列に対する最尤状態系列（最も確率の高い隠れ状態の系列）を見つける.
"""

from __future__ import annotations

import numpy as np
from hmm.process_data import Array, Sequence, convert_to_log_params


def _viterbi_log(
    log_PI: Array,
    log_A: Array,
    log_B: Array,
    O: Sequence,
) -> tuple[float, list[int], Array]:
    """
    対数スケールで Viterbi アルゴリズムを実行.

    対数領域での計算により、アンダーフローを防ぐ.
    最尤状態系列を動的計画法により求める.

    Args:
        log_PI (Array): 初期状態確率（対数領域）
                       shape: (l,)
        log_A (Array): 状態遷移確率行列（対数領域）
                      shape: (l, l)
        log_B (Array): 出力確率行列（対数領域）
                      shape: (l, n_symbols)
        O (Sequence): 観測系列
                     shape: (T,)

    Returns:
        tuple[float, list[int], Array]: (最高対数確率, 最尤状態系列, delta行列)
            - 最高対数確率: float, log P(最尤状態系列, O | λ)
            - 最尤状態系列: list[int], 最も確率の高い隠れ状態の系列
            - delta: shape (T, l), 各時刻・状態での最高確率
    """
    l = log_A.shape[0]  # 状態数
    T = len(O)  # 観測系列の長さ

    # Viterbi 変数:
    delta = np.zeros((T, l))

    psi = np.zeros((T, l), dtype=int)

    # t=0 での初期化
    delta[0] = log_PI + log_B[:, O[0]]

    # t=1～T-1 での再帰的な更新
    for t in range(T - 1):
        # 各次状態 j に対して、前状態 i からの最大スコアを計算
        trans = delta[t][:, None] + log_A
        # 各 j について、最大を与える i を記録
        psi[t + 1] = np.argmax(trans, axis=0)
        # 最大値と出力確率を組み合わせて次時刻の delta を更新
        delta[t + 1] = np.max(trans, axis=0) + log_B[:, O[t + 1]]

    # 最終時刻での最高確率と最終状態を取得
    log_prob = float(np.max(delta[-1]))
    best_last = int(np.argmax(delta[-1]))

    # バックトレース: 最終状態から遡って最尤状態系列を構築
    best_path = [best_last]
    for t in range(T - 1, 0, -1):
        # 現在の状態の前の状態を psi から取得
        best_path.append(int(psi[t][best_path[-1]]))
    # リスト逆順なので反転
    best_path.reverse()

    return log_prob, best_path, delta


def viterbi(
    PI: Array,
    A: Array,
    B: Array,
    O: Sequence,
) -> tuple[float, list[int], Array]:
    """
    Viterbi アルゴリズムを実行します（公開インターフェース）.

    通常領域のHMMパラメータを対数領域に変換した後、
    _viterbi_log を実行します.

    Args:
        PI (Array): 初期状態確率（通常領域）
                   shape: (l,)
        A (Array): 状態遷移確率行列
                  shape: (l, l)
        B (Array): 出力確率行列
                  shape: (l, n_symbols)
        O (Sequence): 観測系列
                     shape: (T,)

    Returns:
        tuple[float, list[int], Array]: (最高対数確率, 最尤状態系列, delta行列)
            - 最高対数確率: float
            - 最尤状態系列: list[int]
            - delta: shape (T, l)
    """
    # HMMパラメータを対数領域に変換
    log_PI, log_A, log_B = convert_to_log_params(PI, A, B)
    # 対数領域での Viterbi アルゴリズムを実行
    return _viterbi_log(log_PI, log_A, log_B, O)
