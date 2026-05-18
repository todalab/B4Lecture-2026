"""
複数HMMモデルの系列スコアリングモジュール

このモジュールは、複数の異なるHMMモデルを用いて、
観測系列ごとに各モデルのスコア（対数尤度または対数確率）を計算し、
最も高いスコアを持つモデルで分類.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from hmm.forward import _forward_log
from hmm.process_data import Array, Sequence, convert_to_log_params
from hmm.viterbi import _viterbi_log


def score_sequences(
    outputs: list[Sequence],
    PI_all: Array,
    A_all: Array,
    B_all: Array,
    method: Literal["forward", "viterbi"] = "forward",
) -> tuple[Array, Array, list | None]:
    """
    複数の観測系列に対して、複数のHMMモデルによるスコアリングを実行.

    Args:
        outputs (list[Sequence]): 観測系列のリスト
                                 outputs[i] = i 番目の観測系列（整数リスト）
        PI_all (Array): すべてのモデルの初期状態確率
                       shape: (k, l), PI_all[m] = モデル m の初期状態確率
        A_all (Array): すべてのモデルの状態遷移確率行列
                      shape: (k, l, l)
        B_all (Array): すべてのモデルの出力確率行列
                      shape: (k, l, n_symbols)
        method (Literal["forward", "viterbi"]): 使用するアルゴリズム
                                               デフォルト: "forward"

    Returns:
        tuple[Array, Array, list | None]:
            - 予測ラベル: shape (p,), dtype int, 各系列の予測モデル番号
            - スコア行列: shape (p, k), スコア[i, m] = 系列i のモデルm でのスコア
            - 最尤状態系列: (method == "viterbi" の場合のみ)
              list[list[list[int] | None]], best_paths[i][m] = 系列i のモデルm での最尤状態系列
              (method == "forward" の場合は None)
    """
    k = PI_all.shape[0]  # モデル数
    p = len(outputs)  # 系列数

    # スコア行列: scores[i, m] = 系列i のモデルm でのスコア
    scores = np.zeros((p, k))

    # Viterbi用: 最尤状態系列を格納（method == "forward" の場合は None）
    best_paths = [[None] * k for _ in range(p)] if method == "viterbi" else None

    # 各モデルのパラメータを事前に対数領域に変換（計算効率化）
    log_params = [
        convert_to_log_params(PI_all[m], A_all[m], B_all[m]) for m in range(k)
    ]

    # 各観測系列ごとに処理
    for seq_idx, O in enumerate(outputs):
        # 各HMMモデルごとに処理
        for m_idx in range(k):
            log_PI, log_A, log_B = log_params[m_idx]
            A = A_all[m_idx]

            if method == "forward":
                # Forward アルゴリズムで対数尤度を計算
                score, _ = _forward_log(log_PI, A, log_B, O)
            else:  # method == "viterbi"
                # Viterbi アルゴリズムで最尤確率と最尤状態系列を計算
                score, path, _ = _viterbi_log(log_PI, log_A, log_B, O)
                best_paths[seq_idx][m_idx] = path

            scores[seq_idx, m_idx] = score

    # 各系列について最高スコアを持つモデルを予測ラベルとして取得
    predictions = np.argmax(scores, axis=1)
    return predictions, scores, best_paths
