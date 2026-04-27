"""
hmm.py - HMMの基本アルゴリズム実装

- forward_algorithm  : Forward アルゴリズムによる尤度計算
- viterbi_algorithm  : Viterbi アルゴリズムによる最適パス確率計算
- predict_model      : 尤度/対数確率が最大のモデルを返す
"""

import numpy as np


def forward_algorithm(O: np.ndarray, PI: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """
    Forward アルゴリズムで出力系列の尤度 P(O | lambda) を計算する．

    Parameters
    ----------
    O  : (T,)       出力系列（整数インデックス）
    PI : (l, 1)     初期確率
    A  : (l, l)     状態遷移確率行列
    B  : (l, n)     出力確率行列

    Returns
    -------
    float : P(O | lambda)  ※アンダーフロー対策のため対数スケールで計算
    """
    T = len(O)
    l = A.shape[0]

    # --- 初期化 ---
    # alpha[i] = pi[i] * B[i, O[0]]   shape: (l,)
    log_alpha = np.log(PI[:, 0] + 1e-300) + np.log(B[:, O[0]] + 1e-300)

    # --- 漸化式（対数スケール + log-sum-exp） ---
    for t in range(1, T):
        # log_alpha_next[j] = log( sum_i exp(log_alpha[i] + log(A[i,j])) ) + log(B[j, O[t]])
        # shape broadcast: log_alpha (l,1) + log(A) (l,l) -> (l,l)
        log_trans = log_alpha[:, np.newaxis] + np.log(A + 1e-300)  # (l, l)
        log_alpha = _log_sum_exp_axis0(log_trans) + np.log(B[:, O[t]] + 1e-300)

    # --- 終端 ---
    log_likelihood = _log_sum_exp(log_alpha)
    return log_likelihood


def viterbi_algorithm(O: np.ndarray, PI: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """
    Viterbi アルゴリズムで最適パスの対数確率を計算する．

    Parameters
    ----------
    O  : (T,)       出力系列（整数インデックス）
    PI : (l, 1)     初期確率
    A  : (l, l)     状態遷移確率行列
    B  : (l, n)     出力確率行列

    Returns
    -------
    float : 最適パスの対数確率 log P*(O | lambda)
    """
    T = len(O)

    # --- 初期化 ---
    log_delta = np.log(PI[:, 0] + 1e-300) + np.log(B[:, O[0]] + 1e-300)  # (l,)

    # --- 漸化式 ---
    for t in range(1, T):
        # log_delta_next[j] = max_i (log_delta[i] + log(A[i,j])) + log(B[j, O[t]])
        log_trans = log_delta[:, np.newaxis] + np.log(A + 1e-300)  # (l, l)
        log_delta = log_trans.max(axis=0) + np.log(B[:, O[t]] + 1e-300)

    return log_delta.max()


def predict_models(
    outputs: np.ndarray,
    PI: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    algorithm: str = "forward",
) -> np.ndarray:
    """
    各出力系列に対して，最も尤もらしいモデルを推定する．

    Parameters
    ----------
    outputs   : (p, T)   出力系列の集合
    PI        : (k, l, 1)
    A         : (k, l, l)
    B         : (k, l, n)
    algorithm : "forward" or "viterbi"

    Returns
    -------
    np.ndarray : (p,) 各系列の推定モデルインデックス
    """
    p, _ = outputs.shape
    k = PI.shape[0]

    if algorithm == "forward":
        score_fn = forward_algorithm
    elif algorithm == "viterbi":
        score_fn = viterbi_algorithm
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    scores = np.zeros((p, k))
    for i, O in enumerate(outputs):
        for j in range(k):
            scores[i, j] = score_fn(O, PI[j], A[j], B[j])

    return scores.argmax(axis=1)


# ---------- 内部ユーティリティ ----------

def _log_sum_exp(log_x: np.ndarray) -> float:
    """log( sum(exp(log_x)) ) をオーバーフロー/アンダーフローなしに計算する"""
    c = log_x.max()
    return c + np.log(np.exp(log_x - c).sum())


def _log_sum_exp_axis0(log_x: np.ndarray) -> np.ndarray:
    """log-sum-exp を axis=0 方向に適用する  (l, l) -> (l,)"""
    c = log_x.max(axis=0)
    return c + np.log(np.exp(log_x - c).sum(axis=0))
