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


def viterbi_algorithm(
    O: np.ndarray, PI: np.ndarray, A: np.ndarray, B: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Viterbi アルゴリズムで最適パスの対数確率と最尤状態系列を計算する．

    Parameters
    ----------
    O  : (T,)       出力系列（整数インデックス）
    PI : (l, 1)     初期確率
    A  : (l, l)     状態遷移確率行列
    B  : (l, n)     出力確率行列

    Returns
    -------
    log_prob  : float        最適パスの対数確率 log P*(O | lambda)
    best_path : (T,) ndarray 最尤状態系列（各時刻の状態インデックス）
    """
    T = len(O)
    l = A.shape[0]
    log_A = np.log(A + 1e-300)

    # --- 初期化 ---
    log_delta = np.log(PI[:, 0] + 1e-300) + np.log(B[:, O[0]] + 1e-300)  # (l,)
    # psi[t, j] = 時刻tに状態jへ来たとき、直前の最適状態
    psi = np.zeros((T, l), dtype=int)

    # --- 漸化式 ---
    for t in range(1, T):
        log_trans = log_delta[:, np.newaxis] + log_A  # (l, l)
        psi[t] = log_trans.argmax(axis=0)             # 各jへの最適な直前状態を記録
        log_delta = log_trans.max(axis=0) + np.log(B[:, O[t]] + 1e-300)

    # --- バックトラッキングで最尤状態系列を復元 ---
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = log_delta.argmax()
    for t in range(T - 2, -1, -1):
        best_path[t] = psi[t + 1, best_path[t + 1]]

    return log_delta.max(), best_path


def predict_models(
    outputs: np.ndarray,
    PI: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    algorithm: str = "forward",
) -> tuple[np.ndarray, np.ndarray | None]:
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
    y_pred     : (p,) 各系列の推定モデルインデックス
    best_paths : (p, T) 最尤状態系列（viterbiのみ。forwardはNone）
    """
    p, T = outputs.shape
    k = PI.shape[0]

    scores = np.zeros((p, k))
    best_paths = None

    if algorithm == "forward":
        for i, O in enumerate(outputs):
            for j in range(k):
                scores[i, j] = forward_algorithm(O, PI[j], A[j], B[j])

    elif algorithm == "viterbi":
        # best_paths[i] = 推定モデルに対する最尤状態系列
        best_paths = np.zeros((p, T), dtype=int)
        for i, O in enumerate(outputs):
            for j in range(k):
                log_prob, path = viterbi_algorithm(O, PI[j], A[j], B[j])
                scores[i, j] = log_prob
            # スコア最大のモデルのパスだけ保存
            best_model = scores[i].argmax()
            _, best_paths[i] = viterbi_algorithm(O, PI[best_model], A[best_model], B[best_model])

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return scores.argmax(axis=1), best_paths


# ---------- 内部ユーティリティ ----------

def _log_sum_exp(log_x: np.ndarray) -> float:
    """log( sum(exp(log_x)) ) をオーバーフロー/アンダーフローなしに計算する"""
    c = log_x.max()
    return c + np.log(np.exp(log_x - c).sum())


def _log_sum_exp_axis0(log_x: np.ndarray) -> np.ndarray:
    """log-sum-exp を axis=0 方向に適用する  (l, l) -> (l,)"""
    c = log_x.max(axis=0)
    return c + np.log(np.exp(log_x - c).sum(axis=0))
