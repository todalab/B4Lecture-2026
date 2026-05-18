"""
HMMデータ処理・パラメータ抽出モジュール.

このモジュールは、Pickleファイルからのデータセット読み込みや、
HMMパラメータの抽出、および対数領域への変換を行う.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

# 型別名
Array = np.ndarray
Sequence = list[int]

# アンダーフロー防止用の極めて小さい値
EPS = 1e-300


def convert_to_log_params(
    PI: Array,
    A: Array,
    B: Array,
) -> tuple[Array, Array, Array]:
    """
    HMMパラメータを通常領域から対数領域に変換.

    対数領域での計算により、確率の掛け算でのアンダーフローを防ぐ.

    注意:
    - 確率値が 0 になる場合、EPS を加えることで log(0) エラーを回避
    - この処理により数値的な安定性が向上

    Args:
        PI (Array): 初期状態確率分布
                   shape: (l,), Σ_i PI[i] = 1.0
        A (Array): 状態遷移確率行列
                  shape: (l, l), Σ_j A[i, j] = 1.0 for all i
        B (Array): 出力確率行列
                  shape: (l, n_symbols), Σ_o B[i, o] = 1.0 for all i

    Returns:
        tuple[Array, Array, Array]: (log_PI, log_A, log_B)
            - log_PI: shape (l,), log_PI[i] = log(PI[i])
            - log_A: shape (l, l), log_A[i, j] = log(A[i, j])
            - log_B: shape (l, n_symbols), log_B[i, o] = log(B[i, o])
    """
    log_PI = np.log(PI[:, 0] + EPS)
    log_A = np.log(A + EPS)
    log_B = np.log(B + EPS)
    return log_PI, log_A, log_B


def load_dataset(path: str | Path) -> dict:
    """
    Pickleファイルからデータセットを読み込む.

    Args:
        path (str | Path): Pickleファイルのパス

    Returns:
        dict: 読み込んだデータセット辞書
              期待されるキー:
              - "models": HMMモデルパラメータ
                - "PI": 初期状態確率 (shape: (k, l))
                - "A": 状態遷移確率 (shape: (k, l, l))
                - "B": 出力確率行列 (shape: (k, l, n_symbols))
              - "output": 観測系列のリスト (list[Sequence])
              - "answer_models": 正解モデルラベル (list[int])
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def get_hmm_params(data: dict) -> tuple:
    """
    データセット辞書からHMMパラメータを抽出.

    Args:
        data (dict): load_dataset で読み込んだデータセット辞書

    Returns:
        tuple: (outputs, PI_all, A_all, B_all, true_labels, k)
            - outputs: 観測系列のリスト (list[Sequence])
            - PI_all: すべてのモデルの初期状態確率 (shape: (k, l))
            - A_all: すべてのモデルの状態遷移確率 (shape: (k, l, l))
            - B_all: すべてのモデルの出力確率行列 (shape: (k, l, n_symbols))
            - true_labels: 正解モデルラベル (np.ndarray, shape: (p,))
            - k: モデル数 (int)
    """
    # データセットから各種パラメータを抽出
    models = data["models"]
    PI_all = models["PI"]  # すべてのモデルの初期状態確率
    A_all = models["A"]  # すべてのモデルの状態遷移確率
    B_all = models["B"]  # すべてのモデルの出力確率行列
    outputs = data["output"]  # 観測系列のリスト
    true_labels = np.array(data["answer_models"])  # 正解モデルラベル
    k = PI_all.shape[0]  # モデル数

    return outputs, PI_all, A_all, B_all, true_labels, k
