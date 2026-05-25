"""
HMM推論結果の評価モジュール.

このモジュールは、推論結果と正解ラベルを比較して、
混同行列の計算と分類精度の算出を行う.
"""

from __future__ import annotations

import numpy as np
from hmm.process_data import Array


def confusion_matrix(true_labels: Array, pred_labels: Array, k: int) -> Array:
    """
    混同行列を計算します.

    真のラベルと予測ラベルから、k×k の混同行列を構築.
    cm[i, j] = 真ラベルが i で予測ラベルが j のサンプル数

    Args:
        true_labels (Array): 真のラベル配列 (shape: (n,))
        pred_labels (Array): 予測ラベル配列 (shape: (n,))
        k (int): クラス数（モデル数）

    Returns:
        Array: 混同行列 (shape: (k, k), dtype: int)
    """
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(cm: Array) -> float:
    """
    混同行列から分類精度を計算.

    精度 = (正解数) / (総サンプル数)
         = trace(cm) / sum(cm)

    Args:
        cm (Array): 混同行列 (shape: (k, k))

    Returns:
        float: 分類精度（0.0～1.0）
    """
    cm_sum = cm.sum()
    return float(np.trace(cm) / cm_sum) if cm_sum != 0 else 0.0


def evaluate(
    true_labels: Array,
    pred_labels: Array,
    k: int,
) -> tuple[Array, float]:
    """
    推論結果を総合的に評価.

    真のラベルと予測ラベルから混同行列と精度を計算.

    Args:
        true_labels (Array): 真のラベル配列 (shape: (n,))
        pred_labels (Array): 予測ラベル配列 (shape: (n,))
        k (int): クラス数（モデル数）

    Returns:
        tuple[Array, float]: (混同行列, 精度) のタプル
            - 混同行列: shape (k, k), dtype int
            - 精度: float (0.0～1.0)
    """
    cm = confusion_matrix(true_labels, pred_labels, k)
    return cm, accuracy(cm)
