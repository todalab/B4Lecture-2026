"""
結果可視化モジュール

このモジュールは、HMM推論結果の混同行列を
カラーマップを用いて可視化。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from hmm.process_data import Array


def plot_confusion_matrix(
    cm: Array,
    title: str,
    ax: plt.Axes,
    cmap: str = "Blues",
) -> None:
    """
    混同行列をヒートマップで可視化。

    Args:
        cm (Array): 混同行列
                   shape: (k, k), cm[i, j] = 真ラベルi、予測ラベルj のサンプル数
        title (str): グラフのタイトル
        ax (plt.Axes): 描画対象の Matplotlib Axes オブジェクト
        cmap (str): カラーマップ名（デフォルト: "Blues"）

    Returns:
        None (plt.Axes オブジェクトが副作用で更新される)
    """
    k = cm.shape[0]  # クラス数

    # ヒートマップを描画
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # 軸ラベルを設定: モデル番号 m0, m1, ..., m_{k-1}
    labels = [f"m{i}" for i in range(k)]
    ax.set(
        xticks=np.arange(k),
        yticks=np.arange(k),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )

    # 各セルに値を数字で表示
    # セルの背景色に応じて、文字色を黒または白に切り替え
    thresh = cm.max() / 2.0
    for i in range(k):
        for j in range(k):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
