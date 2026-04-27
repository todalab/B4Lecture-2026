"""
evaluate.py - 混同行列・正解率の計算と可視化ユーティリティ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """
    混同行列を計算する．

    Parameters
    ----------
    y_true    : (p,) 正解ラベル
    y_pred    : (p,) 予測ラベル
    n_classes : クラス数

    Returns
    -------
    np.ndarray : (n_classes, n_classes) 混同行列
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """正解率を計算する"""
    return (y_true == y_pred).mean()


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    ax: plt.Axes = None,
) -> plt.Axes:
    """混同行列をヒートマップとして描画する"""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    n = cm.shape[0]
    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels([f"m{i}" for i in range(n)])
    ax.set_yticks(tick_marks)
    ax.set_yticklabels([f"m{i}" for i in range(n)])

    # セルに数値を表示
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    ax.set_ylabel("True model")
    ax.set_xlabel("Predicted model")
    ax.set_title(title)
    return ax


def plot_results(results: list[dict], data_name: str, out_dir: str = "out") -> None:
    """
    ForwardとViterbiの結果をまとめて描画して保存する．

    Parameters
    ----------
    results   : [{"algo": str, "cm": ndarray, "acc": float, "elapsed": float}, ...]
    data_name : データセット名（例 "data1"）
    out_dir   : 画像の保存先ディレクトリ
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    n_algo = len(results)
    fig, axes = plt.subplots(1, n_algo, figsize=(5 * n_algo, 4.5))
    if n_algo == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        title = (
            f"{res['algo']}\n"
            f"Acc={res['acc']:.3f}, Time={res['elapsed']:.4f}s"
        )
        plot_confusion_matrix(res["cm"], title=title, ax=ax)

    fig.suptitle(f"HMM Model Estimation — {data_name}", fontsize=13, y=1.02)
    fig.tight_layout()

    save_path = os.path.join(out_dir, f"{data_name}_result.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
