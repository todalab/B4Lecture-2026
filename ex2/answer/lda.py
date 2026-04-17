import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """ヘッダー付きCSVから特徴量とラベルを読み込む。

    Args:
        path: 入力CSVファイルのパス。

    Returns:
        特徴量行列、ラベルベクトル、列名リストのタプル。

    Raises:
        ValueError: 特徴量列とラベル列を含むだけの列数がない場合。
    """
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = list(data.dtype.names or [])
    if len(names) < 3:
        raise ValueError("feature columns and label column are required")
    columns = np.column_stack([data[name] for name in names])
    return columns[:, :-1], columns[:, -1].astype(int), names


def fit_lda(features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray | float]:
    """2クラスLDAを一般化固有値問題として解く。

    Args:
        features: 入力特徴量行列。
        labels: 各データ点のクラスラベル。

    Returns:
        クラス、クラス平均、クラス内分散行列、クラス間分散行列、
        固有値、LDA方向、射影値、分類しきい値、予測ラベル、
        accuracyを持つ辞書。

    Raises:
        ValueError: クラス数が2でない場合。
    """
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("this example supports exactly two classes")

    overall_mean = np.mean(features, axis=0)
    sw = np.zeros((features.shape[1], features.shape[1]))
    sb = np.zeros_like(sw)
    class_means = []
    for cls in classes:
        x_cls = features[labels == cls]
        mean = np.mean(x_cls, axis=0)
        centered = x_cls - mean
        sw += centered.T @ centered
        mean_diff = (mean - overall_mean).reshape(-1, 1)
        sb += x_cls.shape[0] * (mean_diff @ mean_diff.T)
        class_means.append(mean)

    matrix = np.linalg.pinv(sw) @ sb
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    order = np.argsort(eigenvalues)[::-1]
    w = eigenvectors[:, order[0]]
    w = w / np.linalg.norm(w)
    projected = features @ w

    class0_values = projected[labels == classes[0]]
    class1_values = projected[labels == classes[1]]
    threshold = float((np.mean(class0_values) + np.mean(class1_values)) / 2.0)
    if np.mean(class0_values) < np.mean(class1_values):
        pred = np.where(projected < threshold, classes[0], classes[1])
    else:
        pred = np.where(projected >= threshold, classes[0], classes[1])
    accuracy = float(np.mean(pred == labels))

    return {
        "classes": classes,
        "class_means": np.array(class_means),
        "sw": sw,
        "sb": sb,
        "eigenvalues": eigenvalues[order],
        "direction": w,
        "projected": projected,
        "threshold": threshold,
        "prediction": pred,
        "accuracy": accuracy,
    }


def output_path(input_path: Path, suffix: str) -> Path:
    """LDAの出力画像パスを作る。

    Args:
        input_path: 入力CSVファイルのパス。
        suffix: 出力ファイル名に付けるサフィックス。

    Returns:
        ``out/`` 以下に保存する画像ファイルのパス。
    """
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    return out_dir / f"{input_path.stem}_{suffix}.png"


def plot_axis(
    features: np.ndarray,
    labels: np.ndarray,
    result: dict[str, np.ndarray | float],
    path: Path,
) -> None:
    """元データの散布図にLDA軸を重ねる。

    Args:
        features: 入力特徴量行列。
        labels: 各データ点のクラスラベル。
        result: ``fit_lda`` が返すLDAの結果辞書。
        path: 保存先の画像ファイルパス。
    """
    classes = result["classes"]
    direction = result["direction"]
    center = np.mean(features, axis=0)
    length = float(np.max(np.std(features, axis=0)) * 3.0)
    start = center - direction * length
    end = center + direction * length

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, color in zip(classes, ["tab:blue", "tab:orange"]):
        x_cls = features[labels == cls]
        ax.scatter(x_cls[:, 0], x_cls[:, 1], s=28, alpha=0.75, label=f"class {cls}", color=color)
    ax.plot([start[0], end[0]], [start[1], end[1]], color="tab:red", linewidth=3, label="LDA軸")
    ax.set_title("解答例: LDA: 元データと射影軸")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_projection(
    labels: np.ndarray,
    result: dict[str, np.ndarray | float],
    path: Path,
) -> None:
    """LDA軸へ射影した1次元データを可視化する。

    Args:
        labels: 各データ点のクラスラベル。
        result: ``fit_lda`` が返すLDAの結果辞書。
        path: 保存先の画像ファイルパス。
    """
    classes = result["classes"]
    projected = result["projected"]
    threshold = float(result["threshold"])

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for y_pos, cls, color in zip([0.0, 0.18], classes, ["tab:blue", "tab:orange"]):
        values = projected[labels == cls]
        ax.scatter(values, np.full_like(values, y_pos), s=28, alpha=0.75, label=f"class {cls}", color=color)
    ax.axvline(threshold, color="tab:red", linestyle="--", label=f"threshold={threshold:.3f}")
    ax.set_title(f"解答例: LDA: 1次元射影 (accuracy={result['accuracy']:.3f})")
    ax.set_xlabel("LDA軸上の値")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    """LDA解答例のコマンドラインインターフェースを実行する。"""
    parser = argparse.ArgumentParser(description="LDAの解答例")
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()

    features, labels, names = load_csv(args.input)
    result = fit_lda(features, labels)

    print("class means:")
    for cls, mean in zip(result["classes"], result["class_means"]):
        print(f"  class {cls}: {mean}")
    print("LDA direction:")
    print(f"  {result['direction']}")
    print(f"threshold: {result['threshold']:.6f}")
    print(f"accuracy: {result['accuracy']:.6f}")

    axis_path = output_path(args.input, "axis")
    projection_path = output_path(args.input, "projection")
    plot_axis(features, labels, result, axis_path)
    plot_projection(labels, result, projection_path)
    print(f"saved: {axis_path}")
    print(f"saved: {projection_path}")


if __name__ == "__main__":
    main()
