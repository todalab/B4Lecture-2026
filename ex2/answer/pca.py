import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np


def load_csv(path: Path) -> np.ndarray:
    """ヘッダーなしCSVを2次元配列として読み込む。

    Args:
        path: 入力CSVファイルのパス。

    Returns:
        CSVから読み込んだデータ行列。1列だけの場合も2次元配列として返す。
    """
    data = np.genfromtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def normalize(data: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCAの前処理として平均中心化または標準化を行う。

    Args:
        data: 入力データ行列。
        method: 前処理方法。``center`` の場合は平均中心化のみ、
            ``standardize`` の場合は標準化を行う。

    Returns:
        前処理後のデータ行列、各列の平均、各列のスケールのタプル。
    """
    mean = np.mean(data, axis=0)
    scale = np.ones(data.shape[1])
    if method == "standardize":
        scale = np.std(data, axis=0, ddof=1)
        scale[scale == 0.0] = 1.0
    return (data - mean) / scale, mean, scale


def fit_pca(data: np.ndarray, method: str) -> dict[str, np.ndarray]:
    """共分散行列の固有値分解によりPCAを行う。

    Args:
        data: 入力データ行列。
        method: PCA前の処理方法。``center`` または ``standardize``。

    Returns:
        前処理済みデータ、平均、スケール、共分散行列、固有値、
        固有ベクトル、主成分スコア、寄与率、累積寄与率を持つ辞書。
    """
    z, mean, scale = normalize(data, method)
    covariance = z.T @ z / (z.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    scores = z @ eigenvectors
    explained_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_ratio = np.cumsum(explained_ratio)
    return {
        "normalized": z,
        "mean": mean,
        "scale": scale,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "scores": scores,
        "explained_ratio": explained_ratio,
        "cumulative_ratio": cumulative_ratio,
    }


def output_path(input_path: Path, suffix: str) -> Path:
    """PCAの出力画像パスを作る。

    Args:
        input_path: 入力CSVファイルのパス。
        suffix: 出力ファイル名に付けるサフィックス。

    Returns:
        ``out/`` 以下に保存する画像ファイルのパス。
    """
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    return out_dir / f"{input_path.stem}_{suffix}.png"


def plot_pca_2d(data: np.ndarray, result: dict[str, np.ndarray], path: Path) -> None:
    """2次元データの散布図に全主成分軸を重ねる。

    Args:
        data: 元の2次元データ行列。
        result: ``fit_pca`` が返すPCAの結果辞書。
        path: 保存先の画像ファイルパス。
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(data[:, 0], data[:, 1], s=28, alpha=0.75)

    base_scale = float(np.sqrt(result["eigenvalues"][0])) * 2.5
    colors = ["tab:red", "tab:green"]
    for i in range(2):
        direction = result["eigenvectors"][:, i] * result["scale"]
        direction = direction / np.linalg.norm(direction)
        length = base_scale * np.sqrt(result["eigenvalues"][i] / result["eigenvalues"][0])
        start = result["mean"] - direction * length
        end = result["mean"] + direction * length
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=colors[i],
            linewidth=3,
            label=f"PC{i + 1} ({result['explained_ratio'][i] * 100:.1f}%)",
        )

    ax.set_title("解答例: PCA: 2次元データと主成分軸")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_pca_3d(data: np.ndarray, result: dict[str, np.ndarray], path: Path) -> None:
    """3次元データの散布図に全主成分軸を重ねる。

    Args:
        data: 元の3次元データ行列。
        result: ``fit_pca`` が返すPCAの結果辞書。
        path: 保存先の画像ファイルパス。
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=28, alpha=0.75)

    base_scale = float(np.sqrt(result["eigenvalues"][0])) * 2.5
    colors = ["tab:red", "tab:green", "tab:purple"]
    for i in range(3):
        direction = result["eigenvectors"][:, i] * result["scale"]
        direction = direction / np.linalg.norm(direction)
        length = base_scale * np.sqrt(result["eigenvalues"][i] / result["eigenvalues"][0])
        start = result["mean"] - direction * length
        end = result["mean"] + direction * length
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=colors[i],
            linewidth=3,
            label=f"PC{i + 1} ({result['explained_ratio'][i] * 100:.1f}%)",
        )

    ax.set_title("解答例: PCA: 3次元データと主成分軸")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.view_init(elev=24, azim=-58)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_scores_2d(result: dict[str, np.ndarray], path: Path, title: str) -> None:
    """PCAスコアの第1・第2主成分を散布図として描く。

    Args:
        result: ``fit_pca`` が返すPCAの結果辞書。
        path: 保存先の画像ファイルパス。
        title: 図のタイトル。
    """
    scores = result["scores"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(scores[:, 0], scores[:, 1], s=28, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"解答例: {title}")
    ax.set_xlabel(f"PC1 ({result['explained_ratio'][0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({result['explained_ratio'][1] * 100:.1f}%)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_cumulative_ratio(result: dict[str, np.ndarray], path: Path) -> int:
    """累積寄与率を描き、90%以上となる最小次元数を返す。

    Args:
        result: ``fit_pca`` が返すPCAの結果辞書。
        path: 保存先の画像ファイルパス。

    Returns:
        累積寄与率が90%以上となる最小の主成分数。
    """
    cumulative = result["cumulative_ratio"]
    dims = np.arange(1, len(cumulative) + 1)
    min_dim = int(np.searchsorted(cumulative, 0.9) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, cumulative, marker="o", markersize=3)
    ax.axhline(0.9, color="tab:red", linestyle="--", label="90%")
    ax.axvline(min_dim, color="tab:green", linestyle="--", label=f"{min_dim}次元")
    ax.set_title("解答例: PCA: 累積寄与率")
    ax.set_xlabel("主成分の数")
    ax.set_ylabel("累積寄与率")
    ax.set_ylim(0.0, 1.03)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return min_dim


def main() -> None:
    """PCA解答例のコマンドラインインターフェースを実行する。"""
    parser = argparse.ArgumentParser(description="PCAの解答例")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--normalize",
        choices=["center", "standardize"],
        default="center",
        help="PCA前の処理方法",
    )
    args = parser.parse_args()

    data = load_csv(args.input)
    result = fit_pca(data, args.normalize)

    print("eigenvalues:")
    for i, value in enumerate(result["eigenvalues"], start=1):
        print(f"  PC{i}: {value:.6f}")
    print("explained ratio:")
    for i, (ratio, cumulative) in enumerate(
        zip(result["explained_ratio"], result["cumulative_ratio"]), start=1
    ):
        print(f"  PC{i}: {ratio:.6f}, cumulative={cumulative:.6f}")

    if data.shape[1] == 2:
        path = output_path(args.input, "axes")
        plot_pca_2d(data, result, path)
        print(f"saved: {path}")

    if data.shape[1] == 3:
        path = output_path(args.input, "axes")
        plot_pca_3d(data, result, path)
        print(f"saved: {path}")

    if data.shape[1] >= 3:
        path = output_path(args.input, "scores_2d")
        plot_scores_2d(result, path, f"PCA: {args.input.name} の2次元圧縮")
        print(f"saved: {path}")

    if data.shape[1] >= 10:
        path = output_path(args.input, "cumulative_ratio")
        min_dim = plot_cumulative_ratio(result, path)
        print(f"90%以上の累積寄与率に必要な最小次元数: {min_dim}")
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
