import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np


def load_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    """ヘッダー行つきのCSVファイルを読み込む。

    Args:
        path: 入力CSVファイルのパス。

    Returns:
        数値データ行列と列名リストのタプル。

    Raises:
        ValueError: CSVファイルにヘッダー行がない場合。
    """
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = list(data.dtype.names or [])
    if not names:
        raise ValueError("CSV header is required")
    columns = np.column_stack([data[name] for name in names])
    return columns, names


def choose_columns(
    names: list[str], features: list[str] | None, target: str | None
) -> tuple[list[str], str]:
    """CSVの列名から説明変数と目的変数の列を選ぶ。

    Args:
        names: CSVファイルに含まれる列名。
        features: ユーザーが指定した説明変数の列名。
        target: ユーザーが指定した目的変数の列名。

    Returns:
        説明変数の列名リストと目的変数の列名のタプル。

    Raises:
        ValueError: 指定された列が存在しない場合、または説明変数が残らない場合。
    """
    if target is None:
        target = "y" if "y" in names else names[-1]
    if target not in names:
        raise ValueError(f"target column {target!r} is not in {names}")

    if features:
        missing = [name for name in features if name not in names]
        if missing:
            raise ValueError(f"feature columns {missing} are not in {names}")
    else:
        features = [name for name in names if name != target]
    if not features:
        raise ValueError("at least one feature column is required")
    return features, target


def sigmoid(z: np.ndarray) -> np.ndarray:
    """シグモイド関数を計算する。

    Args:
        z: 入力配列。

    Returns:
        要素ごとのシグモイド関数の値。
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def regularization_value(w: np.ndarray, regularization: str, reg_lambda: float) -> float:
    """目的関数に加える正則化項を計算する。

    バイアス項は正則化の対象から除外する。

    Args:
        w: バイアス項を含む重み。
        regularization: 正則化の種類。``none``、``ridge``、``lasso`` のいずれか。
        reg_lambda: 正則化の強さ。

    Returns:
        損失に加える正則化項の値。
    """
    w_no_bias = w.copy()
    w_no_bias[0] = 0.0
    if regularization == "ridge":
        return 0.5 * reg_lambda * float(np.sum(w_no_bias**2))
    if regularization == "lasso":
        return reg_lambda * float(np.sum(np.abs(w_no_bias)))
    return 0.0


def regularization_grad(w: np.ndarray, regularization: str, reg_lambda: float) -> np.ndarray:
    """正則化項の勾配を計算する。

    バイアス項は正則化の対象から除外する。

    Args:
        w: バイアス項を含む重み。
        regularization: 正則化の種類。``none``、``ridge``、``lasso`` のいずれか。
        reg_lambda: 正則化の強さ。

    Returns:
        正則化項の勾配ベクトル。
    """
    grad = np.zeros_like(w)
    if regularization == "ridge":
        grad = reg_lambda * w
    elif regularization == "lasso":
        grad = reg_lambda * np.sign(w)
    grad[0] = 0.0
    return grad


def output_path(input_path: Path, regularization: str, reg_lambda: float) -> Path:
    """ロジスティック回帰の図を保存する出力パスを作る。

    Args:
        input_path: 入力CSVファイルのパス。
        regularization: 正則化の種類。
        reg_lambda: 正則化の強さ。

    Returns:
        図を保存する ``out/`` 以下のパス。
    """
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    suffix = regularization
    if regularization != "none":
        suffix += f"_lambda{str(reg_lambda).replace('-', 'm').replace('.', 'p')}"
    return out_dir / f"{input_path.stem}_logistic_{suffix}.png"


def fit_logistic(
    features: np.ndarray,
    y: np.ndarray,
    lr: float,
    iters: int,
    regularization: str,
    reg_lambda: float,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """勾配降下法によりロジスティック回帰を学習する。

    関数内では、バイアス項の列を追加した計画行列を ``X`` とし、
    正解の値を ``y``、重みを ``w`` と書く。

    Args:
        features: 元の説明変数行列。
        y: 2値の目的変数。
        lr: 学習率。
        iters: 勾配降下法の反復回数。
        regularization: 正則化の種類。``none``、``ridge``、``lasso`` のいずれか。
        reg_lambda: 正則化の強さ。

    Returns:
        学習した重み ``w`` と評価指標の履歴のタプル。
    """
    n = features.shape[0]
    X = np.column_stack([np.ones(n), features])
    w = np.zeros(X.shape[1])
    history = {"loss": [], "log_likelihood": [], "accuracy": []}
    eps = 1e-12

    for i in range(iters):
        prob = sigmoid(X @ w)
        log_likelihood = float(
            np.sum(y * np.log(prob + eps) + (1.0 - y) * np.log(1.0 - prob + eps))
        )
        loss = float(-log_likelihood / n + regularization_value(w, regularization, reg_lambda))
        pred_label = (prob >= 0.5).astype(int)
        accuracy = float(np.mean(pred_label == y))

        grad = X.T @ (prob - y) / n
        grad += regularization_grad(w, regularization, reg_lambda)
        w -= lr * grad

        history["loss"].append(loss)
        history["log_likelihood"].append(log_likelihood)
        history["accuracy"].append(accuracy)

        if i % max(1, iters // 10) == 0:
            print(f"iter {i}: loss={loss:.6f}, log_likelihood={log_likelihood:.3f}, acc={accuracy:.3f}")

    return w, history


def plot_history(history: dict[str, list[float]], path: Path) -> None:
    """ロジスティック回帰の学習曲線を描画する。

    Args:
        history: ``loss``、``log_likelihood``、``accuracy`` の履歴を持つ辞書。
        path: 出力画像のパス。
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(history["loss"], color="tab:red")
    axes[0].set_ylabel("損失")
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["log_likelihood"], color="tab:blue")
    axes[1].set_ylabel("対数尤度")
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["accuracy"], color="tab:green")
    axes[2].set_ylabel("正解率")
    axes[2].set_xlabel("反復回数")
    axes[2].grid(alpha=0.3)

    fig.suptitle("解答例: ロジスティック回帰")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    """ロジスティック回帰のコマンドラインインターフェースを実行する。"""
    parser = argparse.ArgumentParser(description="ロジスティック回帰の解答例")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--features", nargs="*")
    parser.add_argument("--target")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--regularization", choices=["none", "ridge", "lasso"], default="none")
    parser.add_argument("--lambda", dest="reg_lambda", type=float, default=0.0)
    args = parser.parse_args()

    data, names = load_csv(args.input)
    feature_names, target_name = choose_columns(names, args.features, args.target)
    name_to_idx = {name: i for i, name in enumerate(names)}
    x = data[:, [name_to_idx[name] for name in feature_names]]
    y = data[:, name_to_idx[target_name]]

    w, history = fit_logistic(
        x, y, args.lr, args.iters, args.regularization, args.reg_lambda
    )
    path = output_path(args.input, args.regularization, args.reg_lambda)
    plot_history(history, path)

    print("w:")
    print(f"  bias: {w[0]:.6f}")
    for name, coef in zip(feature_names, w[1:]):
        print(f"  {name}: {coef:.6f}")
    print(f"final loss: {history['loss'][-1]:.6f}")
    print(f"final log_likelihood: {history['log_likelihood'][-1]:.6f}")
    print(f"final accuracy: {history['accuracy'][-1]:.6f}")
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
