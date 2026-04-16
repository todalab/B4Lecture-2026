import argparse
from itertools import product
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
        if "z" in names:
            target = "z"
        elif "y" in names:
            target = "y"
        else:
            target = names[-1]
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


def total_degree_terms(n_features: int, degree: int) -> list[tuple[int, ...]]:
    """指定した総次数以下の多項式項を列挙する。

    Args:
        n_features: 入力特徴量の数。
        degree: 多項式の最大総次数。

    Returns:
        総次数順に並べた指数タプルのリスト。
    """
    terms: list[tuple[int, ...]] = []
    for powers in product(range(degree + 1), repeat=n_features):
        if sum(powers) <= degree:
            terms.append(powers)
    return sorted(terms, key=lambda powers: (sum(powers), powers))


def design_matrix(
    features: np.ndarray, degree: int
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """多項式回帰の計画行列を作る。

    計画行列とは、各データ点についてモデルで使う特徴量
    （例: ``1``, ``x``, ``x^2``）を1行に並べた行列である。

    Args:
        features: 元の説明変数行列。
        degree: 多項式の最大総次数。

    Returns:
        計画行列 ``X`` と、各列に対応する指数タプルのリスト。

    Raises:
        ValueError: ``degree`` が負の場合。
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    terms = total_degree_terms(features.shape[1], degree)
    cols = []
    for powers in terms:
        col = np.ones(features.shape[0])
        for i, power in enumerate(powers):
            if power:
                col = col * features[:, i] ** power
        cols.append(col)
    return np.column_stack(cols), terms


def fit_closed_form(
    X: np.ndarray, y: np.ndarray, regularization: str, reg_lambda: float
) -> np.ndarray:
    """閉形式解により重み ``w`` を推定する。

    Args:
        X: 計画行列（各データ点の特徴量を行として並べた行列）。
        y: 正解の値を並べたベクトル。
        regularization: 正則化の種類。対応する値は ``none`` と ``ridge``。
        reg_lambda: 正則化の強さ。

    Returns:
        推定された重み ``w``。

    Raises:
        ValueError: 閉形式解で対応していない正則化が指定された場合。
    """
    if regularization == "none":
        return np.linalg.lstsq(X, y, rcond=None)[0]
    if regularization != "ridge":
        raise ValueError("closed form is available only for none or ridge")

    penalty = np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    a = X.T @ X + reg_lambda * penalty
    b = X.T @ y
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


def soft_threshold(value: float, threshold: float) -> float:
    """Lassoで使うソフトしきい値処理を適用する。

    Args:
        value: 入力スカラー。
        threshold: しきい値。

    Returns:
        しきい値処理後のスカラー。
    """
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def fit_lasso(
    X: np.ndarray, y: np.ndarray, reg_lambda: float, iters: int
) -> np.ndarray:
    """座標降下法によりLasso回帰を行う。

    バイアス項以外の列は最適化中に標準化し、返す前に元のスケールへ戻す。

    Args:
        X: 計画行列（各データ点の特徴量を行として並べた行列）。
        y: 正解の値を並べたベクトル。
        reg_lambda: L1正則化の強さ。
        iters: 座標降下法の反復回数。

    Returns:
        元の特徴量スケールに戻した重み ``w``。
    """
    means = np.zeros(X.shape[1])
    scales = np.ones(X.shape[1])
    means[1:] = np.mean(X[:, 1:], axis=0)
    scales[1:] = np.std(X[:, 1:], axis=0)
    scales[scales == 0.0] = 1.0

    X_scaled = X.copy()
    X_scaled[:, 1:] = (X[:, 1:] - means[1:]) / scales[1:]

    w_scaled = np.zeros(X.shape[1])
    n = X.shape[0]
    col_norms = np.sum(X_scaled**2, axis=0) / n

    for _ in range(iters):
        for j in range(X_scaled.shape[1]):
            residual = y - X_scaled @ w_scaled + X_scaled[:, j] * w_scaled[j]
            rho = float(X_scaled[:, j] @ residual / n)
            if j == 0:
                w_scaled[j] = rho / col_norms[j]
            else:
                w_scaled[j] = soft_threshold(rho, reg_lambda) / col_norms[j]

    w = w_scaled / scales
    w[0] = w_scaled[0] - float(np.sum(w_scaled[1:] * means[1:] / scales[1:]))
    return w


def predict_from_terms(
    features: np.ndarray, terms: list[tuple[int, ...]], w: np.ndarray
) -> np.ndarray:
    """入力サンプルに対して多項式モデルの予測値を計算する。

    Args:
        features: 元の説明変数行列。
        terms: 各多項式項に対応する指数タプル。
        w: ``terms`` に対応する重み。

    Returns:
        各サンプルの予測値。
    """
    pred = np.zeros(features.shape[0])
    for coef, powers in zip(w, terms):
        col = np.ones(features.shape[0])
        for i, power in enumerate(powers):
            if power:
                col = col * features[:, i] ** power
        pred += coef * col
    return pred


def metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """回帰モデルの評価指標を計算する。

    Args:
        y: 正解の値を並べたベクトル。
        pred: 目的変数の予測値。

    Returns:
        平均二乗誤差と決定係数のタプル。
    """
    mse = float(np.mean((y - pred) ** 2))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mse, r2


def fit_model(
    features: np.ndarray,
    y: np.ndarray,
    degree: int,
    regularization: str,
    reg_lambda: float,
    iters: int,
) -> tuple[np.ndarray, list[tuple[int, ...]], tuple[float, float]]:
    """多項式回帰モデルを学習する。

    Args:
        features: 元の説明変数行列。
        y: 正解の値を並べたベクトル。
        degree: 多項式の最大総次数。
        regularization: 正則化の種類。``none``、``ridge``、``lasso`` のいずれか。
        reg_lambda: 正則化の強さ。
        iters: Lassoで使う座標降下法の反復回数。

    Returns:
        重み ``w``、多項式項、評価指標のタプル。
    """
    X, terms = design_matrix(features, degree)
    if regularization == "lasso":
        w = fit_lasso(X, y, reg_lambda, iters)
    else:
        w = fit_closed_form(X, y, regularization, reg_lambda)
    pred = X @ w
    return w, terms, metrics(y, pred)


def standardize_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """説明変数を平均0、標準偏差1に標準化する。

    Args:
        x: 元の説明変数行列。

    Returns:
        標準化後の説明変数行列、各列の平均、各列の標準偏差のタプル。
    """
    means = np.mean(x, axis=0)
    scales = np.std(x, axis=0)
    scales[scales == 0.0] = 1.0
    return (x - means) / scales, means, scales


def safe_float(value: float) -> str:
    """浮動小数点数をファイル名に使いやすい文字列へ変換する。

    Args:
        value: 変換する浮動小数点数。

    Returns:
        符号と小数点を置換した文字列。
    """
    return str(value).replace("-", "m").replace(".", "p")


def output_path(
    input_path: Path,
    degree: int,
    regularization: str,
    reg_lambda: float,
    standardized: bool,
) -> Path:
    """回帰結果の図を保存する出力パスを作る。

    Args:
        input_path: 入力CSVファイルのパス。
        degree: モデルに使った多項式の次数。
        regularization: 正則化の種類。
        reg_lambda: 正則化の強さ。
        standardized: 説明変数を標準化したかどうか。

    Returns:
        図を保存する ``out/`` 以下のパス。
    """
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    suffix = f"degree{degree}_{regularization}"
    if regularization != "none":
        suffix += f"_lambda{safe_float(reg_lambda)}"
    if standardized:
        suffix += "_standardized"
    return out_dir / f"{input_path.stem}_regression_{suffix}.png"


def term_label(term: tuple[int, ...], feature_names: list[str]) -> str:
    """多項式項を表示用の文字列に整形する。

    Args:
        term: 多項式項の指数タプル。
        feature_names: 指数に対応する特徴量名。

    Returns:
        整形後のラベル。
    """
    parts = []
    for name, power in zip(feature_names, term):
        if power == 1:
            parts.append(name)
        elif power > 1:
            parts.append(f"{name}^{power}")
    return "1" if not parts else "*".join(parts)


def plot_1d(
    x: np.ndarray,
    y: np.ndarray,
    feature_name: str,
    target_name: str,
    selected: tuple[int, np.ndarray, list[tuple[int, ...]]],
    transform: tuple[np.ndarray, np.ndarray] | None,
    path: Path,
) -> None:
    """1次元データとフィッティング曲線を描画する。

    Args:
        x: 1列の説明変数行列。
        y: 目的変数の値。
        feature_name: 説明変数の列名。
        target_name: 目的変数の列名。
        selected: 採用したモデルの次数、重み、項。
        transform: 予測時に使う標準化用の平均と標準偏差。標準化しない場合は ``None``。
        path: 出力画像のパス。
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(x[:, 0], y, s=24, alpha=0.75, label="データ")
    x_line = np.linspace(float(x[:, 0].min()), float(x[:, 0].max()), 300).reshape(-1, 1)

    degree, w, terms = selected
    x_eval = (x_line - transform[0]) / transform[1] if transform else x_line
    pred = predict_from_terms(x_eval, terms, w)
    regression_label = "回帰直線" if degree == 1 else "回帰曲線"
    plt.plot(x_line[:, 0], pred, color="red", linewidth=2.0, label=regression_label)

    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.title(f"解答例: 多項式回帰（{degree}次）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_2d_surface(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    target_name: str,
    degree: int,
    w: np.ndarray,
    terms: list[tuple[int, ...]],
    transform: tuple[np.ndarray, np.ndarray] | None,
    path: Path,
) -> None:
    """2次元データとフィッティング曲面を描画する。

    Args:
        x: 2列の説明変数行列。
        y: 目的変数の値。
        feature_names: 2つの説明変数の列名。
        target_name: 目的変数の列名。
        degree: フィッティング曲面に使った多項式の次数。
        w: 重み。
        terms: ``w`` に対応する多項式項。
        transform: 予測時に使う標準化用の平均と標準偏差。標準化しない場合は ``None``。
        path: 出力画像のパス。
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], y, s=18, alpha=0.75, label="データ")

    grid_x, grid_y = np.meshgrid(
        np.linspace(float(x[:, 0].min()), float(x[:, 0].max()), 40),
        np.linspace(float(x[:, 1].min()), float(x[:, 1].max()), 40),
    )
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_eval = (grid - transform[0]) / transform[1] if transform else grid
    grid_z = predict_from_terms(grid_eval, terms, w).reshape(grid_x.shape)
    ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.45, color="red")

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(target_name)
    ax.set_title(f"解答例: 多項式曲面回帰（{degree}次）")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    """多項式回帰のコマンドラインインターフェースを実行する。"""
    parser = argparse.ArgumentParser(description="多項式回帰の解答例")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--features", nargs="*")
    parser.add_argument("--target")
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument(
        "--regularization", choices=["none", "ridge", "lasso"], default="none"
    )
    parser.add_argument("--lambda", dest="reg_lambda", type=float, default=0.0)
    parser.add_argument(
        "--iters", type=int, default=1000, help="coordinate descent sweeps for lasso"
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="多項式項を作る前に説明変数を標準化する",
    )
    args = parser.parse_args()

    data, names = load_csv(args.input)
    feature_names, target_name = choose_columns(names, args.features, args.target)
    name_to_idx = {name: i for i, name in enumerate(names)}
    x = data[:, [name_to_idx[name] for name in feature_names]]
    y = data[:, name_to_idx[target_name]]
    if args.standardize:
        x_model, means, scales = standardize_features(x)
        transform = (means, scales)
    else:
        x_model = x
        transform = None

    w, terms, (mse, r2) = fit_model(
        x_model, y, args.degree, args.regularization, args.reg_lambda, args.iters
    )

    print(f"input: {args.input}")
    print(f"features: {feature_names}")
    print(f"target: {target_name}")
    print(f"degree: {args.degree}")
    print(f"regularization: {args.regularization}, lambda={args.reg_lambda}")
    print(f"standardize: {args.standardize}")
    print(f"MSE: {mse:.6f}")
    print(f"R2: {r2:.6f}")
    print("w:")
    for term, coef in zip(terms, w):
        print(f"  {term_label(term, feature_names):>12s}: {coef:.6f}")

    path = output_path(
        args.input, args.degree, args.regularization, args.reg_lambda, args.standardize
    )
    if x.shape[1] == 1:
        plot_1d(
            x,
            y,
            feature_names[0],
            target_name,
            (args.degree, w, terms),
            transform,
            path,
        )
    elif x.shape[1] == 2:
        plot_2d_surface(
            x, y, feature_names, target_name, args.degree, w, terms, transform, path
        )
    else:
        raise ValueError("visualization supports one or two feature columns")
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
