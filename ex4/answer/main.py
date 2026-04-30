"""
main.py - HMM課題 メインスクリプト

実行例:
    uv run python main.py --data data/data1.pickle
    uv run python main.py --data data/data1.pickle data/data2.pickle data/data3.pickle data/data4.pickle
"""

import argparse
import pickle
import time

import numpy as np
from evaluate import accuracy, confusion_matrix, plot_results
from hmm import predict_models


def run_experiment(data_path: str, out_dir: str) -> None:
    """1つのデータセットに対してForward・Viterbiを実行し結果を表示・保存する"""
    data_name = data_path.split("/")[-1].replace(".pickle", "")
    print(f"\n{'=' * 50}")
    print(f"Dataset: {data_path}")
    print(f"{'=' * 50}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    outputs: np.ndarray = data["output"]  # (p, T)
    answer: np.ndarray = data["answer_models"]  # (p,)
    PI: np.ndarray = data["models"]["PI"]  # (k, l, 1)
    A: np.ndarray = data["models"]["A"]  # (k, l, l)
    B: np.ndarray = data["models"]["B"]  # (k, l, n)

    k = PI.shape[0]
    p, T = outputs.shape
    print(f"  #models={k}, #sequences={p}, seq_len={T}")

    results = []

    for algo in ("forward", "viterbi"):
        t_start = time.perf_counter()
        y_pred, best_paths = predict_models(outputs, PI, A, B, algorithm=algo)
        elapsed = time.perf_counter() - t_start

        cm = confusion_matrix(answer, y_pred, n_classes=k)
        acc = accuracy(answer, y_pred)

        print(f"\n  [{algo.upper()}]")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Time     : {elapsed:.4f} s")
        print(f"    Confusion Matrix:\n{cm}")

        # Viterbi のときだけ最尤状態系列を先頭3件表示
        if algo == "viterbi" and best_paths is not None:
            print(f"    最尤状態系列（先頭3件）:")
            for idx in range(min(3, p)):
                print(
                    f"      系列{idx} (推定モデル=m{y_pred[idx]}, 正解=m{answer[idx]}): {best_paths[idx]}"
                )

        results.append(
            {"algo": algo.capitalize(), "cm": cm, "acc": acc, "elapsed": elapsed}
        )

    plot_results(results, data_name=data_name, out_dir=out_dir)


def main():
    parser = argparse.ArgumentParser(description="HMM モデル推定")
    parser.add_argument(
        "--data",
        nargs="+",
        default=["data/data1.pickle"],
        help="入力pickleファイルのパス（複数指定可）",
    )
    parser.add_argument(
        "--out",
        default="fig",
        help="結果画像の保存先ディレクトリ（デフォルト: fig）",
    )
    args = parser.parse_args()

    for data_path in args.data:
        run_experiment(data_path, out_dir=args.out)


if __name__ == "__main__":
    main()
