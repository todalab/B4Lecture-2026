import argparse
import os
import time

import matplotlib.pyplot as plt
from hmm.eval import evaluate
from hmm.plot import plot_confusion_matrix
from hmm.process_data import get_hmm_params, load_dataset
from hmm.score import score_sequences


def main(file_name, method, show_details=False):
    if not os.path.exists("../data/" + file_name):
        print(f"File {file_name} not found.")
        return
    dataset_path = "../data/" + file_name
    data = load_dataset(dataset_path)
    outputs, PI_all, A_all, B_all, true_labels, k = get_hmm_params(data)

    start_time = time.perf_counter()

    pred_labels, scores, best_paths = score_sequences(
        outputs, PI_all, A_all, B_all, method=method
    )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    cm, acc = evaluate(true_labels, pred_labels, k)

    print(f"正解率 (Accuracy): {acc:.4f}")
    print(f"計算時間: {elapsed_time:.4f} 秒")
    print("混同行列 :")
    print(cm)

    if show_details:
        num_outputs = len(outputs)
        if method == "viterbi":
            print("\n=== 各系列の対数確率および最尤状態系列 (Viterbi) ===")
            for i in range(num_outputs):
                print(f"系列 {i}:")
                for m in range(k):
                    print(f" モデル m{m} のスコア: {scores[i, m]:.4f}")
                    print(f" モデル m{m} の最尤状態系列: {best_paths[i][m]}")
                print(f" 推定モデル: m{pred_labels[i]}")

        elif method == "forward":
            print("\n=== 各系列の対数尤度 (Forward) ===")
            for i in range(num_outputs):
                print(f"系列 {i}:")
                for m in range(k):
                    print(f" モデル m{m} の対数尤度: {scores[i, m]:.4f}")
                print(f" 推定モデル: m{pred_labels[i]}")
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_confusion_matrix(cm, title=f"Confusion Matrix ({method})", ax=ax)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM 推論スクリプト")
    parser.add_argument(
        "--file",
        type=str,
        default="data1.pickle",
        help="読み込むデータセットのファイルパスを指定",
    )
    parser.add_argument(
        "--show_details",
        action="store_true",
        help="すべての系列に対する詳細な結果を出力",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["forward", "viterbi"],
        default="forward",
        help="使用するアルゴリズムを指定 (forward または viterbi)",
    )
    args = parser.parse_args()
    main(args.file, method=args.method, show_details=args.show_details)
