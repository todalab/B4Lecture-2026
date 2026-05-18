import argparse
import time

import matplotlib.pyplot as plt
from hmm_eval import evaluate
from hmm_plot import plot_confusion_matrix
from hmm_score import score_sequences
from process_data import get_hmm_params, load_dataset


def main(dataset_path, method):
    data = load_dataset(dataset_path)
    outputs, PI_all, A_all, B_all, true_labels, k = get_hmm_params(data)

    start_time = time.perf_counter()

    pred_labels, scores, _ = score_sequences(
        outputs, PI_all, A_all, B_all, method=method
    )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    cm, acc = evaluate(true_labels, pred_labels, k)

    print(f"正解率 (Accuracy): {acc:.4f}")
    print(f"計算時間: {elapsed_time:.4f} 秒")
    print("混同行列 :")
    print(cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_confusion_matrix(cm, title=f"Confusion Matrix ({method.capitalize()})", ax=ax)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM 推論スクリプト")
    parser.add_argument(
        "--file",
        type=str,
        default="../data/data1.pickle",
        help="読み込むデータセットのファイルパスを指定",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["forward", "viterbi"],
        default="forward",
        help="推論に使用するアルゴリズムを指定 ",
    )
    args = parser.parse_args()

    main(args.file, args.method)
