import os

import numpy as np
from hmm.process_data import get_hmm_params, load_dataset
from hmm.score import score_sequences


def is_left_to_right(A, eps=1e-5):
    """A行列の下三角成分（対角成分を除く）がすべてeps以下であるか判定します"""
    return np.all(np.tril(A, k=-1) <= eps)


def analyze_dataset(file_name):
    dataset_path = "../data/" + file_name
    if not os.path.exists(dataset_path):
        return

    data = load_dataset(dataset_path)
    outputs, PI_all, A_all, B_all, true_labels, k = get_hmm_params(data)

    # モデルのトポロジー分類
    ltr_models = []
    erg_models = []
    for m in range(k):
        if is_left_to_right(A_all[m]):
            ltr_models.append(m)
        else:
            erg_models.append(m)

    print("\n=========================================")
    print(f"データセット: {file_name}")
    print(
        f"モデル内訳 - Left-to-Right: {len(ltr_models)}個, Ergodic: {len(erg_models)}個"
    )

    # 各アルゴリズムで推論と集計を実行
    for method in ["forward", "viterbi"]:
        pred_labels, scores, best_paths = score_sequences(
            outputs, PI_all, A_all, B_all, method=method
        )

        print(f"\n--- アルゴリズム: {method.upper()} ---")

        # Left-to-Rightモデル由来のデータの評価
        if ltr_models:
            ltr_mask = np.isin(true_labels, ltr_models)
            if np.sum(ltr_mask) > 0:
                ltr_acc = np.sum(
                    pred_labels[ltr_mask] == true_labels[ltr_mask]
                ) / np.sum(ltr_mask)
                print(
                    f"  Left-to-Right データの正解率: {ltr_acc:.4f} (対象: {np.sum(ltr_mask)}件)"
                )

                if method == "viterbi":
                    print("  Left-to-Right データの最尤状態系列パターン（先頭3件）:")
                    indices = np.where(ltr_mask)[0][:3]
                    for idx in indices:
                        pred_m = pred_labels[idx]
                        true_m = true_labels[idx]
                        print(
                            f"    系列 {idx} (真: m{true_m}, 推定: m{pred_m}): {best_paths[idx][pred_m]}"
                        )

        # Ergodicモデル由来のデータの評価
        if erg_models:
            erg_mask = np.isin(true_labels, erg_models)
            if np.sum(erg_mask) > 0:
                erg_acc = np.sum(
                    pred_labels[erg_mask] == true_labels[erg_mask]
                ) / np.sum(erg_mask)
                print(
                    f"  Ergodic       データの正解率: {erg_acc:.4f} (対象: {np.sum(erg_mask)}件)"
                )

                if method == "viterbi":
                    print("  Ergodic       データの最尤状態系列パターン（先頭3件）:")
                    indices = np.where(erg_mask)[0][:3]
                    for idx in indices:
                        pred_m = pred_labels[idx]
                        true_m = true_labels[idx]
                        print(
                            f"    系列 {idx} (真: m{true_m}, 推定: m{pred_m}): {best_paths[idx][pred_m]}"
                        )


def main():
    target_files = ["data1.pickle", "data2.pickle", "data3.pickle", "data4.pickle"]
    for f in target_files:
        analyze_dataset(f)


if __name__ == "__main__":
    main()
