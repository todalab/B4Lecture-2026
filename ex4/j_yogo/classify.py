"""
HMM モデルトポロジー分類・分析スクリプト.

このモジュールは、複数のHMMモデルをそのトポロジー（Left-to-Right または Ergodic）で分類し、
各グループの推論性能や最尤状態系列パターンを分析.
"""

import os

import numpy as np
from hmm.process_data import get_hmm_params, load_dataset
from hmm.score import score_sequences


def is_left_to_right(A, eps=1e-5):
    """
    遷移行列AがLeft-to-Right構造かどうかを判定.

    Left-to-Rightモデルでは、状態は一方向にのみ遷移するため、遷移行列の下三角成分はほぼ0になる.

    Args:
        A (np.ndarray): 状態遷移確率行列
                       A[i, j] = 状態iから状態jへの遷移確率
        eps (float): 下三角成分の閾値.この値以下なら0とみなす.
                    デフォルト: 1e-5

    Returns:
        bool: Left-to-Right構造なら True、そうでなければ False
    """
    return np.all(np.tril(A, k=-1) <= eps)


def analyze_dataset(file_name):
    """
    単一のデータセットを分析し、モデルのトポロジーごとに推論性能を評価します.

    処理内容:
    1. データセットを読み込み、HMMパラメータを抽出
    2. 各モデルをLeft-to-Rightまたはergodicに分類
    3. Forward と Viterbi アルゴリズムで推論
    4. 各トポロジーグループの精度と最尤状態系列を出力

    Args:
        file_name (str): 分析対象のデータセットファイル名
                        (相対パス: ../data/{file_name})

    Returns:
        None (結果はコンソール出力)
    """
    # データセットの読み込みとHMMパラメータの抽出
    dataset_path = "../data/" + file_name
    if not os.path.exists(dataset_path):
        return

    # データセットの読み込みとHMMパラメータの抽出
    data = load_dataset(dataset_path)
    outputs, PI_all, A_all, B_all, true_labels, k = get_hmm_params(data)

    # モデルのトポロジー分類: Left-to-Right vs Ergodic
    ltr_models = []
    erg_models = []
    for m in range(k):
        if is_left_to_right(A_all[m]):
            ltr_models.append(m)
        else:
            erg_models.append(m)

    print("\n=========================================")
    print(f"データセット: {file_name}")
    print(f"Left-to-Right: {len(ltr_models)}個, Ergodic: {len(erg_models)}個")

    # Forward と Viterbi の両方で推論を実行
    for method in ["forward", "viterbi"]:
        pred_labels, scores, best_paths = score_sequences(
            outputs, PI_all, A_all, B_all, method=method
        )

        print(f"\n--- アルゴリズム: {method.upper()} ---")

        # ===== Left-to-Right モデル由来のデータの評価 =====
        if ltr_models:
            # Left-to-Rightモデルから生成されたデータのマスク
            ltr_mask = np.isin(true_labels, ltr_models)
            if np.sum(ltr_mask) > 0:
                # Left-to-Rightデータのみで正解率を計算
                ltr_acc = np.sum(
                    pred_labels[ltr_mask] == true_labels[ltr_mask]
                ) / np.sum(ltr_mask)
                print(
                    f"  Left-to-Right データの正解率: {ltr_acc:.4f} (対象: {np.sum(ltr_mask)}件)"
                )

                if method == "viterbi":
                    # Viterbiのみ、最尤状態系列パターンを表示
                    print("  Left-to-Right データの最尤状態系列パターン（先頭3件）:")
                    indices = np.where(ltr_mask)[0][:3]
                    for idx in indices:
                        pred_m = pred_labels[idx]
                        true_m = true_labels[idx]
                        print(
                            f"    系列 {idx} (真: m{true_m}, 推定: m{pred_m}): {best_paths[idx][pred_m]}"
                        )

        # ===== Ergodic モデル由来のデータの評価 =====
        if erg_models:
            # Ergodicモデルから生成されたデータのマスク
            erg_mask = np.isin(true_labels, erg_models)
            if np.sum(erg_mask) > 0:
                # Ergodicデータのみで正解率を計算
                erg_acc = np.sum(
                    pred_labels[erg_mask] == true_labels[erg_mask]
                ) / np.sum(erg_mask)
                print(
                    f"  Ergodic       データの正解率: {erg_acc:.4f} (対象: {np.sum(erg_mask)}件)"
                )

                if method == "viterbi":
                    # Viterbiのみ、最尤状態系列パターンを表示
                    print("  Ergodic       データの最尤状態系列パターン（先頭3件）:")
                    indices = np.where(erg_mask)[0][:3]
                    for idx in indices:
                        pred_m = pred_labels[idx]
                        true_m = true_labels[idx]
                        print(
                            f"    系列 {idx} (真: m{true_m}, 推定: m{pred_m}): {best_paths[idx][pred_m]}"
                        )


def main():
    """
    4つのすべてのデータセット（data1～data4）に対してanalyze_dataset を実行し、総合的な分析結果を出力.
    """
    target_files = ["data1.pickle", "data2.pickle", "data3.pickle", "data4.pickle"]
    for f in target_files:
        analyze_dataset(f)


if __name__ == "__main__":
    # メイン処理を実行
    main()
