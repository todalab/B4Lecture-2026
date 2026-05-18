"""
HMM推論スクリプト

このモジュールは、複数のHMM（隠れマルコフモデル）を用いて、観測系列を最も尤度の高いモデルに分類する.
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
from hmm.eval import evaluate
from hmm.plot import plot_confusion_matrix
from hmm.process_data import get_hmm_params, load_dataset
from hmm.score import score_sequences


def run_inference(file_name, method):
    """
    単一のデータセットに対してHMM推論を実行し、結果を返す.

    Args:
        file_name (str): 読み込むデータセットのファイル名
        method (str): 使用するアルゴリズム ("forward" または "viterbi")

    Returns:
        dict | None: 推論結果を含む辞書.以下のキーを持つ:
            - "acc": 分類精度（0.0～1.0）
            - "time": 処理時間（秒）
            - "cm": 混同行列（numpy array）
            - "pred": 予測ラベル配列
            - "scores": 各モデルのスコア行列

            ファイルが存在しない場合は None を返します.
    """
    dataset_path = "../data/" + file_name
    if not os.path.exists(dataset_path):
        return None

    # データセットの読み込みとHMMパラメータの抽出
    data = load_dataset(dataset_path)
    outputs, PI_all, A_all, B_all, true_labels, k = get_hmm_params(data)

    # 指定されたアルゴリズムで各系列のスコアを計算（時間計測）
    start_time = time.perf_counter()
    pred_labels, scores, best_paths = score_sequences(
        outputs, PI_all, A_all, B_all, method=method
    )
    elapsed_time = time.perf_counter() - start_time

    # 推論結果の評価（混同行列と精度を計算）
    cm, acc = evaluate(true_labels, pred_labels, k)
    return {
        "acc": acc,
        "time": elapsed_time,
        "cm": cm,
        "pred": pred_labels,
        "scores": scores,
    }


def main(file_name, method, show_details=False):
    """
    HMM推論のメイン処理を実行.

    3つのモードに対応:
    1. "compare" モード: 複数データセット間でForwardとViterbiの精度・処理時間を比較
    2. "forward" モード: Forward アルゴリズムで推論
    3. "viterbi" モード: Viterbi アルゴリズムで推論

    Args:
        file_name (str): 読み込むデータセットのファイル名
                         (compare モードでは無視される)
        method (str): 推論方法 ("forward", "viterbi", "compare")
        show_details (bool): 詳細結果の出力有無.
                             True の場合、全系列の詳細情報を表示.
                             デフォルト: False

    Returns:
        None (結果はコンソール出力と混同行列プロットで表示)
    """
    if method == "compare":
        # 複数データセット間での性能比較モード
        print("=== 6-3 性能比較（data1〜data4） ===")
        results = {}
        for f in ["data1.pickle", "data2.pickle", "data3.pickle", "data4.pickle"]:
            # Forward と Viterbi の両方を実行
            fwd = run_inference(f, "forward")
            vtb = run_inference(f, "viterbi")

            if fwd and vtb:
                results[f] = {
                    "fwd_acc": fwd["acc"],
                    "vtb_acc": vtb["acc"],
                    "fwd_time": fwd["time"],
                    "vtb_time": vtb["time"],
                }
                # 精度と処理時間の比較結果を出力
                print(
                    f"[{f}] Forward: {fwd['acc']:.4f} ({fwd['time']:.4f}秒) | Viterbi: {vtb['acc']:.4f} ({vtb['time']:.4f}秒)"
                )
        return
    if not os.path.exists("../data/" + file_name):
        print(f"File {file_name} not found.")
        return None

    dataset_path = "../data/" + file_name
    # データセットの読み込みとHMMパラメータの抽出
    data = load_dataset(dataset_path)
    outputs, PI_all, A_all, B_all, true_labels, k = get_hmm_params(data)

    # 指定されたアルゴリズムで推論実行（時間計測）
    start_time = time.perf_counter()

    pred_labels, scores, best_paths = score_sequences(
        outputs, PI_all, A_all, B_all, method=method
    )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # 推論結果の評価
    cm, acc = evaluate(true_labels, pred_labels, k)

    # 主要な結果を出力
    print(f"正解率 (Accuracy): {acc:.4f}")
    print(f"計算時間: {elapsed_time:.4f} 秒")
    print("混同行列 :")
    print(cm)

    if show_details:
        # 詳細な推論結果を系列ごとに出力
        num_outputs = len(outputs)
        if method == "viterbi":
            print("\n=== 各系列の対数確率および最尤状態系列 (Viterbi) ===")
            for i in range(num_outputs):
                print(f"系列 {i}:")
                # 各モデルの対数確率と最尤状態系列を出力
                for m in range(k):
                    print(f" モデル m{m} のスコア: {scores[i, m]:.4f}")
                    print(f" モデル m{m} の最尤状態系列: {best_paths[i][m]}")
                print(f" 推定モデル: m{pred_labels[i]}")

        elif method == "forward":
            print("\n=== 各系列の対数尤度 (Forward) ===")
            for i in range(num_outputs):
                print(f"系列 {i}:")
                # 各モデルの対数尤度を出力
                for m in range(k):
                    print(f" モデル m{m} の対数尤度: {scores[i, m]:.4f}")
                print(f" 推定モデル: m{pred_labels[i]}")

    # 混同行列を可視化
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_confusion_matrix(
        cm, title=f"Confusion Matrix ({method}) -- {file_name}", ax=ax
    )
    plt.show()


if __name__ == "__main__":
    # コマンドライン引数のパース
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
        choices=["forward", "viterbi", "compare"],
        default="forward",
        help="使用するアルゴリズムを指定 (forward または viterbi)",
    )
    args = parser.parse_args()
    # メイン処理を実行
    main(args.file, method=args.method, show_details=args.show_details)
