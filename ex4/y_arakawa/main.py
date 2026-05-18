# -*- coding: utf-8 -*-
"""課題4 HMM Forwardアルゴリズム Viterbiアルゴリズム."""

import pickle
import time

import matplotlib.pyplot as plt
import numpy as np


def forward_algorithm(
    PI: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    output: np.ndarray,
) -> np.ndarray:
    """forwardアルゴリズム.複数モデルについて同時に計算する.

    Args:
        PI (np.ndarray): 初期確率(K, L, 1)
        A (np.ndarray): 状態遷移確率行列(K, L, L)
        B (np.ndarray): (K, L, N)
        output (np.ndarray): (T,)
            K: モデル数
            L: 状態数
            N: 出力記号数
            P: 出力系列数
            T: 系列長

    Returns:
        np.ndarray: 前向き確率(K, L, T)
    """
    L = PI.shape[1]
    T = output.shape[0]
    K = PI.shape[0]
    forward_prob = np.zeros((K, L, T))

    # 初期化
    for k in range(K):
        forward_prob[k, :, 0] = PI[k].reshape(L) * B[k, :, output[0]]

    # 帰納
    for k in range(K):
        for t in range(1, T):
            for l in range(L):
                forward_prob[k, l, t] = (
                    np.sum(np.dot(forward_prob[k, :, t - 1], A[k, :, l]))
                    * B[k, l, output[t]]
                )
    return forward_prob


def calculate_forward_likelihood(forward_prob: np.ndarray) -> np.ndarray:
    """前向き確率を用いたモデル全体での尤度の計算.

    Args:
        forward_prob (np.ndarray): 前向き確率(K, L, T)

    Returns:
        np.ndarray: 尤度(k,)
    """
    K, L, T = forward_prob.shape
    likelihood = np.zeros(K)
    for k in range(K):
        likelihood[k] = np.sum(forward_prob[k, :, T - 1])
    return likelihood


def run_forward_algorithm_against_all_output_series(
    PI: np.ndarray, A: np.ndarray, B: np.ndarray, output: np.ndarray
) -> tuple[np.ndarray, float]:
    """全ての出力系列に対してforwardアルゴリズムを実行する.実行時間を計測する.

    Args:
        PI (np.ndarray): 初期確率(K, L, 1)
        A (np.ndarray): 状態遷移確率行列(K, L, L)
        B (np.ndarray): (K, L, N)
        output (np.ndarray): (P, T)
            K: モデル数
            L: 状態数
            N: 出力記号数
            P: 出力系列数
            T: 系列長
    Returns:
        tuple[np.ndarray, float]: 前向き確率(K, L, T), 実行時間
    """
    start_time = time.time()
    K, L = PI.shape[0], PI.shape[1]
    P, T = output.shape
    forward_prob = np.zeros((K, L, T))
    likelihoods = np.zeros((K, P))
    for p in range(P):
        forward_prob = forward_algorithm(PI, A, B, output[p])
        likelihoods[:, p] = calculate_forward_likelihood(forward_prob)
    end_time = time.time()
    return likelihoods, end_time - start_time


def evaluate_model_selection_by_forward_algorithm(
    likelihoods: np.ndarray, answer_models: np.ndarray, time: float, data_number: int
) -> None:
    """前向き確率を用いたモデル選択の評価.各出力系列について、尤度が最大のモデルが正解モデルと一致しているかを評価し、混同行列と正解率を出力する.

    Args:
        likelihoods (np.ndarray): 尤度(K, P)
        answer_models (np.ndarray): 正解モデル(P,)
    """
    K, P = likelihoods.shape
    predicted_models = np.argmax(
        likelihoods, axis=0
    )  # axis=0で列方向で最大値のインデックスを取得
    confusion_matrix = np.zeros((K, K), dtype=int)
    for p in range(P):
        confusion_matrix[answer_models[p], predicted_models[p]] += 1
    print("Confusion Matrix:")
    print(confusion_matrix)

    accuracy = np.sum(predicted_models == answer_models) / P
    print(f"Accuracy: {accuracy:.2f}")

    # matplotlibで混同行列を可視化
    fig, ax = plt.subplots()
    im = ax.imshow(
        confusion_matrix, cmap="Blues"
    )  # imshowでデータを画像として、特にヒートマップのように表示
    ax.set_xlabel("Predicted Model")
    ax.set_ylabel("True Model")
    ax.set_title(
        f"data{data_number} Forward Confusion Matrix\nAcc: {accuracy:.3f}, Time={time:.4f}s"
    )
    fig.colorbar(im)
    for i in range(K):
        for j in range(K):
            ax.text(
                j,
                i,
                confusion_matrix[i, j],
                ha="center",
                va="center",
                color=(
                    "white"
                    if confusion_matrix[i, j] > confusion_matrix.max() / 2
                    else "black"
                ),
            )
    plt.savefig(f"output/data{data_number}_forward_confusion_matrix.png")


def viterbi_algorithm(
    PI: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """viterbiアルゴリズム.複数モデルについて同時に計算する.

    Args:
        PI (np.ndarray): 初期確率(K, L, 1)
        A (np.ndarray): 状態遷移確率行列(K, L, L)
        B (np.ndarray): (K, L, N)
        output (np.ndarray): (P,)
            K: モデル数
            L: 状態数
            N: 出力記号数
            P: 出力系列数
            T: 系列長

    Returns:
        tuple[np.ndarray, np.ndarray]: 最も尤もらしい状態系列(K, T), viterbiアルゴリズムの最終的な確率(K, L, T)
    """
    L = PI.shape[1]
    T = output.shape[0]
    K = PI.shape[0]
    viterbi_prob = np.zeros((K, L, T))
    backpointer = np.zeros((K, L, T), dtype=int)

    # 初期化
    for k in range(K):
        viterbi_prob[k, :, 0] = PI[k].reshape(L) * B[k, :, output[0]]

    # 帰納
    for k in range(K):
        for t in range(1, T):
            for l in range(L):
                prob = viterbi_prob[k, :, t - 1] * A[k, :, l] * B[k, l, output[t]]
                backpointer[k, l, t] = np.argmax(prob)
                viterbi_prob[k, l, t] = np.max(prob)

    # 最も尤もらしい状態系列の復元
    best_path = np.zeros((K, T), dtype=int)
    for k in range(K):
        best_path[k, T - 1] = np.argmax(viterbi_prob[k, :, T - 1])
        for t in range(T - 2, -1, -1):
            best_path[k, t] = backpointer[k, best_path[k, t + 1], t + 1]

    return best_path, viterbi_prob


def calculate_viterbi_log_likelihood(viterbi_prob: np.ndarray) -> np.ndarray:
    """viterbiアルゴリズムの最終的な確率を用いて、モデル全体での対数尤度の計算.

    Args:
        viterbi_prob (np.ndarray): viterbiアルゴリズムの最終的な確率(K, L, T)

    Returns:
        np.ndarray: 対数尤度(k,)
    """
    K, L, T = viterbi_prob.shape
    log_likelihood = np.zeros(K)
    for k in range(K):
        log_likelihood[k] = np.log(np.max(viterbi_prob[k, :, T - 1]))
    return log_likelihood


def run_viterbi_algorithm_against_all_output_series(
    PI: np.ndarray, A: np.ndarray, B: np.ndarray, output: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """全ての出力系列に対してviterbiアルゴリズムを実行する.実行時間を計測する.

    Args:
        PI (np.ndarray): 初期確率(K, L, 1)
        A (np.ndarray): 状態遷移確率行列(K, L, L)
        B (np.ndarray): (K, L, N)
        output (np.ndarray): (P, T)
            K: モデル数
            L: 状態数
            N: 出力記号数
            P: 出力系列数
            T: 系列長

    Returns:
        tuple[np.ndarray, float]: 最尤状態系列(K, P, T), 実行時間
    """
    start_time = time.time()
    K = PI.shape[0]
    P, T = output.shape
    best_paths = np.zeros((K, P, T), dtype=int)
    likelihoods = np.zeros((K, P))
    for p in range(P):
        best_path, viterbi_prob = viterbi_algorithm(PI, A, B, output[p])
        best_paths[:, p, :] = best_path
        likelihoods[:, p] = calculate_viterbi_log_likelihood(viterbi_prob)
    end_time = time.time()
    return best_paths, likelihoods, end_time - start_time


def evaluate_model_selection_by_viterbi_algorithm(
    best_paths: np.ndarray,
    likelihoods: np.ndarray,
    answer_models: np.ndarray,
    time: float,
    data_number: int,
) -> None:
    """ビタビアルゴリズムを用いたモデル選択の評価.各出力系列について、尤度が最大のモデルが正解モデルと一致しているかを評価し、混同行列と正解率を出力する.また、最尤状態系列をテキストファイルに出力する.

    Args:
        best_paths (np.ndarray): 最尤状態系列(K, P, T)
        likelihoods (np.ndarray): 対数尤度(K, P)
        answer_models (np.ndarray): 正解モデル(P,)
        time (float): 実行時間
        data_number (int): データ番号
    """
    K, P, T = best_paths.shape
    predicted_models = np.argmax(
        likelihoods, axis=0
    )  # axis=0で列方向で最大値のインデックスを取得
    confusion_matrix = np.zeros((K, K), dtype=int)
    for p in range(P):
        confusion_matrix[answer_models[p], predicted_models[p]] += 1
    print("Confusion Matrix:")
    print(confusion_matrix)

    accuracy = np.sum(predicted_models == answer_models) / P
    print(f"Accuracy: {accuracy:.2f}")

    # matplotlibで混同行列を可視化
    fig, ax = plt.subplots()
    im = ax.imshow(
        confusion_matrix, cmap="Blues"
    )  # imshowでデータを画像として、特にヒートマップのように表示
    ax.set_xlabel("Predicted Model")
    ax.set_ylabel("True Model")
    ax.set_title(
        f"data{data_number} Viterbi Confusion Matrix\nAcc: {accuracy:.3f}, Time={time:.4f}s"
    )
    fig.colorbar(im)
    for i in range(K):
        for j in range(K):
            ax.text(
                j,
                i,
                confusion_matrix[i, j],
                ha="center",
                va="center",
                color=(
                    "white"
                    if confusion_matrix[i, j] > confusion_matrix.max() / 2
                    else "black"
                ),
            )
    plt.savefig(f"output/data{data_number}_viterbi_confusion_matrix.png")

    # 最尤状態系列を出力
    with open(f"output/data{data_number}_viterbi_best_paths.txt", "w") as f:
        for p in range(P):
            f.write(f"Output Series {p}:\n")
            f.write(
                f"Model {predicted_models[p]}: {best_paths[predicted_models[p], p, :]}\n"
            )
            f.write("\n")


def main():
    """data1~data4に対してforwardアルゴリズムとviterbiアルゴリズムを実行し、モデル選択の評価を行う."""
    data1 = pickle.load(open("../data/data1.pickle", "rb"))
    data2 = pickle.load(open("../data/data2.pickle", "rb"))
    data3 = pickle.load(open("../data/data3.pickle", "rb"))
    data4 = pickle.load(open("../data/data4.pickle", "rb"))

    all_data = [data1, data2, data3, data4]
    for i, data in enumerate(all_data, start=1):
        likelihoods, time = run_forward_algorithm_against_all_output_series(
            data["models"]["PI"],
            data["models"]["A"],
            data["models"]["B"],
            data["output"],
        )
        evaluate_model_selection_by_forward_algorithm(
            likelihoods, data["answer_models"], time, data_number=i
        )

        best_paths, likelihoods, time = run_viterbi_algorithm_against_all_output_series(
            data["models"]["PI"],
            data["models"]["A"],
            data["models"]["B"],
            data["output"],
        )
        evaluate_model_selection_by_viterbi_algorithm(
            best_paths, likelihoods, data["answer_models"], time, data_number=i
        )


if __name__ == "__main__":
    main()
