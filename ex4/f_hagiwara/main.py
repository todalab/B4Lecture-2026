"""The answer of Ex04 by Hagiwara Futa."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.special import logsumexp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import time

# from scipy.stats import multivariate_normal

plt.rcParams["font.family"] = "Noto Sans CJK JP"

DATA1_PATH = Path("../data/data1.pickle")
DATA2_PATH = Path("../data/data2.pickle")
DATA3_PATH = Path("../data/data3.pickle")
DATA4_PATH = Path("../data/data4.pickle")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="outputs",
        help="Path to output result directory.",
    )
    parser.add_argument(
        "--input_paths",
        type=Path,
        nargs="*",
        default=[DATA1_PATH, DATA2_PATH, DATA3_PATH, DATA4_PATH],
        help="Path to input data.",
    )

    return parser.parse_args()


def forward(datapath, output_dir):
    """Run codes about Forward algorithm."""
    data = pickle.load(open(datapath, "rb"))

    PI = data["models"]["PI"]
    A = data["models"]["A"]
    B = data["models"]["B"]
    outputs = data["output"]
    answer_models = data["answer_models"]
    pred_models = []

    k, l, _ = PI.shape
    _, _, n = B.shape
    p, t = outputs.shape
    print(f"k={k},l={l},n={n},p={p},t={t}")

    # for o in outputs:
    #     alpha = np.zeros((k, t, l))

    #     # 初期化
    #     alpha[:, 0, :] = PI[:, :, 0] * B[:, :, o[0]]

    #     # 帰納
    #     # 各モデル各時刻ごとに順に計算を行う
    #     for times in range(1, t):
    #         for model in range(k):
    #             # 同一時刻における各状態に対する演算を一気に行う
    #             alpha[model, times] = (alpha[model, times - 1] @ A[model]) * B[
    #                 model, :, o[times]
    #             ]

    #     # 尤度
    #     likelihood = np.sum(alpha[:, t - 1, :], axis=1)

    #     pred_models.append(np.argmax(likelihood))

    # cm = confusion_matrix(answer_models, pred_models)

    start = time.time()
    logA = np.log(A)
    logB = np.log(B)
    logPI = np.log(PI)

    for o in outputs:
        log_alpha = np.zeros((k, t, l))

        # 初期化
        log_alpha[:, 0, :] = logPI[:, :, 0] + logB[:, :, o[0]]

        # 帰納
        # 各モデル各時刻ごとに順に計算を行う
        for times in range(1, t):
            # 同一時刻における各状態に対する演算を一気に行う
            log_sum = (
                log_alpha[:, times - 1, :, np.newaxis] + logA
            )  # newaxisにより(l,) → (l, 1) にして broadcast
            log_alpha[:, times] = logsumexp(log_sum, axis=1) + logB[:, :, o[times]]

        # 対数尤度
        log_likelihood = logsumexp(log_alpha[:, t - 1, :], axis=1)

        pred_models.append(np.argmax(log_likelihood))

    end = time.time()

    acc = accuracy_score(answer_models, pred_models)
    cm = confusion_matrix(answer_models, pred_models)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[f"m{i}" for i in range(k)]
    )
    disp.plot(cmap="Blues")
    plt.xlabel("Predicted models")
    plt.ylabel("Answer models")

    # データファイル名をタイトルに使う
    title = f"{datapath.stem}:混合行列(Forward,Acc={acc:.4f})"
    plt.title(title)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{datapath.stem}_cmfor.png")
    plt.close()

    return end - start


def viterbi(datapath, output_dir):
    """Run codes about Viterbi algorithm."""
    data = pickle.load(open(datapath, "rb"))

    PI = data["models"]["PI"]
    A = data["models"]["A"]
    B = data["models"]["B"]
    outputs = data["output"]
    answer_models = data["answer_models"]
    logP = []
    pred_models = []
    pred_routes = []

    k, l, _ = PI.shape
    _, _, n = B.shape
    p, t = outputs.shape
    print(f"k={k},l={l},n={n},p={p},t={t}")

    start = time.time()

    # 演算
    logA = np.log(A)
    logB = np.log(B)
    logPI = np.log(PI)

    for o in outputs:
        delta_list = np.zeros((k, t, l))
        psi_list = np.zeros((k, t, l))
        delta_list[:, 0, :] = logPI[:, :, 0] + logB[:, :, o[0]]
        for times in range(1, t):
            temp_list = (
                delta_list[:, times - 1, :, np.newaxis] + logA
            )  # newaxisにより(l,) → (l, 1) にして broadcast
            psi_list[:, times, :] = np.argmax(temp_list, axis=1)
            delta_list[:, times, :] = np.max(temp_list, axis=1) + logB[:, :, o[times]]

        # 対数尤度
        log_likelihood = np.max(delta_list[:, t - 1, :], axis=1)
        logP.append(log_likelihood)

        pred_model = np.argmax(log_likelihood)
        pred_models.append(pred_model)

        route = np.zeros(t, dtype=int)

        route[-1] = np.argmax(delta_list[pred_model, t - 1])

        for i in reversed(range(t - 1)):
            route[i] = psi_list[pred_model, i + 1, route[i + 1]]

        pred_routes.append(route)

    end = time.time()

    print(f"{datapath.stem}:最尤状態系列（先頭5出力）")
    for i in range(5):
        print(pred_routes[i])

    acc = accuracy_score(answer_models, pred_models)
    cm = confusion_matrix(answer_models, pred_models)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[f"m{i}" for i in range(k)]
    )
    disp.plot(cmap="Blues")
    plt.xlabel("Predicted models")
    plt.ylabel("Answer models")
    # データファイル名をタイトルに使う
    title = f"{datapath.stem}:混合行列(Viterbi,Acc={acc:.4f})"
    plt.title(title)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{datapath.stem}_cmvit.png")
    plt.close()

    # 推定した最尤状態系列の可視化
    for route in pred_routes:
        plt.plot(route, alpha=0.7)
    plt.xlabel("時間")
    plt.xticks(np.arange(0, t + 1, 5))
    plt.ylabel("状態")
    plt.yticks(np.arange(0, l, 1))
    plt.title(f"{datapath.stem}:経路の可視化")
    plt.savefig(output_dir / f"{datapath.stem}_routes.png")
    plt.close()

    return end - start


def main():
    """Run main function."""
    args = parse_args()
    times_list_for = []
    times_list_vit = []

    for input_path in args.input_paths:
        times_list_for.append(forward(input_path, args.output_dir))
        times_list_vit.append(viterbi(input_path, args.output_dir))

    # 計算時間の出力
    labels = [p.stem for p in args.input_paths]
    fig, ax = plt.subplots()
    plt.bar(labels, times_list_for)
    plt.xlabel("データセット")
    plt.ylabel("経過時間(s)")
    plt.title("Forward Algorithmの経過時間")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    ax.bar_label(ax.containers[0], padding=1, size=8, fmt="%.3f")
    plt.savefig(args.output_dir / "time_list_for.png")
    plt.close()

    labels = [p.stem for p in args.input_paths]
    fig, ax = plt.subplots()
    plt.bar(labels, times_list_vit)
    plt.xlabel("データセット")
    plt.ylabel("経過時間(s)")
    plt.title("Viterbi Algorithmの経過時間")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    ax.bar_label(ax.containers[0], padding=1, size=8, fmt="%.3f")
    plt.savefig(args.output_dir / "time_list_vit.png")
    plt.close()

    # 計算時間の比較
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    bars1 = ax.bar(
        x - width / 2, times_list_for, width, label="Forward", color="steelblue"
    )
    bars2 = ax.bar(
        x + width / 2, times_list_vit, width, label="Viterbi", color="skyblue"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.xlabel("データセット")
    plt.ylabel("経過時間(s)")
    plt.title("経過時間比較")
    ax.legend()
    ax.bar_label(bars1, padding=1, size=8, fmt="%.3f")
    ax.bar_label(bars2, padding=1, size=8, fmt="%.3f")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(args.output_dir / "time_compare.png")
    plt.close()


if __name__ == "__main__":
    main()
