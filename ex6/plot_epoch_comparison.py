"""エポック別評価結果を可視化するスクリプト

`evaluate.py --compare_epochs` の出力を CSV / テキストファイルから読み込み、
Val Loss / Perplexity / ChrF のエポック推移をグラフとして保存する。

使用例:
    # 同梱のサンプルデータ (large モデル) をプロット
    uv run plot_epoch_comparison.py

    # 自前で用意した CSV から描画
    uv run plot_epoch_comparison.py --csv my_epochs.csv --model_size medium

    # `evaluate.py --compare_epochs` のテキスト出力をそのまま渡す
    uv run plot_epoch_comparison.py --txt epoch_results.txt
"""

import argparse
import os
import re
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

# evaluate.py --model_size large --compare_epochs の結果 (組み込みデフォルト)
DEFAULT_DATA_LARGE: List[Tuple[Optional[int], float, float, float]] = [
    (5, 5.8392, 343.50, 0.55),
    (10, 5.0070, 149.46, 2.48),
    (15, 4.3046, 74.04, 4.76),
    (20, 3.7458, 42.34, 7.72),
    (25, 3.3731, 29.17, 8.55),
    (30, 3.0755, 21.66, 9.26),
    (35, 2.8490, 17.27, 9.81),
    (40, 2.6808, 14.60, 10.53),
    (45, 2.5349, 12.62, 10.79),
    (50, 2.4258, 11.31, 10.71),
    (55, 2.3364, 10.34, 11.31),
    (60, 2.2751, 9.73, 12.19),
    (65, 2.2319, 9.32, 12.03),
    (70, 2.1839, 8.88, 12.36),
    (75, 2.1783, 8.83, 12.86),
    (80, 2.1584, 8.66, 13.58),
    (85, 2.1567, 8.64, 13.87),
    (90, 2.1644, 8.71, 13.93),
    (95, 2.1954, 8.98, 14.10),
    (100, 2.2253, 9.26, 15.08),
    (105, 2.2924, 9.90, 14.79),
    (110, 2.3056, 10.03, 14.87),
    (115, 2.3754, 10.76, 14.95),
    (120, 2.4245, 11.30, 14.64),
    (125, 2.4976, 12.15, 15.54),
    (130, 2.5825, 13.23, 15.29),
    (135, 2.6410, 14.03, 14.99),
    (140, 2.7165, 15.13, 15.01),
    (145, 2.7790, 16.10, 14.70),
    (150, 2.8578, 17.42, 14.27),
    (155, 2.9192, 18.53, 15.00),
    (160, 2.9885, 19.86, 15.18),
    (165, 3.0615, 21.36, 15.13),
    (170, 3.1119, 22.46, 15.28),
    (175, 3.1645, 23.68, 14.60),
    (180, 3.2252, 25.16, 14.76),
    (185, 3.2739, 26.41, 15.12),
    (190, 3.3293, 27.92, 15.07),
    (195, 3.3701, 29.08, 15.37),
    (200, 3.4069, 30.17, 15.11),
    (205, 3.4360, 31.06, 15.27),
    (210, 3.4920, 32.85, 14.70),
    (215, 3.5087, 33.41, 15.68),
    (220, 3.5445, 34.62, 15.44),
    (225, 3.5793, 35.85, 15.29),
    (230, 3.6047, 36.77, 15.19),
    (235, 3.6238, 37.48, 15.02),
    (240, 3.6396, 38.07, 14.99),
    (245, 3.6746, 39.43, 15.50),
    (250, 3.6829, 39.76, 15.66),
    (255, 3.7003, 40.46, 15.40),
    (260, 3.7050, 40.65, 15.12),
    (265, 3.7191, 41.23, 15.37),
    (270, 3.7250, 41.47, 15.31),
    (275, 3.7297, 41.67, 15.34),
    (280, 3.7317, 41.75, 15.49),
    (285, 3.7337, 41.84, 15.36),
    (290, 3.7346, 41.87, 15.52),
    (295, 3.7353, 41.90, 15.43),
    (300, 3.7357, 41.92, 15.47),
    (None, 2.1494, 8.58, 13.95),  # best
]


# ---------------------------------------------------------------------------
# 入力ファイルのパース
# ---------------------------------------------------------------------------
def parse_csv(path: str) -> List[Tuple[Optional[int], float, float, float]]:
    """ヘッダ付き CSV (epoch,val_loss,perplexity,chrf) を読む。
    `epoch` 列が "best" or 空欄なら None として扱う。
    """
    rows: List[Tuple[Optional[int], float, float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return rows
    # ヘッダを判定
    first = lines[0].lower()
    start = 1 if ("epoch" in first and "loss" in first) else 0
    for line in lines[start:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        epoch_str, loss_str, ppl_str, chrf_str = parts[:4]
        epoch: Optional[int] = (
            None if epoch_str.lower() in ("", "best") else int(epoch_str)
        )
        rows.append((epoch, float(loss_str), float(ppl_str), float(chrf_str)))
    return rows


def parse_txt(path: str) -> List[Tuple[Optional[int], float, float, float]]:
    """`evaluate.py --compare_epochs` のテキスト出力をパースする"""
    # 例: "  5            5.8392       343.50     0.55"
    #     "  best         2.1494         8.58    13.95"
    pat = re.compile(
        r"^\s*(\d+|best)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
        re.IGNORECASE,
    )
    rows: List[Tuple[Optional[int], float, float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            epoch_str, loss_str, ppl_str, chrf_str = m.groups()
            epoch = None if epoch_str.lower() == "best" else int(epoch_str)
            rows.append((epoch, float(loss_str), float(ppl_str), float(chrf_str)))
    return rows


# ---------------------------------------------------------------------------
# プロット
# ---------------------------------------------------------------------------
def plot_metrics(
    rows: List[Tuple[Optional[int], float, float, float]],
    model_size: str,
    out_path: str,
    log_ppl: bool = True,
) -> None:
    """3 つの指標を縦並びサブプロットで描画して保存する"""
    epoch_rows = [r for r in rows if r[0] is not None]
    best_rows = [r for r in rows if r[0] is None]
    if not epoch_rows:
        raise ValueError("epoch 付きのデータが 1 行もありません")

    epochs = [r[0] for r in epoch_rows]
    losses = [r[1] for r in epoch_rows]
    ppls = [r[2] for r in epoch_rows]
    chrfs = [r[3] for r in epoch_rows]

    # 最良エポックを (実エポックの中から) 算出
    best_loss_idx = min(range(len(losses)), key=lambda i: losses[i])
    best_chrf_idx = max(range(len(chrfs)), key=lambda i: chrfs[i])

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(
        f"Epoch comparison ({model_size})",
        fontsize=14,
        fontweight="bold",
    )

    # ---- Val Loss -----------------------------------------------------
    ax = axes[0]
    ax.plot(epochs, losses, marker="o", color="tab:blue", label="val loss")
    ax.axvline(
        epochs[best_loss_idx],
        color="tab:blue",
        linestyle=":",
        alpha=0.5,
        label=f"min loss @ ep{epochs[best_loss_idx]} ({losses[best_loss_idx]:.3f})",
    )
    if best_rows:
        ax.axhline(
            best_rows[0][1],
            color="black",
            linestyle="--",
            alpha=0.5,
            label=f"best.pt loss = {best_rows[0][1]:.3f}",
        )
    ax.set_ylabel("Val Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # ---- Perplexity ---------------------------------------------------
    ax = axes[1]
    ax.plot(epochs, ppls, marker="o", color="tab:orange", label="perplexity")
    if log_ppl:
        ax.set_yscale("log")
    if best_rows:
        ax.axhline(
            best_rows[0][2],
            color="black",
            linestyle="--",
            alpha=0.5,
            label=f"best.pt ppl = {best_rows[0][2]:.2f}",
        )
    ax.set_ylabel("Perplexity" + (" (log)" if log_ppl else ""))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)

    # ---- ChrF ---------------------------------------------------------
    ax = axes[2]
    ax.plot(epochs, chrfs, marker="o", color="tab:green", label="ChrF")
    ax.axvline(
        epochs[best_chrf_idx],
        color="tab:green",
        linestyle=":",
        alpha=0.5,
        label=f"max ChrF @ ep{epochs[best_chrf_idx]} ({chrfs[best_chrf_idx]:.2f})",
    )
    if best_rows:
        ax.axhline(
            best_rows[0][3],
            color="black",
            linestyle="--",
            alpha=0.5,
            label=f"best.pt ChrF = {best_rows[0][3]:.2f}",
        )
    ax.set_ylabel("ChrF")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="evaluate.py --compare_epochs の結果からグラフを生成する"
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--csv",
        default=None,
        help="CSV (epoch,val_loss,perplexity,chrf) のパス",
    )
    src.add_argument(
        "--txt",
        default=None,
        help="evaluate.py --compare_epochs のテキスト出力のパス",
    )
    parser.add_argument(
        "--model_size",
        default="large",
        help="グラフタイトル / 出力ファイル名に使うモデル名",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="出力 PNG パス (未指定なら fig/epoch_comparison_{size}.png)",
    )
    parser.add_argument(
        "--no_log_ppl",
        action="store_true",
        help="Perplexity を線形軸でプロットする (デフォルトは log)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv:
        rows = parse_csv(args.csv)
    elif args.txt:
        rows = parse_txt(args.txt)
    else:
        rows = DEFAULT_DATA_LARGE
        print("No --csv/--txt given. Using built-in 'large' results.")

    out_path = args.out or os.path.join(
        "fig", f"epoch_comparison_{args.model_size}.png"
    )
    plot_metrics(
        rows=rows,
        model_size=args.model_size,
        out_path=out_path,
        log_ppl=not args.no_log_ppl,
    )


if __name__ == "__main__":
    main()
