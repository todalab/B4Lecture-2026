"""Warmup + Cosine Decay 学習率スケジューラの可視化スクリプト

`training_utils.LearningRateScheduler` と同じロジックでステップごとの学習率を
計算し、グラフを描画する。総ステップ数・warmup ステップ数・ベース LR は
下のグローバル変数で変更できる。
"""

import math
import os

import matplotlib.pyplot as plt

# =====================================================================
# 設定（変更したいときはここをいじる）
# =====================================================================
TOTAL_STEPS: int = 4030  # 総ステップ数
WARMUP_STEPS: int = int(TOTAL_STEPS * 0.2)  # warmup ステップ数
BASE_LR: float = 1e-4  # ピーク（warmup 終了時）の学習率
OUTPUT_PATH: str = "fig/lr_schedule.png"  # 保存先（None ならファイル保存しない）
SHOW_FIGURE: bool = False  # True なら plt.show() で表示
# =====================================================================


def compute_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """1 ステップ分の学習率を返す（training_utils.LearningRateScheduler と同じ式）"""
    if step <= warmup_steps:
        return base_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_schedule(
    total_steps: int, warmup_steps: int, base_lr: float
) -> tuple[list[int], list[float]]:
    """ステップ 1 ~ total_steps の学習率列を返す"""
    steps = list(range(1, total_steps + 1))
    lrs = [compute_lr(s, warmup_steps, total_steps, base_lr) for s in steps]
    return steps, lrs


def plot_schedule(
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    output_path: str | None,
    show: bool,
) -> None:
    steps, lrs = build_schedule(total_steps, warmup_steps, base_lr)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, lrs, color="tab:blue", linewidth=1.5, label="learning rate")
    ax.axvline(
        warmup_steps,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=f"warmup end (step={warmup_steps})",
    )
    ax.axhline(
        base_lr,
        color="tab:gray",
        linestyle=":",
        linewidth=1.0,
        label=f"base_lr={base_lr:g}",
    )

    ax.set_title(
        f"Warmup + Cosine Decay LR Schedule\n"
        f"total_steps={total_steps}, warmup_steps={warmup_steps}, "
        f"base_lr={base_lr:g}"
    )
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_xlim(0, total_steps)
    ax.set_ylim(0, base_lr * 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_schedule(
        total_steps=TOTAL_STEPS,
        warmup_steps=WARMUP_STEPS,
        base_lr=BASE_LR,
        output_path=OUTPUT_PATH,
        show=SHOW_FIGURE,
    )
