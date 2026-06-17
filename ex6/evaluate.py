"""
Evaluation and Visualization for Ex6 B4 Lecture - Translation Task
BLEU スコアの計算と学習曲線の可視化

使用例:
    python evaluate.py --checkpoint checkpoints/translation_small_best.pt \
                       --model_size small --num_samples 500
"""

import argparse
import json
import logging
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# sacrebleu が利用可能なら使用、なければ簡易実装にフォールバック
try:
    import sacrebleu as _sacrebleu

    def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
        """BLEU スコアを計算する (sacrebleu 使用)"""
        result = _sacrebleu.corpus_bleu(hypotheses, [references])
        return result.score

except ImportError:
    from collections import Counter

    def _ngram_counts(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    def compute_bleu(hypotheses: List[str], references: List[str], max_n: int = 4) -> float:
        """簡易 BLEU スコア (sacrebleu 未インストール時のフォールバック)"""
        precisions = []
        for n in range(1, max_n + 1):
            correct, total = 0, 0
            for hyp, ref in zip(hypotheses, references):
                hyp_counts = _ngram_counts(hyp.split(), n)
                ref_counts = _ngram_counts(ref.split(), n)
                clipped = {k: min(v, ref_counts[k]) for k, v in hyp_counts.items()}
                correct += sum(clipped.values())
                total += sum(hyp_counts.values())
            precisions.append(correct / max(total, 1))

        if min(precisions) == 0:
            return 0.0

        # Brevity penalty
        hyp_len = sum(len(h.split()) for h in hypotheses)
        ref_len = sum(len(r.split()) for r in references)
        bp = 1.0 if hyp_len >= ref_len else np.exp(1 - ref_len / max(hyp_len, 1))

        score = bp * np.exp(sum(np.log(p) for p in precisions) / max_n)
        return score * 100.0


def translate_sentence(
    model: nn.Module,
    src_sentence: str,
    src_tokenizer,
    tgt_tokenizer,
    device: torch.device,
    max_len: int = 100,
    bos_idx: int = 2,
    eos_idx: int = 3,
) -> str:
    """1 文を翻訳する (greedy decoding)"""
    model.eval()
    src_ids = src_tokenizer.encode(src_sentence, add_eos=True, max_len=max_len)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        generated = model.generate(src_tensor, bos_idx, eos_idx, max_len=max_len)

    return tgt_tokenizer.decode(generated[0].cpu().tolist())


def evaluate_bleu(
    model: nn.Module,
    val_loader,
    src_tokenizer,
    tgt_tokenizer,
    device: torch.device,
    num_samples: int = 500,
    max_len: int = 100,
) -> float:
    """検証データで BLEU スコアを計算する

    Args:
        num_samples: 評価するサンプル数 (多いほど正確だが時間がかかる)

    Returns:
        BLEU スコア (0~100)
    """
    from data_loader import BOS_IDX, EOS_IDX

    model.eval()
    hypotheses, references = [], []
    count = 0

    with torch.no_grad():
        for src, _, tgt_out in val_loader:
            src = src.to(device)
            generated = model.generate(src, BOS_IDX, EOS_IDX, max_len=max_len)

            for i in range(src.size(0)):
                hyp = tgt_tokenizer.decode(generated[i].cpu().tolist())
                ref = tgt_tokenizer.decode(tgt_out[i].tolist())
                hypotheses.append(hyp)
                references.append(ref)
                count += 1
                if count >= num_samples:
                    break

            if count >= num_samples:
                break

    bleu = compute_bleu(hypotheses, references)
    logger.info(f"BLEU score ({count} samples): {bleu:.2f}")
    return bleu


def plot_training_curves(
    metrics_paths: List[str],
    model_names: List[str],
    save_path: Optional[str] = None,
):
    """学習曲線をプロット"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for metrics_path, model_name in zip(metrics_paths, model_names):
        if not os.path.exists(metrics_path):
            logger.warning(f"Not found: {metrics_path}")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        epochs = list(range(1, len(metrics["train_losses"]) + 1))

        axes[0].plot(epochs, metrics["train_losses"], label=f"{model_name} (train)", linestyle="-")
        axes[0].plot(epochs, metrics["val_losses"], label=f"{model_name} (val)", linestyle="--")
        axes[1].plot(epochs, metrics["val_perplexities"], label=model_name)
        axes[2].plot(epochs, metrics["learning_rates"], label=model_name)

    for ax, title, ylabel in zip(
        axes,
        ["Loss", "Validation Perplexity", "Learning Rate"],
        ["Loss", "Perplexity", "LR"],
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    axes[1].set_yscale("log")
    axes[2].set_yscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def show_sample_translations(
    model: nn.Module,
    src_sentences: List[str],
    src_tokenizer,
    tgt_tokenizer,
    device: torch.device,
    max_len: int = 64,
):
    """サンプル文の翻訳結果を表示する"""
    print("\n=== Sample Translations ===")
    for src in src_sentences:
        tgt = translate_sentence(
            model, src, src_tokenizer, tgt_tokenizer, device, max_len
        )
        print(f"  EN: {src}")
        print(f"  JA: {tgt}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate translation model")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument(
        "--model_size",
        choices=["tiny", "small", "medium", "large"],
        default="small",
    )
    parser.add_argument("--num_samples", type=int, default=500, help="Samples for BLEU eval")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--plot_training", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    from data_loader import create_data_loaders
    from training_utils import get_device
    from transformer_skeleton import TranslationModel, get_model_config

    device = get_device()

    # データ読み込み
    logger.info("Loading data ...")
    _, val_loader, src_tokenizer, tgt_tokenizer = create_data_loaders(
        max_len=args.max_len, batch_size=args.batch_size
    )

    # モデル読み込み
    config = get_model_config(args.model_size)
    model = TranslationModel(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_encoder_layers=config["n_encoder_layers"],
        n_decoder_layers=config["n_decoder_layers"],
        d_ff=config["d_ff"],
        max_seq_len=args.max_len * 2,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # BLEU 評価
    bleu = evaluate_bleu(model, val_loader, src_tokenizer, tgt_tokenizer, device, args.num_samples)
    print(f"\nBLEU score: {bleu:.2f}")

    # サンプル翻訳
    sample_sentences = [
        "I will check the schedule .",
        "Thank you for your help .",
        "Please send me the report .",
        "The meeting is at three o'clock .",
    ]
    show_sample_translations(model, sample_sentences, src_tokenizer, tgt_tokenizer, device)

    # 学習曲線
    if args.plot_training:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        metrics_paths = [
            os.path.join(checkpoint_dir, f"translation_{size}_metrics.json")
            for size in ["tiny", "small", "medium", "large"]
        ]
        names = ["tiny", "small", "medium", "large"]
        plot_training_curves(
            metrics_paths,
            names,
            save_path=os.path.join(args.output_dir, "training_curves.png"),
        )


if __name__ == "__main__":
    main()
