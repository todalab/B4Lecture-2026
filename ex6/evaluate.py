"""Evaluate all trained model sizes and plot perplexity / ChrF."""

import json
from inspect import signature
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sacrebleu.metrics.chrf import CHRF

from data_loader import BOS_IDX, EOS_IDX, create_data_loaders
from training_utils import evaluate, get_device
from transformer_skeleton import TranslationModel, get_model_config


MODELS = ["tiny", "small", "medium", "large"]
MAX_LEN = signature(create_data_loaders).parameters["max_len"].default


@torch.no_grad()
def calculate_chrf(model, val_loader, tgt_tokenizer, device):
    predictions, references = [], []

    for src, _, tgt_out in val_loader:
        generated = model.generate(src.to(device), BOS_IDX, EOS_IDX, max_len=MAX_LEN)

        for pred_ids, ref_ids in zip(generated.cpu().tolist(), tgt_out.tolist()):
            predictions.append(tgt_tokenizer.decode(pred_ids))
            references.append(tgt_tokenizer.decode(ref_ids))

    return CHRF().corpus_score(predictions, [references]).score


def main():
    device = get_device()
    _, val_loader, src_tokenizer, tgt_tokenizer = create_data_loaders()

    results = []
    for model_size in MODELS:
        checkpoint_path = Path("checkpoints") / f"translation_{model_size}_best.pt"
        if not checkpoint_path.exists():
            print(f"Skipping {model_size}: checkpoint not found")
            continue

        config = get_model_config(model_size)
        model = TranslationModel(
            len(src_tokenizer),
            len(tgt_tokenizer),
            max_seq_len=MAX_LEN * 2,
            **config,
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        _, perplexity = evaluate(model, val_loader, device)
        chrf = calculate_chrf(model, val_loader, tgt_tokenizer, device)
        results.append({"model_size": model_size, "perplexity": perplexity, "chrf": chrf})

        print(f"{model_size}: perplexity={perplexity:.2f}, ChrF={chrf:.2f}")

    if not results:
        raise SystemExit("No checkpoints found.")

    Path("fig").mkdir(exist_ok=True)
    names = [r["model_size"] for r in results]

    plt.figure(figsize=(6, 4))
    for model_size in names:
        metrics_path = Path("checkpoints") / f"translation_{model_size}_metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open() as f:
            metrics = json.load(f)
        epochs = range(1, len(metrics["val_perplexities"]) + 1)
        plt.plot(epochs, metrics["val_perplexities"], marker="o", label=model_size)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig/perplexity.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.bar(names, [r["chrf"] for r in results])
    plt.tight_layout()
    plt.savefig("fig/chrf.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
