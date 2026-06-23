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
def calculate_ChrF(model, val_loader, tgt_tokeniser, device):
    predictions, references = [], []

    for src, _, tgt_out in val_loader:
        generated = model.generate(src.to(device), BOS_IDX, EOS_IDX, max_len=MAX_LEN)
        generated = generated.cpu()
        tgt_out = tgt_out.cpu()

        for prediction_indices, reference_indices in zip(generated, tgt_out):
            predictions.append(tgt_tokeniser.decode(prediction_indices.tolist()))
            references.append(tgt_tokeniser.decode(reference_indices.tolist()))

    return CHRF().corpus_score(predictions, [references]).score


def main():
    device = get_device()
    _, val_loader, src_tokeniser, tgt_tokeniser = create_data_loaders()

    results = []
    for model_size in MODELS:
        checkpoint_path = Path("checkpoints") / f"translation_{model_size}_best.pt"
        if not checkpoint_path.exists():
            print(f"Skipping {model_size}: checkpoint not found")
            continue

        config = get_model_config(model_size)
        model = TranslationModel(
            len(src_tokeniser),
            len(tgt_tokeniser),
            max_seq_len=MAX_LEN * 2,
            **config,
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        _, perplexity = evaluate(model, val_loader, device)
        chrf = calculate_ChrF(model, val_loader, tgt_tokeniser, device)
        results.append({"model_size": model_size, "perplexity": perplexity, "chrf": chrf})

        print(f"{model_size}: perplexity={perplexity:.2f}, ChrF={chrf:.2f}")

    if not results:
        raise SystemExit("No checkpoints found.")

    Path("fig").mkdir(exist_ok=True)
    names = [r["model_size"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epoch_ranges = [(1, 15), (16, 30)]
    for model_size in names:
        metrics_path = Path("checkpoints") / f"translation_{model_size}_metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open() as f:
            val_perplexities = json.load(f)["val_perplexities"]
        for ax, (start, end) in zip(axes, epoch_ranges):
            epochs = range(start, min(end, len(val_perplexities)) + 1)
            values = val_perplexities[start - 1 : end]
            ax.plot(epochs, values, label=model_size)

    for ax, (start, end) in zip(axes, epoch_ranges):
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(start, end + 1))
        ax.set_xlim(start, end)
        ax.legend()
    fig.suptitle("Perplexity")
    fig.tight_layout()
    fig.savefig("fig/perplexity.png", dpi=200)
    plt.close(fig)

    plt.figure(figsize=(5, 4))
    plt.plot(names, [r["chrf"] for r in results], marker="o")
    plt.title("ChrF Score")
    plt.xlabel("Model")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("fig/chrf.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
