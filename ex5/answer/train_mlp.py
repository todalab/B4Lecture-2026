import argparse
from pathlib import Path

import numpy as np

from data_utils import (
    accuracy,
    confusion_matrix,
    ensure_dir,
    load_npz,
    plot_confusion_matrix,
    plot_spectrum_examples,
    plot_training_curves,
    plot_waveform_examples,
    spectral_features,
    standardize_train_test,
    train_test_split,
)
from mlp import MLP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="answer/data/synthetic_music.npz")
    parser.add_argument("--fig-dir", default="answer/fig")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def evaluate(model, X, y, l2):
    probs, _ = model.forward(X)
    loss = model.loss(X, y, l2=l2)
    pred = np.argmax(probs, axis=1)
    return loss, accuracy(y, pred)


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    ensure_dir(fig_dir)

    X_wave, y, labels, sample_rate = load_npz(args.input)
    plot_waveform_examples(X_wave, y, labels, sample_rate, fig_dir / "waveform_examples.png")
    plot_spectrum_examples(X_wave, y, labels, sample_rate, fig_dir / "spectrum_examples.png")

    X = spectral_features(X_wave, sample_rate=sample_rate, n_bins=args.bins)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_ratio=args.test_ratio, seed=args.seed
    )
    X_train, X_test = standardize_train_test(X_train, X_test)

    model = MLP(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden,
        output_dim=len(labels),
        seed=args.seed,
        weight_scale=0.05,
    )

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in range(args.epochs):
        loss, grads = model.gradients(X_train, y_train, l2=args.l2)
        model.step(grads, lr=args.lr)

        train_loss, train_acc = evaluate(model, X_train, y_train, l2=args.l2)
        test_loss, test_acc = evaluate(model, X_test, y_test, l2=args.l2)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == args.epochs:
            print(
                f"epoch {epoch + 1:4d} "
                f"loss={loss:.4f} "
                f"train_acc={train_acc:.3f} "
                f"test_acc={test_acc:.3f}"
            )

    pred_test = model.predict(X_test)
    matrix = confusion_matrix(y_test, pred_test, n_classes=len(labels))
    plot_training_curves(history, fig_dir / "training_curves.png")
    plot_confusion_matrix(matrix, labels, fig_dir / "confusion_matrix.png")

    print(f"final test accuracy: {accuracy(y_test, pred_test):.4f}")


if __name__ == "__main__":
    main()
