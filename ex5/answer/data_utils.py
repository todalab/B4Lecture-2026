from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["labels"], int(data["sample_rate"])


def train_test_split(X, y, test_ratio=0.25, seed=0):
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        indices = rng.permutation(indices)
        n_test = int(len(indices) * test_ratio)
        test_parts.append(indices[:n_test])
        train_parts.append(indices[n_test:])
    train_idx = rng.permutation(np.concatenate(train_parts))
    test_idx = rng.permutation(np.concatenate(test_parts))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def spectral_features(waves, sample_rate, n_bins=128):
    window = np.hanning(waves.shape[1])
    spectrum = np.abs(np.fft.rfft(waves * window[None, :], axis=1))
    power = spectrum**2
    power_sum = power.sum(axis=1, keepdims=True) + 1e-12
    power_norm = power / power_sum
    log_spectrum = np.log1p(spectrum)

    max_freq = sample_rate / 2.0
    edges = np.linspace(0.0, max_freq, n_bins + 1)
    freqs = np.fft.rfftfreq(waves.shape[1], d=1.0 / sample_rate)

    features = np.zeros((waves.shape[0], n_bins), dtype=np.float64)
    for i in range(n_bins):
        mask = (edges[i] <= freqs) & (freqs < edges[i + 1])
        if np.any(mask):
            features[:, i] = log_spectrum[:, mask].mean(axis=1)

    non_dc = power_norm[:, 1:]
    top_power = np.sort(non_dc, axis=1)[:, -16:][:, ::-1]
    entropy = -np.sum(power_norm * np.log(power_norm + 1e-12), axis=1, keepdims=True)
    centroid = (power_norm * freqs[None, :]).sum(axis=1, keepdims=True) / max_freq
    bandwidth = np.sqrt(
        (power_norm * (freqs[None, :] - centroid * max_freq) ** 2).sum(axis=1, keepdims=True)
    )
    bandwidth = bandwidth / max_freq
    flatness = np.exp(np.mean(np.log(spectrum[:, 1:] + 1e-12), axis=1, keepdims=True))
    flatness = flatness / (np.mean(spectrum[:, 1:], axis=1, keepdims=True) + 1e-12)
    zero_crossing = np.mean(waves[:, 1:] * waves[:, :-1] < 0.0, axis=1, keepdims=True)

    summary = np.concatenate(
        [top_power, entropy, centroid, bandwidth, flatness, zero_crossing], axis=1
    )
    return np.concatenate([features, summary], axis=1)


def standardize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_waveform_examples(X, y, labels, sample_rate, output):
    n_classes = len(labels)
    t = np.arange(X.shape[1]) / sample_rate
    fig_height = max(7, 1.3 * n_classes)
    fig, axes = plt.subplots(n_classes, 1, figsize=(9, fig_height), sharex=True)
    axes = np.atleast_1d(axes)
    for label, ax in enumerate(axes):
        idx = np.where(y == label)[0][0]
        ax.plot(t, X[idx], lw=1.0)
        ax.set_ylabel(str(labels[label]))
        ax.set_ylim(-1.1, 1.1)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_spectrum_examples(X, y, labels, sample_rate, output):
    n_classes = len(labels)
    fig_height = max(7, 1.3 * n_classes)
    fig, axes = plt.subplots(n_classes, 1, figsize=(9, fig_height), sharex=True)
    axes = np.atleast_1d(axes)
    for label, ax in enumerate(axes):
        idx = np.where(y == label)[0][0]
        window = np.hanning(X.shape[1])
        spec = np.abs(np.fft.rfft(X[idx] * window))
        freq = np.fft.rfftfreq(X.shape[1], d=1.0 / sample_rate)
        ax.plot(freq, np.log1p(spec), lw=1.0)
        ax.set_ylabel(str(labels[label]))
    axes[-1].set_xlabel("frequency [Hz]")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_training_curves(history, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["test_loss"], label="test")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["test_acc"], label="test")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def plot_confusion_matrix(matrix, labels, output):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
