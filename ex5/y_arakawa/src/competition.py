"""Evaluate AE-encoder + MLPClassifier on the eval set defined by eval_mapping.csv.

For each WAV file under ``data/eval/`` whose ground-truth label is recorded in
``eval_mapping.csv``, run the trained encoder + MLP classifier, then compute the
per-model F1 score and their harmonic mean (the competition metric).

The threshold selection logic mirrors ``eval_classifier.py``: by default it uses
the validation set (``dataset_normal_and_anomaly.eval_list``) to pick a threshold
that maximizes the harmonic mean of per-model F1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset

from dataloaders.dataloader import MelSpectrogramDataset, create_dataloader, normalize_mel_db
from eval_classifier import (
    _collect_scores,
    _load_ae_state_dict,
    _load_classifier_state_dict,
    _plot_confusion_matrix,
    _plot_prob_boxplot,
    _plot_prob_histogram,
    _plot_roc,
    _resolve_data_dir_path,
    _resolve_existing_path,
    _resolve_list_path,
    _select_threshold_by_f1_hmean,
)
from models.autoencoder import Autoencoder
from models.mlp_classifier import MLPClassifier
from utils.seed import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OmegaConf.register_new_resolver("round", round, replace=True)


class EvalMappingDataset(Dataset):
    """Dataset that pairs eval WAV files with ground-truth labels from a mapping CSV."""

    def __init__(
        self,
        records: list[tuple[Path, int, str]],
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        target_frames: int,
        db_min: float,
        db_max: float,
        device: torch.device,
    ) -> None:
        self.records = records
        self.sample_rate = sample_rate
        self.target_frames = target_frames
        self.db_min = db_min
        self.db_max = db_max
        self.device = device
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        ).to(device)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power").to(device)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        file_path, label, model_id = self.records[index]
        waveform_np, _ = librosa.load(file_path, sr=self.sample_rate)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(self.device)
        mel_db = self.to_db(self.mel(waveform))
        mel_db = MelSpectrogramDataset._pad_or_trim_mel_tensor(mel_db, self.target_frames)
        mel_norm = normalize_mel_db(mel_db, self.db_min, self.db_max)
        return mel_norm, int(label), model_id


def _build_eval_records(mapping_csv: Path, eval_dir: Path) -> list[tuple[Path, int, str]]:
    """Read eval_mapping.csv and return (path, label, model_id) records."""
    df = pd.read_csv(mapping_csv)
    required = {"eval_filename", "condition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"eval_mapping.csv is missing required columns: {missing}")

    records: list[tuple[Path, int, str]] = []
    for _, row in df.iterrows():
        fname = str(row["eval_filename"]).strip()
        condition = str(row["condition"]).strip().lower()
        if condition not in ("normal", "anomaly"):
            raise ValueError(f"Unexpected condition '{condition}' for {fname}")
        label = 0 if condition == "normal" else 1
        # filename like "model_03_07zU.wav" -> model_id = "03"
        parts = Path(fname).stem.split("_")
        if len(parts) < 2:
            raise ValueError(f"Unexpected filename format: {fname}")
        model_id = parts[1]
        path = eval_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Eval WAV not found: {path}")
        records.append((path, label, model_id))
    return records


def _compute_per_model_f1(labels: np.ndarray, preds: np.ndarray, model_ids: list[str]) -> dict[str, float]:
    """Compute per-model F1 score from binary labels/preds."""
    model_ids_arr = np.asarray(model_ids)
    per_model_f1: dict[str, float] = {}
    for mid in sorted(np.unique(model_ids_arr).tolist()):
        mask = model_ids_arr == mid
        y = labels[mask].astype(int)
        p = preds[mask].astype(int)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2.0 * precision * recall / (precision + recall)
        per_model_f1[str(mid)] = f1
    return per_model_f1


def _harmonic_mean_f1(per_model_f1: dict[str, float], eps: float = 1e-12) -> float:
    """Harmonic mean of per-model F1 scores (final competition metric)."""
    if not per_model_f1:
        return 0.0
    vals = np.asarray([max(v, eps) for v in per_model_f1.values()])
    return float(len(vals) / np.sum(1.0 / vals))


def _select_threshold(
    cfg: DictConfig,
    autoencoder: Autoencoder,
    classifier: MLPClassifier,
    device: torch.device,
    db_min: float,
    db_max: float,
    data_dir_path: Path | None,
) -> float:
    """Select a decision threshold according to ``cfg.eval.threshold_mode``."""
    eval_cfg = cfg.eval
    threshold_mode = str(eval_cfg.get("threshold_mode", "fixed"))
    if threshold_mode not in ("youden_val", "f1_hmean_val"):
        return float(eval_cfg.get("threshold", 0.5))

    val_list_path = _resolve_list_path(cfg.dataset_normal_and_anomaly.eval_list)
    val_loader = create_dataloader(
        val_list_path,
        data_dir_path,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        n_mels=cfg.dataset.n_mels,
        target_frames=cfg.dataset.target_frames,
        seed=int(cfg.train.seed),
        device=device,
        db_min=db_min,
        db_max=db_max,
    )
    val_probs, val_labels, val_model_ids = _collect_scores(autoencoder, classifier, val_loader, device)
    if np.unique(val_labels).size < 2:
        logger.warning("Validation set has only one class; falling back to fixed threshold.")
        return float(eval_cfg.get("threshold", 0.5))

    if threshold_mode == "youden_val":
        v_fpr, v_tpr, v_thr = roc_curve(val_labels, val_probs)
        best_idx = int(np.argmax(v_tpr - v_fpr))
        threshold = float(v_thr[best_idx])
        logger.info(f"Selected threshold from val Youden's J: {threshold:.4f}")
        return threshold

    grid_n = int(eval_cfg.get("f1_grid_size", 991))
    grid = np.linspace(0.01, 0.99, grid_n)
    threshold, best_hmean, per_model_f1 = _select_threshold_by_f1_hmean(val_probs, val_labels, val_model_ids, grid=grid)
    logger.info(f"Selected threshold from val F1 harmonic mean: {threshold:.4f} (H-mean F1={best_hmean:.4f})")
    for mid, f1 in sorted(per_model_f1.items()):
        logger.info(f"  val F1[{mid}] @ {threshold:.4f} = {f1:.4f}")
    return threshold


def evaluate(cfg: DictConfig) -> float:
    """Run the encoder+MLP on the eval set and return the harmonic mean of per-model F1."""
    set_seed(int(cfg.train.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU available:", torch.cuda.is_available())

    eval_cfg = cfg.eval

    # Resolve the eval mapping CSV and corresponding WAV directory.
    repo_root = Path(__file__).resolve().parent.parent
    default_csv = repo_root.parent / "data" / "eval" / "eval_mapping.csv"
    mapping_csv = Path(str(eval_cfg.get("eval_mapping_csv", default_csv)))
    if not mapping_csv.is_absolute():
        mapping_csv = (Path.cwd() / mapping_csv).resolve()
    mapping_csv = _resolve_existing_path(mapping_csv)
    eval_dir = Path(str(eval_cfg.get("eval_data_dir", mapping_csv.parent)))
    if not eval_dir.is_absolute():
        eval_dir = (Path.cwd() / eval_dir).resolve()
    logger.info(f"Eval mapping CSV: {mapping_csv}")
    logger.info(f"Eval WAV dir    : {eval_dir}")

    # db_min/db_max from the training normal+anomaly list (matches training stats).
    train_list_path = _resolve_list_path(cfg.dataset_normal_and_anomaly.train_list)
    data_dir_path = _resolve_data_dir_path(cfg.dataset_normal_and_anomaly.get("data_dir", cfg.dataset.data_dir))
    db_min, db_max = MelSpectrogramDataset.compute_db_min_max(
        train_list_path,
        data_dir_path,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        n_mels=cfg.dataset.n_mels,
        target_frames=cfg.dataset.target_frames,
        device=torch.device("cpu"),
    )

    # Build models.
    variant = str(cfg.model.get("variant", "fc"))
    autoencoder = Autoencoder(
        in_channels=1,
        hidden_channels1=cfg.model.hidden_channels1,
        hidden_channels2=cfg.model.hidden_channels2,
        latent_channels=cfg.model.latent_channels,
        variant=variant,
    ).to(device)

    with torch.no_grad():
        dummy = torch.zeros(1, 1, cfg.dataset.n_mels, cfg.dataset.target_frames, device=device)
        latent_shape = autoencoder.encode(dummy).shape
    input_dim = int(torch.tensor(latent_shape[1:]).prod().item())
    logger.info(f"Latent shape: {tuple(latent_shape)} -> MLP input_dim={input_dim}")

    classifier = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=list(cfg.classifier.get("hidden_dims", [64, 32])),
        dropout=float(cfg.classifier.get("dropout", 0.0)),
    ).to(device)

    clf_ckpt_path = _resolve_existing_path(eval_cfg.classifier_ckpt)
    logger.info(f"Loading classifier checkpoint from {clf_ckpt_path}")
    ckpt = torch.load(clf_ckpt_path, map_location=device)
    autoencoder.load_state_dict(_load_ae_state_dict(ckpt, cfg.classifier.get("ae_ckpt")))
    classifier.load_state_dict(_load_classifier_state_dict(ckpt))

    # Threshold selection.
    threshold = _select_threshold(cfg, autoencoder, classifier, device, db_min, db_max, data_dir_path)

    # Build the eval loader.
    records = _build_eval_records(mapping_csv, eval_dir)
    logger.info(f"Eval samples: {len(records)}")
    eval_dataset = EvalMappingDataset(
        records=records,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        n_mels=cfg.dataset.n_mels,
        target_frames=cfg.dataset.target_frames,
        db_min=db_min,
        db_max=db_max,
        device=device,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    probs, labels, model_ids = _collect_scores(autoencoder, classifier, eval_loader, device)
    preds = (probs >= threshold).astype(int)

    # Save predictions.
    output_dir = Path(str(eval_cfg.get("output_dir", "outputs/eval_classifier"))) / "competition"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eval_predictions.csv"
    pd.DataFrame(
        {
            "eval_filename": [r[0].name for r in records],
            "model_id": model_ids,
            "label": labels.astype(int),
            "prob_anomaly": probs,
            "pred": preds,
        }
    ).to_csv(csv_path, index=False)
    logger.info(f"Saved predictions: {csv_path}")

    # Save plots (same set as eval_classifier.py).
    if np.unique(labels).size >= 2:
        roc_path = output_dir / "roc_curve.png"
        roc_auc = _plot_roc(labels, probs, roc_path, title="Encoder + MLP (competition eval)", threshold=threshold)
        logger.info(f"Saved ROC curve: {roc_path} (AUC={roc_auc:.4f})")
    else:
        logger.warning("Eval set has only one class; skipping ROC curve.")

    hist_path = output_dir / "prob_histogram.png"
    _plot_prob_histogram(labels, probs, threshold, hist_path, title="MLP output distribution (competition eval)")
    logger.info(f"Saved probability histogram: {hist_path}")

    box_path = output_dir / "prob_boxplot.png"
    _plot_prob_boxplot(labels, probs, box_path, title="MLP output per class (competition eval)")
    logger.info(f"Saved probability boxplot: {box_path}")

    cm_path = output_dir / "confusion_matrix.png"
    _plot_confusion_matrix(labels, preds, cm_path, title=f"Confusion matrix @ threshold={threshold:.3f}")
    logger.info(f"Saved confusion matrix: {cm_path}")

    # Compute metrics.
    per_model_f1 = _compute_per_model_f1(labels, preds, model_ids)
    f1_hmean = _harmonic_mean_f1(per_model_f1)

    # Per-model counts (for logging clarity).
    model_ids_arr = np.asarray(model_ids)

    print("\n=== Competition eval results ===")
    print(f"Threshold        : {threshold:.4f}")
    print(f"N total samples  : {len(records)}")
    print(f"N normal         : {int((labels == 0).sum())}")
    print(f"N anomaly        : {int((labels == 1).sum())}")
    print(f"\n--- Per-model F1 @ threshold={threshold:.4f} ---")
    for mid in sorted(per_model_f1.keys()):
        mask = model_ids_arr == mid
        y = labels[mask].astype(int)
        p = preds[mask].astype(int)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        print(f"  model_{mid}: F1={per_model_f1[mid]:.4f}  TP={tp} FP={fp} FN={fn} TN={tn}  (n={mask.sum()})")
    print(f"\nHarmonic mean of F1 (final metric): {f1_hmean:.4f}")

    # Append a summary file.
    summary_path = output_dir / "eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"threshold={threshold:.6f}\n")
        for mid in sorted(per_model_f1.keys()):
            f.write(f"f1_model_{mid}={per_model_f1[mid]:.6f}\n")
        f.write(f"f1_harmonic_mean={f1_hmean:.6f}\n")
    logger.info(f"Saved summary: {summary_path}")

    return f1_hmean


@hydra.main(version_base=None, config_path="../configs", config_name="config_classifier")
def main(cfg: DictConfig) -> float:
    return evaluate(cfg)


if __name__ == "__main__":
    main()
