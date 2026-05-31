"""Debug script to analyze autoencoder thresholds and compute MSE metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from dataloaders.dataloader import MelSpectrogramDataset
from models.autoencoder import Autoencoder

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs/config_optuna_for_comp.yaml"
OPTUNA_RESULTS_PATH = BASE_DIR / "logs/mel_ae_optuna_for_comp/optimization_results.yaml"
OUTPUT_DIR = BASE_DIR / "outputs/debug_thresholds"


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_model(cfg: dict, optuna_results_path: Path, device: torch.device) -> Autoencoder:
    with open(optuna_results_path, "r", encoding="utf-8") as f:
        optuna_results = yaml.safe_load(f)

    best_params = optuna_results.get("best_params", {})
    hidden_channels1 = int(best_params.get("model.hidden_channels1", cfg["model"].get("hidden_channels1", 32)))
    hidden_channels2 = int(best_params.get("model.hidden_channels2", cfg["model"].get("hidden_channels2", 16)))
    learning_rate = float(best_params.get("train.learning_rate", cfg["train"]["learning_rate"]))

    log_dir = Path(cfg["hydra"]["sweep"]["dir"])
    ckpt_path = (
        log_dir
        / Path(f"hidden{hidden_channels1}_hidden{hidden_channels2}_lr{round(learning_rate, 4)}")
        / "ckpts"
        / "model_epoch_0009.pt"
    )

    print(f"Loading model from: {ckpt_path}")
    model = Autoencoder(
        in_channels=1,
        hidden_channels1=hidden_channels1,
        hidden_channels2=hidden_channels2,
        latent_channels=cfg["model"]["latent_channels"],
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def _compute_mse_metrics(
    input_mel_norm: torch.Tensor, recon_mel_norm: torch.Tensor, dataset: MelSpectrogramDataset
) -> dict:
    # input_mel_norm, recon_mel_norm : (B=1, 1, n_mels, frames)
    input_norm = input_mel_norm.squeeze(0)
    recon_norm = recon_mel_norm.squeeze(0)

    normalized_mse = torch.mean((input_norm - recon_norm) ** 2).item()

    # denormalize and compute dB-space MSE
    input_denorm = dataset.denormalize(input_norm)
    recon_denorm = dataset.denormalize(recon_norm)
    denormalized_mse = torch.mean((input_denorm - recon_denorm) ** 2).item()

    # per-frame MSE (mean over mel bins)
    per_frame_mse = torch.mean((input_norm - recon_norm) ** 2, dim=1)
    per_frame_mse_mean = torch.mean(per_frame_mse).item()
    per_frame_mse_max = torch.max(per_frame_mse).item()

    return {
        "normalized_mse": normalized_mse,
        "denormalized_mse": denormalized_mse,
        "per_frame_mse_mean": per_frame_mse_mean,
        "per_frame_mse_max": per_frame_mse_max,
    }


def _create_dataloader(cfg: dict, device: torch.device, db_min: float, db_max: float) -> DataLoader:
    data_config = cfg["dataset_normal_and_anomaly"]
    eval_list_path = BASE_DIR / data_config["eval_list"]
    data_dir = BASE_DIR / data_config["data_dir"]

    dataset = MelSpectrogramDataset(
        file_list_path=eval_list_path,
        data_dir_path=data_dir,
        sample_rate=cfg["dataset"]["sample_rate"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        n_mels=cfg["dataset"]["n_mels"],
        target_frames=cfg["dataset"]["target_frames"],
        db_min=db_min,
        db_max=db_max,
        device=device,
    )

    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def _plot_histograms(labels, normalized_mses, denormalized_mses, max_frame_mses, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_idx = [i for i, label in enumerate(labels) if label == 0]
    anomaly_idx = [i for i, label in enumerate(labels) if label == 1]

    metrics = [
        ("normalized_mse", normalized_mses),
        ("denormalized_mse", denormalized_mses),
        ("max_frame_mse", max_frame_mses),
    ]
    for metric_name, scores in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        normal_scores = [scores[i] for i in normal_idx]
        anomaly_scores = [scores[i] for i in anomaly_idx]
        if normal_scores:
            ax.hist(normal_scores, bins=30, alpha=0.6, label=f"Normal (n={len(normal_scores)})", color="blue")
        if anomaly_scores:
            ax.hist(anomaly_scores, bins=30, alpha=0.6, label=f"Anomaly (n={len(anomaly_scores)})", color="red")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {metric_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / f"hist_{metric_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved histogram: {output_dir / f'hist_{metric_name}.png'}")


def main() -> None:
    cfg = _load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing db_min/db_max from training data...")
    data_config = cfg["dataset_normal_and_anomaly"]
    train_list_path = BASE_DIR / data_config["train_list"]
    data_dir = BASE_DIR / data_config["data_dir"]

    db_min, db_max = MelSpectrogramDataset.compute_db_min_max(
        train_list_path,
        data_dir,
        sample_rate=cfg["dataset"]["sample_rate"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        n_mels=cfg["dataset"]["n_mels"],
        target_frames=cfg["dataset"]["target_frames"],
        device=torch.device("cpu"),
    )
    print(f"db_min: {db_min:.4f}, db_max: {db_max:.4f}")

    print("Loading model...")
    model = _load_model(cfg, OPTUNA_RESULTS_PATH, device)
    model.eval()

    print("Creating validation DataLoader...")
    val_loader = _create_dataloader(cfg, device, db_min, db_max)
    print(f"Validation dataset size: {len(val_loader.dataset)}")

    all_scores = {
        "normalized_mse": [],
        "denormalized_mse": [],
        "per_frame_mse_mean": [],
        "per_frame_mse_max": [],
        "label": [],
        "model_id": [],
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            mel_norm, label, model_id = batch
            mel_norm = mel_norm.to(device)
            recon_mel_norm = model(mel_norm)
            metrics = _compute_mse_metrics(mel_norm, recon_mel_norm, val_loader.dataset)
            all_scores["normalized_mse"].append(metrics["normalized_mse"])
            all_scores["denormalized_mse"].append(metrics["denormalized_mse"])
            all_scores["per_frame_mse_mean"].append(metrics["per_frame_mse_mean"])
            all_scores["per_frame_mse_max"].append(metrics["per_frame_mse_max"])
            all_scores["label"].append(int(label.item()))
            all_scores["model_id"].append(str(model_id[0]))
            if batch_idx % 50 == 0:
                print(f"  Processed {batch_idx}/{len(val_loader)} samples")

    labels = np.array(all_scores["label"])
    normalized_mses = np.array(all_scores["normalized_mse"])
    denormalized_mses = np.array(all_scores["denormalized_mse"])
    max_frame_mses = np.array(all_scores["per_frame_mse_max"])

    # Compute AUCs
    fpr_norm, tpr_norm, _ = roc_curve(labels, normalized_mses)
    auc_norm = auc(fpr_norm, tpr_norm)
    fpr_denorm, tpr_denorm, _ = roc_curve(labels, denormalized_mses)
    auc_denorm = auc(fpr_denorm, tpr_denorm)
    fpr_max_frame, tpr_max_frame, _ = roc_curve(labels, max_frame_mses)
    auc_max_frame = auc(fpr_max_frame, tpr_max_frame)

    print("\n=== AUC Scores ===")
    print(f"Normalized MSE AUC: {auc_norm:.4f}")
    print(f"Denormalized MSE AUC: {auc_denorm:.4f}")
    print(f"Max Frame MSE AUC: {auc_max_frame:.4f}")

    # Save CSV
    df = pd.DataFrame(
        {
            "model_id": all_scores["model_id"],
            "label": all_scores["label"],
            "normalized_mse": all_scores["normalized_mse"],
            "denormalized_mse": all_scores["denormalized_mse"],
            "per_frame_mse_mean": all_scores["per_frame_mse_mean"],
            "per_frame_mse_max": all_scores["per_frame_mse_max"],
        }
    )
    csv_path = OUTPUT_DIR / "debug_thresholds.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved scores to: {csv_path}")

    # Plot histograms
    _plot_histograms(labels, normalized_mses, denormalized_mses, max_frame_mses, OUTPUT_DIR)

    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
