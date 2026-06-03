"""Debug script to analyze autoencoder thresholds and compute anomaly scores."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
    ckpt_path = log_dir / Path(f"lr{round(learning_rate, 4)}") / "ckpts" / "model_epoch_0009.pt"

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


def _split_normal_and_anomaly_paths(file_list_path: Path, data_dir_path: Path) -> tuple[list[Path], list[Path]]:
    files = MelSpectrogramDataset._load_file_list(file_list_path, data_dir_path)
    normal_files: list[Path] = []
    anomaly_files: list[Path] = []
    for file_path in files:
        parts = file_path.stem.split("_")
        if len(parts) < 4:
            raise ValueError(f"Unexpected file name format: {file_path.name}")
        label = parts[2]
        if label == "normal":
            normal_files.append(file_path)
        elif label == "anomaly":
            anomaly_files.append(file_path)
        else:
            raise ValueError(f"Unknown label in file name: {file_path.name}")
    return normal_files, anomaly_files


def _build_dataset(
    file_list_path: Path,
    data_dir: Path,
    cfg: dict,
    device: torch.device,
    db_min: float,
    db_max: float,
) -> MelSpectrogramDataset:
    return MelSpectrogramDataset(
        file_list_path=file_list_path,
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


def _collect_latent_stats(
    model: Autoencoder, dataset: MelSpectrogramDataset, device: torch.device
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    latents: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for mel_norm, _, _ in loader:
            z = model.encode(mel_norm.to(device))
            latents.append(z.cpu().numpy())
    latent_mat = np.concatenate(latents, axis=0)
    mean = latent_mat.mean(axis=0)
    std = latent_mat.std(axis=0) + 1e-6
    return {"mean": mean, "std": std}


def _collect_features_and_labels(
    model: Autoencoder,
    dataset: MelSpectrogramDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    latent_list: list[np.ndarray] = []
    flat_mel_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for mel_norm, label, _ in loader:
            z = model.encode(mel_norm.to(device)).cpu().numpy()
            latent_list.append(z)
            flat_mel_list.append(mel_norm.numpy().reshape(mel_norm.shape[0], -1))
            label_list.append(label.numpy())
    return (
        np.concatenate(latent_list, axis=0),
        np.concatenate(flat_mel_list, axis=0),
        np.concatenate(label_list, axis=0),
    )


def run_mlp_hypersearch(train_X: np.ndarray, train_y: np.ndarray, cv: int = 3) -> Pipeline:
    """Grid-search small MLP hyperparameters and return best pipeline.

    Returns a fitted Pipeline with steps ('scaler','mlp').
    """
    pipe = Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(max_iter=2000, random_state=0))])
    param_grid = {
        "mlp__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [1e-3, 1e-4],
        "mlp__activation": ["relu"],
    }
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=1)
    gs.fit(train_X, train_y)
    print(f"Best MLP params: {gs.best_params_}, best CV ROC-AUC: {gs.best_score_:.4f}")
    return gs.best_estimator_


def _collect_engineered_features(
    model: Autoencoder,
    dataset: MelSpectrogramDataset,
    device: torch.device,
    latent_stats: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    features: list[list[float]] = []
    labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for mel_norm, label, _ in loader:
            mel_norm = mel_norm.to(device)
            recon = model(mel_norm)
            metrics = _compute_mse_metrics(mel_norm, recon, dataset)
            latent_score = _latent_diag_mahalanobis(model.encode(mel_norm), latent_stats)
            features.append(
                [
                    metrics["normalized_mse"],
                    metrics["denormalized_mse"],
                    metrics["per_frame_mse_max"],
                    metrics["per_frame_mse_top10"],
                    latent_score,
                ]
            )
            labels.append(int(label.item()))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


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
    per_frame_mse_top10 = (
        torch.topk(per_frame_mse.flatten(), k=max(1, int(0.1 * per_frame_mse.numel()))).values.mean().item()
    )

    return {
        "normalized_mse": normalized_mse,
        "denormalized_mse": denormalized_mse,
        "per_frame_mse_mean": per_frame_mse_mean,
        "per_frame_mse_max": per_frame_mse_max,
        "per_frame_mse_top10": per_frame_mse_top10,
    }


def _latent_diag_mahalanobis(z: torch.Tensor, latent_stats: dict[str, np.ndarray]) -> float:
    z_np = z.detach().cpu().numpy().reshape(-1)
    mean = latent_stats["mean"].reshape(-1)
    std = latent_stats["std"].reshape(-1)
    return float(np.mean(((z_np - mean) / std) ** 2))


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


def _plot_histograms(labels, score_map: dict[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_idx = [i for i, label in enumerate(labels) if label == 0]
    anomaly_idx = [i for i, label in enumerate(labels) if label == 1]

    metrics = list(score_map.items())
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
    train_normal_paths, _ = _split_normal_and_anomaly_paths(train_list_path, data_dir)
    eval_normal_paths, eval_anomaly_paths = _split_normal_and_anomaly_paths(
        BASE_DIR / data_config["eval_list"], data_dir
    )

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

    print("Collecting latent stats from normal training data...")
    train_normal_dataset = _build_dataset(
        train_normal_paths,
        data_dir,
        cfg,
        torch.device("cpu"),
        db_min,
        db_max,
    )
    latent_stats = _collect_latent_stats(model, train_normal_dataset, device)
    print("Latent stats collected.")

    print("Collecting supervised features for classifier baselines...")
    train_eval_dataset = _build_dataset(train_list_path, data_dir, cfg, torch.device("cpu"), db_min, db_max)
    eval_dataset = _build_dataset(
        BASE_DIR / data_config["eval_list"], data_dir, cfg, torch.device("cpu"), db_min, db_max
    )
    train_latent_feat, train_flat_feat, train_y = _collect_features_and_labels(model, train_eval_dataset, device)
    eval_latent_feat, eval_flat_feat, eval_y = _collect_features_and_labels(model, eval_dataset, device)
    print("Feature collection done.")

    all_scores = {
        "normalized_mse": [],
        "denormalized_mse": [],
        "per_frame_mse_mean": [],
        "per_frame_mse_max": [],
        "per_frame_mse_top10": [],
        "latent_mahalanobis": [],
        "combined_score": [],
        "latent_logreg_score": [],
        "flat_logreg_score": [],
        "latent_svm_score": [],
        "latent_mlp_score": [],
        "engineered_mlp_score": [],
        "latent_meta_mlp_score": [],
        "label": [],
        "model_id": [],
    }

    latent_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(cfg["train"]["seed"])),
    )
    flat_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(cfg["train"]["seed"])),
    )
    latent_svm = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            gamma="scale",
            random_state=int(cfg["train"]["seed"]),
        ),
    )
    latent_mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            max_iter=1000,
            random_state=int(cfg["train"]["seed"]),
        ),
    )
    engineered_mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            alpha=1e-4,
            max_iter=2000,
            random_state=int(cfg["train"]["seed"]),
        ),
    )
    latent_clf.fit(train_latent_feat, train_y)
    flat_clf.fit(train_flat_feat, train_y)
    latent_svm.fit(train_latent_feat, train_y)
    latent_mlp.fit(train_latent_feat, train_y)
    train_engineered_feat, train_engineered_y = _collect_engineered_features(
        model, train_eval_dataset, device, latent_stats
    )
    eval_engineered_feat, _ = _collect_engineered_features(model, eval_dataset, device, latent_stats)
    engineered_mlp.fit(train_engineered_feat, train_engineered_y)

    # --- MLP hyperparameter search on latent features ---
    try:
        best_latent_pipe = run_mlp_hypersearch(train_latent_feat, train_y, cv=3)
        latent_mlp_hs_scores = best_latent_pipe.predict_proba(eval_latent_feat)[:, 1]
    except Exception as e:
        print(f"MLP hypersearch on latent failed: {e}")
        latent_mlp_hs_scores = latent_mlp_scores

    # --- Combine latent + engineered features and run hypersearch ---
    try:
        train_meta_feat = np.hstack([train_latent_feat, train_engineered_feat])
        eval_meta_feat = np.hstack([eval_latent_feat, eval_engineered_feat])
        best_meta_pipe = run_mlp_hypersearch(train_meta_feat, train_engineered_y, cv=3)
        meta_mlp_scores = best_meta_pipe.predict_proba(eval_meta_feat)[:, 1]
    except Exception as e:
        print(f"Meta MLP hypersearch failed: {e}")
        meta_mlp_scores = engineered_mlp_scores

    latent_logreg_scores = latent_clf.predict_proba(eval_latent_feat)[:, 1]
    flat_logreg_scores = flat_clf.predict_proba(eval_flat_feat)[:, 1]
    latent_svm_scores = latent_svm.predict_proba(eval_latent_feat)[:, 1]
    latent_mlp_scores = latent_mlp.predict_proba(eval_latent_feat)[:, 1]
    engineered_mlp_scores = engineered_mlp.predict_proba(eval_engineered_feat)[:, 1]

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            mel_norm, label, model_id = batch
            mel_norm = mel_norm.to(device)
            recon_mel_norm = model(mel_norm)
            metrics = _compute_mse_metrics(mel_norm, recon_mel_norm, val_loader.dataset)
            latent_score = _latent_diag_mahalanobis(model.encode(mel_norm), latent_stats)
            combined_score = metrics["normalized_mse"] + 0.5 * latent_score
            all_scores["normalized_mse"].append(metrics["normalized_mse"])
            all_scores["denormalized_mse"].append(metrics["denormalized_mse"])
            all_scores["per_frame_mse_mean"].append(metrics["per_frame_mse_mean"])
            all_scores["per_frame_mse_max"].append(metrics["per_frame_mse_max"])
            all_scores["per_frame_mse_top10"].append(metrics["per_frame_mse_top10"])
            all_scores["latent_mahalanobis"].append(latent_score)
            all_scores["combined_score"].append(combined_score)
            all_scores["latent_logreg_score"].append(float(latent_logreg_scores[batch_idx]))
            all_scores["flat_logreg_score"].append(float(flat_logreg_scores[batch_idx]))
            all_scores["latent_svm_score"].append(float(latent_svm_scores[batch_idx]))
            all_scores["latent_mlp_score"].append(float(latent_mlp_scores[batch_idx]))
            all_scores["engineered_mlp_score"].append(float(engineered_mlp_scores[batch_idx]))
            all_scores["latent_meta_mlp_score"].append(float(meta_mlp_scores[batch_idx]))
            all_scores["label"].append(int(label.item()))
            all_scores["model_id"].append(str(model_id[0]))
            if batch_idx % 50 == 0:
                print(f"  Processed {batch_idx}/{len(val_loader)} samples")

    labels = np.array(all_scores["label"])
    normalized_mses = np.array(all_scores["normalized_mse"])
    denormalized_mses = np.array(all_scores["denormalized_mse"])
    max_frame_mses = np.array(all_scores["per_frame_mse_max"])
    top10_mses = np.array(all_scores["per_frame_mse_top10"])
    latent_mahas = np.array(all_scores["latent_mahalanobis"])
    combined_scores = np.array(all_scores["combined_score"])
    latent_logreg_scores = np.array(all_scores["latent_logreg_score"])
    flat_logreg_scores = np.array(all_scores["flat_logreg_score"])
    latent_svm_scores = np.array(all_scores["latent_svm_score"])
    latent_mlp_scores = np.array(all_scores["latent_mlp_score"])
    engineered_mlp_scores = np.array(all_scores["engineered_mlp_score"])
    latent_meta_mlp_scores = np.array(all_scores["latent_meta_mlp_score"])

    # Compute AUCs
    fpr_norm, tpr_norm, _ = roc_curve(labels, normalized_mses)
    auc_norm = auc(fpr_norm, tpr_norm)
    fpr_denorm, tpr_denorm, _ = roc_curve(labels, denormalized_mses)
    auc_denorm = auc(fpr_denorm, tpr_denorm)
    fpr_max_frame, tpr_max_frame, _ = roc_curve(labels, max_frame_mses)
    auc_max_frame = auc(fpr_max_frame, tpr_max_frame)
    fpr_top10, tpr_top10, _ = roc_curve(labels, top10_mses)
    auc_top10 = auc(fpr_top10, tpr_top10)
    fpr_latent, tpr_latent, _ = roc_curve(labels, latent_mahas)
    auc_latent = auc(fpr_latent, tpr_latent)
    fpr_combined, tpr_combined, _ = roc_curve(labels, combined_scores)
    auc_combined = auc(fpr_combined, tpr_combined)
    fpr_latent_logreg, tpr_latent_logreg, _ = roc_curve(labels, latent_logreg_scores)
    auc_latent_logreg = auc(fpr_latent_logreg, tpr_latent_logreg)
    fpr_flat_logreg, tpr_flat_logreg, _ = roc_curve(labels, flat_logreg_scores)
    auc_flat_logreg = auc(fpr_flat_logreg, tpr_flat_logreg)
    fpr_latent_svm, tpr_latent_svm, _ = roc_curve(labels, latent_svm_scores)
    auc_latent_svm = auc(fpr_latent_svm, tpr_latent_svm)
    fpr_latent_mlp, tpr_latent_mlp, _ = roc_curve(labels, latent_mlp_scores)
    auc_latent_mlp = auc(fpr_latent_mlp, tpr_latent_mlp)
    fpr_engineered_mlp, tpr_engineered_mlp, _ = roc_curve(labels, engineered_mlp_scores)
    auc_engineered_mlp = auc(fpr_engineered_mlp, tpr_engineered_mlp)
    fpr_meta_mlp, tpr_meta_mlp, _ = roc_curve(labels, latent_meta_mlp_scores)
    auc_meta_mlp = auc(fpr_meta_mlp, tpr_meta_mlp)

    print("\n=== AUC Scores ===")
    print(f"Normalized MSE AUC: {auc_norm:.4f}")
    print(f"Denormalized MSE AUC: {auc_denorm:.4f}")
    print(f"Max Frame MSE AUC: {auc_max_frame:.4f}")
    print(f"Top10% Frame MSE AUC: {auc_top10:.4f}")
    print(f"Latent Mahalanobis AUC: {auc_latent:.4f}")
    print(f"Combined Score AUC: {auc_combined:.4f}")
    print(f"Latent Logistic Regression AUC: {auc_latent_logreg:.4f}")
    print(f"Flat Logistic Regression AUC: {auc_flat_logreg:.4f}")
    print(f"Latent RBF-SVM AUC: {auc_latent_svm:.4f}")
    print(f"Latent MLP AUC: {auc_latent_mlp:.4f}")
    print(f"Engineered-Feature MLP AUC: {auc_engineered_mlp:.4f}")
    print(f"Latent+Engineered Meta-MLP AUC: {auc_meta_mlp:.4f}")

    # Save CSV
    df = pd.DataFrame(
        {
            "model_id": all_scores["model_id"],
            "label": all_scores["label"],
            "normalized_mse": all_scores["normalized_mse"],
            "denormalized_mse": all_scores["denormalized_mse"],
            "per_frame_mse_mean": all_scores["per_frame_mse_mean"],
            "per_frame_mse_max": all_scores["per_frame_mse_max"],
            "per_frame_mse_top10": all_scores["per_frame_mse_top10"],
            "latent_mahalanobis": all_scores["latent_mahalanobis"],
            "combined_score": all_scores["combined_score"],
            "latent_logreg_score": all_scores["latent_logreg_score"],
            "flat_logreg_score": all_scores["flat_logreg_score"],
            "latent_svm_score": all_scores["latent_svm_score"],
            "latent_mlp_score": all_scores["latent_mlp_score"],
            "engineered_mlp_score": all_scores["engineered_mlp_score"],
            "latent_meta_mlp_score": all_scores["latent_meta_mlp_score"],
        }
    )
    csv_path = OUTPUT_DIR / "debug_thresholds.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved scores to: {csv_path}")

    # Plot histograms
    _plot_histograms(
        labels,
        {
            "normalized_mse": normalized_mses,
            "denormalized_mse": denormalized_mses,
            "max_frame_mse": max_frame_mses,
            "top10_frame_mse": top10_mses,
            "latent_mahalanobis": latent_mahas,
            "combined_score": combined_scores,
            "latent_logreg_score": latent_logreg_scores,
            "flat_logreg_score": flat_logreg_scores,
            "latent_svm_score": latent_svm_scores,
            "latent_mlp_score": latent_mlp_scores,
            "engineered_mlp_score": engineered_mlp_scores,
            "latent_meta_mlp_score": latent_meta_mlp_scores,
        },
        OUTPUT_DIR,
    )

    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
