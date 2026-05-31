"""Training loop for mel-spectrogram autoencoder."""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataloaders.dataloader import MelSpectrogramDataset, create_dataloader
from models.autoencoder import Autoencoder
from utils.seed import set_seed
from utils.visualize import plot_recon_pair

# Set up logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Register a custom resolver to round values for better readability in subdirectory names
OmegaConf.register_new_resolver("round", round)


def _resolve_list_path(path_like: str | Path) -> Path:
    """Resolve a configured file list path and add .txt when the suffix is omitted."""
    raw_path = Path(str(path_like))
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [raw_path, Path.cwd() / raw_path, Path.cwd().parent / raw_path, repo_root / raw_path]

    for candidate in candidates:
        resolved_candidate = candidate if candidate.suffix else candidate.with_suffix(".txt")
        if resolved_candidate.exists():
            return resolved_candidate.resolve()

    fallback = raw_path if raw_path.suffix else raw_path.with_suffix(".txt")
    return fallback.resolve()


def _resolve_data_dir_path(path_like: str | Path | None) -> Path | None:
    """Resolve a configured data directory path if one is provided."""
    if path_like is None:
        return None

    raw_path = Path(str(path_like))
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [raw_path, Path.cwd() / raw_path, Path.cwd().parent / raw_path, repo_root / raw_path]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (repo_root / raw_path).resolve()


def _split_normal_and_anomaly_paths(
    file_list_path: str | Path, data_dir_path: str | Path | None
) -> tuple[list[Path], list[Path]]:
    """Split a mixed file list into normal and anomalous file paths."""
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

    if not normal_files:
        raise ValueError(f"No normal files were found in {file_list_path}")
    if not anomaly_files:
        raise ValueError(f"No anomalous files were found in {file_list_path}")

    return normal_files, anomaly_files


def _infinite_loader(loader: DataLoader):
    """Yield batches from a loader forever by restarting it when exhausted."""
    while True:
        for batch in loader:
            yield batch


def _paired_loss(
    model: nn.Module,
    normal_mels: torch.Tensor,
    anomaly_mels: torch.Tensor,
    loss_fn: nn.Module,
    margin: float,
    anomaly_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute normal loss, anomalous hinge loss, and total loss for one paired batch."""
    normal_recon = model(normal_mels)
    anomaly_recon = model(anomaly_mels)

    normal_loss = loss_fn(normal_recon, normal_mels)
    anomaly_mse = loss_fn(anomaly_recon, anomaly_mels)
    anomaly_loss = torch.relu(torch.tensor(margin, device=anomaly_mse.device) - anomaly_mse)
    total_loss = normal_loss + anomaly_lambda * anomaly_loss
    return total_loss, normal_loss, anomaly_loss, normal_recon, anomaly_recon


def evaluate(
    model: nn.Module,
    normal_loader: DataLoader,
    anomaly_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    margin: float,
    anomaly_lambda: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Compute average paired validation loss over normal and anomalous loaders.

    Parameters
    ----------
    model : nn.Module
        Autoencoder model to evaluate.
    normal_loader : DataLoader
        DataLoader yielding normal mel batches.
    anomaly_loader : DataLoader
        DataLoader yielding anomalous mel batches.
    device : torch.device
        Device to run the model on.
    loss_fn : nn.Module
        Loss function used for reconstruction.
    margin : float
        Hinge margin for anomalous samples.
    anomaly_lambda : float
        Weight applied to the anomalous hinge loss.

    Returns
    -------
    Tuple

    1. total_loss : float
        Average total loss across paired batches.
    2. normal_loss : float
        Average reconstruction loss for normal batches.
    3. anomaly_loss : float
        Average hinge loss for anomalous batches.
    4. normal_errors : np.ndarray
        各正常入力音声の再構成誤差.
    5. anomaly_errors : np.ndarray
        各異常入力音声の再構成誤差.
    """
    model.eval()
    total_loss = 0.0
    normal_total_loss = 0.0
    anomaly_total_loss = 0.0
    normal_errors: list[float] = []
    anomaly_errors: list[float] = []
    steps = max(len(normal_loader), len(anomaly_loader))
    normal_iter = _infinite_loader(normal_loader)
    anomaly_iter = _infinite_loader(anomaly_loader)

    with torch.no_grad():  # 学習なしで計算
        for _ in range(steps):
            normal_mels, _, _ = next(normal_iter)
            anomaly_mels, _, _ = next(anomaly_iter)

            normal_mels = normal_mels.to(device)
            anomaly_mels = anomaly_mels.to(device)

            _, normal_loss, anomaly_loss, _, _ = _paired_loss(
                model,
                normal_mels,
                anomaly_mels,
                loss_fn,
                margin,
                anomaly_lambda,
            )
            total_batch_loss = normal_loss + anomaly_lambda * anomaly_loss

            normal_total_loss += normal_loss.item()
            anomaly_total_loss += anomaly_loss.item()
            total_loss += total_batch_loss.item()
            normal_errors.append(normal_loss.item())
            anomaly_errors.append(anomaly_loss.item())

    if steps <= 0:
        return 0.0, 0.0, 0.0, np.array([]), np.array([])

    return (
        total_loss / steps,
        normal_total_loss / steps,
        anomaly_total_loss / steps,
        np.asarray(normal_errors),
        np.asarray(anomaly_errors),
    )


def train(cfg: DictConfig) -> float:
    """Run the training and validation loop from a Hydra config.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with model, dataset, and training settings.

    Returns
    -------
    best_val_loss : float
        Best validation loss observed during training.
    """
    # Set random seed for reproducibility
    set_seed(int(cfg.train.seed))

    # Determine device to use for training (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_list_path = _resolve_list_path(cfg.dataset_normal_and_anomaly.train_list)
    val_list_path = _resolve_list_path(cfg.dataset_normal_and_anomaly.eval_list)
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

    train_normal_paths, train_anomaly_paths = _split_normal_and_anomaly_paths(train_list_path, data_dir_path)
    val_normal_paths, val_anomaly_paths = _split_normal_and_anomaly_paths(val_list_path, data_dir_path)

    # Create paired data loaders for normal and anomalous datasets
    normal_train_loader = create_dataloader(
        train_normal_paths,
        data_dir_path,
        batch_size=cfg.train.batch_size,
        shuffle=True,
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
    anomaly_train_loader = create_dataloader(
        train_anomaly_paths,
        data_dir_path,
        batch_size=cfg.train.batch_size,
        shuffle=True,
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
    normal_val_loader = create_dataloader(
        val_normal_paths,
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
    anomaly_val_loader = create_dataloader(
        val_anomaly_paths,
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
    print("GPU available: ", torch.cuda.is_available())

    margin = float(cfg.train["margin"]) if "margin" in cfg.train else 0.02
    anomaly_lambda = float(cfg.train["lambda"]) if "lambda" in cfg.train else 1.0

    # Initialize model, optimizer, and loss function
    model = Autoencoder(
        in_channels=1,
        hidden_channels1=cfg.model.hidden_channels1,
        hidden_channels2=cfg.model.hidden_channels2,
        latent_channels=cfg.model.latent_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    loss_fn = nn.MSELoss()

    # Set up logging and checkpoint directories
    log_dir = Path(str(cfg.train.log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(str(cfg.train.ckpt_dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer for logging scalars and figures
    writer = SummaryWriter(log_dir=log_dir)

    # Track the best validation loss for reporting to Optuna
    best_val_loss = float("inf")

    try:
        for epoch in tqdm(range(cfg.train.epochs)):
            model.train()
            running_total_loss = 0.0
            running_normal_loss = 0.0
            running_anomaly_loss = 0.0
            train_steps = max(len(normal_train_loader), len(anomaly_train_loader))
            normal_iter = _infinite_loader(normal_train_loader)
            anomaly_iter = _infinite_loader(anomaly_train_loader)

            # Training loop over paired normal/anomalous batches
            for _ in range(train_steps):
                normal_mels, _, _ = next(normal_iter)
                anomaly_mels, _, _ = next(anomaly_iter)

                normal_mels = normal_mels.to(device)
                anomaly_mels = anomaly_mels.to(device)

                optimizer.zero_grad()
                total_loss, normal_loss, anomaly_loss, _, _ = _paired_loss(
                    model,
                    normal_mels,
                    anomaly_mels,
                    loss_fn,
                    margin,
                    anomaly_lambda,
                )
                total_loss.backward()
                optimizer.step()
                running_total_loss += total_loss.item()
                running_normal_loss += normal_loss.item()
                running_anomaly_loss += anomaly_loss.item()

            # Log average training loss for the epoch
            avg_total_loss = running_total_loss / max(train_steps, 1)
            avg_normal_loss = running_normal_loss / max(train_steps, 1)
            avg_anomaly_loss = running_anomaly_loss / max(train_steps, 1)
            writer.add_scalar("loss/train_total", avg_total_loss, epoch)
            writer.add_scalar("loss/train_normal", avg_normal_loss, epoch)
            writer.add_scalar("loss/train_anomaly", avg_anomaly_loss, epoch)

            # Run validation and log scalars/images for TensorBoard
            if (
                epoch % cfg.train.epoch_per_val == 0
            ):  # configファイルでcfg.train.epoch_per_val=1となっているので毎回実行
                val_total_loss, val_normal_loss, val_anomaly_loss, normal_errors, anomaly_errors = evaluate(
                    model,
                    normal_val_loader,
                    anomaly_val_loader,
                    device,
                    loss_fn,
                    margin,
                    anomaly_lambda,
                )
                best_val_loss = min(best_val_loss, val_total_loss)  # Update best validation loss
                writer.add_scalar("loss/val_total", val_total_loss, epoch)
                writer.add_scalar("loss/val_normal", val_normal_loss, epoch)
                writer.add_scalar("loss/val_anomaly", val_anomaly_loss, epoch)

                # Reconstruct normal and anomalous examples for visualization
                with torch.no_grad():
                    normal_batch, _, _ = next(iter(normal_val_loader))
                    anomaly_batch, _, _ = next(iter(anomaly_val_loader))
                    normal_batch = normal_batch.to(device)
                    anomaly_batch = anomaly_batch.to(device)
                    normal_recon = model(normal_batch)
                    anomaly_recon = model(anomaly_batch)
                normal_batch_cpu = normal_batch.cpu()
                normal_recon_cpu = normal_recon.cpu()
                anomaly_batch_cpu = anomaly_batch.cpu()
                anomaly_recon_cpu = anomaly_recon.cpu()

                # Visualize few examples of reconstructions
                display_count = min(3, normal_batch_cpu.size(0), anomaly_batch_cpu.size(0))
                for idx in range(display_count):
                    fig = plot_recon_pair(normal_batch_cpu[idx, 0], normal_recon_cpu[idx, 0])
                    writer.add_figure(f"recon/val_normal_{idx}", fig, epoch)
                    fig.clf()
                    fig = plot_recon_pair(anomaly_batch_cpu[idx, 0], anomaly_recon_cpu[idx, 0])
                    writer.add_figure(f"recon/val_anomaly_{idx}", fig, epoch)
                    fig.clf()

                logger.info(
                    f"Epoch {epoch:03d} | train_total {avg_total_loss:.4f} | train_normal {avg_normal_loss:.4f} | "
                    f"train_anomaly {avg_anomaly_loss:.4f} | val_total {val_total_loss:.4f}"
                )
            if epoch % cfg.train.epoch_per_ckpt == 0:
                # Save checkpoints
                ckpt_path = ckpt_dir / f"model_epoch_{epoch:04d}.pt"
                torch.save(model.state_dict(), ckpt_path)
    finally:
        writer.close()

    # Report the best validation loss to Optuna for hyperparameter optimization
    return best_val_loss


@hydra.main(version_base=None, config_path="../configs", config_name="config_optuna_for_comp")
def main(cfg: DictConfig) -> float:
    return train(cfg)


if __name__ == "__main__":
    main()
