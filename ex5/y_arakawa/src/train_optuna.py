"""Training loop for mel-spectrogram autoencoder."""

import logging
from pathlib import Path
from typing import Sized

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
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


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module
) -> tuple[float, np.ndarray]:
    """Compute average reconstruction loss over a data loader.

    Parameters
    ----------
    model : nn.Module
        Autoencoder model to evaluate.
    loader : DataLoader
        DataLoader yielding (mel, 正常化異常か（0 or 1）, model番号) batches.
    device : torch.device
        Device to run the model on.
    loss_fn : nn.Module
        Loss function used for reconstruction.

    Returns
    -------
    Tuple

    1. loss : float
        Average loss across all samples.
    2. errors : np.ndarray
        各入力音声の再構成誤差.
    """
    model.eval()
    total_loss = 0.0
    total_count = 0
    errors = np.array([])
    with torch.no_grad():  # 学習なしで計算
        for mels, _, _ in loader:
            mels = mels.to(device)
            recon = model(mels)
            loss = loss_fn(recon, mels)
            errors = np.append(errors, loss.item())
            total_loss += loss.item() * mels.size(0)
            total_count += mels.size(0)
    return total_loss / total_count if total_count > 0 else 0.0, errors


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

    train_list_path = Path(to_absolute_path(str(cfg.dataset.train_list)))
    val_list_path = Path(to_absolute_path(str(cfg.dataset.val_list)))
    data_dir_path = Path(to_absolute_path(str(cfg.dataset.data_dir)))
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

    # Create data loaders for training and validation datasets
    train_loader = create_dataloader(
        train_list_path,
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
    print("GPU available: ", torch.cuda.is_available())

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
            running_loss = 0.0
            # Training loop over batches
            for mels, _, _ in train_loader:
                mels = mels.to(device)

                optimizer.zero_grad()
                recon = model(mels)
                loss = loss_fn(recon, mels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * mels.size(0)

            # Log average training loss for the epoch
            dataset = train_loader.dataset
            assert isinstance(dataset, Sized), "Dataset must implement __len__"
            avg_loss = running_loss / len(dataset)
            writer.add_scalar("loss/train", avg_loss, epoch)

            # Run validation and log scalars/images for TensorBoard
            if (
                epoch % cfg.train.epoch_per_val == 0
            ):  # configファイルでcfg.train.epoch_per_val=1となっているので毎回実行
                val_loss, errors = evaluate(model, val_loader, device, loss_fn)
                best_val_loss = min(best_val_loss, val_loss)  # Update best validation loss
                writer.add_scalar("loss/val", val_loss, epoch)

                # Reconstruct for visualization
                with torch.no_grad():
                    val_batch, _, _ = next(iter(val_loader))
                    val_batch = val_batch.to(device)
                    recon = model(val_batch)
                val_batch_cpu = val_batch.cpu()
                recon_cpu = recon.cpu()

                # Visualize few examples of reconstructions
                display_count = min(3, val_batch_cpu.size(0))
                for idx in range(display_count):
                    fig = plot_recon_pair(val_batch_cpu[idx, 0], recon_cpu[idx, 0])
                    writer.add_figure(f"recon/val_{idx}", fig, epoch)
                    fig.clf()

                logger.info(f"Epoch {epoch:03d} | loss {avg_loss:.4f} | val {val_loss:.4f}")
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
