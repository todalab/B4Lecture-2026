"""Training loop for mel-spectrogram autoencoder."""

import logging
from pathlib import Path
from typing import Sized

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataloaders.dataloader import create_dataloader
from models.autoencoder import Autoencoder
from utils.seed import set_seed
from utils.visualize import plot_recon_pair

# Set up logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> float:
    """Compute average reconstruction loss over a data loader.

    Parameters
    ----------
    model : nn.Module
        Autoencoder model to evaluate.
    loader : DataLoader
        DataLoader yielding (mel, digit, name) batches.
    device : torch.device
        Device to run the model on.
    loss_fn : nn.Module
        Loss function used for reconstruction.

    Returns
    -------
    loss : float
        Average loss across all samples.
    """
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for mels, _, _ in loader:
            mels = mels.to(device)
            recon = model(mels)
            loss = loss_fn(recon, mels)
            total_loss += loss.item() * mels.size(0)
            total_count += mels.size(0)
    return total_loss / total_count if total_count > 0 else 0.0


def train(cfg: DictConfig) -> None:
    """Run the training and validation loop from a Hydra config.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with model, dataset, and training settings.
    """
    # Set random seed for reproducibility
    set_seed(int(cfg.train.seed))

    # Create data loaders for training and validation datasets
    train_loader = create_dataloader(
        Path(to_absolute_path(str(cfg.dataset.train_list))),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        n_mels=cfg.dataset.n_mels,
        target_frames=cfg.dataset.target_frames,
        seed=int(cfg.train.seed),
    )
    val_loader = create_dataloader(
        Path(to_absolute_path(str(cfg.dataset.val_list))),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.dataset.n_fft,
        hop_length=cfg.dataset.hop_length,
        n_mels=cfg.dataset.n_mels,
        target_frames=cfg.dataset.target_frames,
        seed=int(cfg.train.seed),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer, and loss function
    model = Autoencoder(
        in_channels=1,
        hidden_channels=cfg.model.hidden_channels,
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

    try:
        for epoch in tqdm(range(cfg.train.epochs)):
            model.train()
            running_loss = 0.0
            # Training loop over batches
            for mels, _, _ in train_loader:
                mels = mels.to(device)

                optimizer.zero_grad()  # 勾配をリセット
                recon = model(mels)  # 順伝播で再構成（エンコーダ・デコーダを通した）メルスペクトログラムを得る
                loss = loss_fn(recon, mels)  # 損失を計算する
                loss.backward()  # 誤差逆伝播
                optimizer.step()  # パラメータを更新
                running_loss += loss.item() * mels.size(0)

            # Log average training loss for the epoch
            dataset = train_loader.dataset
            assert isinstance(dataset, Sized), "Dataset must implement __len__"
            avg_loss = running_loss / len(dataset)
            writer.add_scalar("loss/train", avg_loss, epoch)

            # Run validation and log scalars/images for TensorBoard
            if epoch % cfg.train.epoch_per_val == 0:
                val_loss = evaluate(model, val_loader, device, loss_fn)
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
