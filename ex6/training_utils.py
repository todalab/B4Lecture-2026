"""
Training Utilities
"""

import json
import logging
import math
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def setup_logging(log_file: str = "logs/training.log", level=logging.INFO):
    """ロギングを設定"""
    # ログは logs/ などのディレクトリにまとめる。無ければ作成する
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


class LearningRateScheduler:
    """Warmup + Cosine Decay スケジューラー"""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class TrainingMetrics:
    """学習メトリクスの記録"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.train_losses: list = []
        self.val_losses: list = []
        self.val_perplexities: list = []
        self.learning_rates: list = []
        self.epoch_times: list = []

    def add_epoch(
        self,
        train_loss: float,
        val_loss: float,
        val_perplexity: float,
        lr: float,
        epoch_time: float,
    ):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_perplexities.append(val_perplexity)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)

    def save(self, filepath: str):
        metrics = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_perplexities": self.val_perplexities,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
        }
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {filepath}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: Optional[LearningRateScheduler] = None,
    grad_accumulation_steps: int = 1,
) -> Tuple[float, float]:
    """1 エポックの学習

    バッチ形式: (src, tgt_in, tgt_out)
        src:     (batch, src_len)  エンコーダ入力
        tgt_in:  (batch, tgt_len)  デコーダ入力 (BOS から始まる)
        tgt_out: (batch, tgt_len)  デコーダ正解 (EOS で終わる)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, (src, tgt_in, tgt_out) in enumerate(progress_bar):
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        logits, loss = model(src, tgt_in, targets=tgt_out)

        loss = loss / grad_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * grad_accumulation_steps
        num_batches += 1

        if batch_idx % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item() * grad_accumulation_steps:.3f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

    avg_loss = total_loss / num_batches
    current_lr = optimizer.param_groups[0]["lr"]
    logger.info(f"Train - Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
    return avg_loss, current_lr


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """検証データで評価する

    Returns:
        avg_loss, perplexity
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for src, tgt_in, tgt_out in progress_bar:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            _, loss = model(src, tgt_in, targets=tgt_out)
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.3f}"})

    avg_loss = total_loss / num_batches
    perplexity = math.exp(min(avg_loss, 20))  # overflow 防止

    logger.info(f"Val   - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    return avg_loss, perplexity


def _show_sample_translations(
    model,
    sample_sentences,
    src_tokenizer,
    tgt_tokenizer,
    device,
    max_len,
    bos_idx,
    eos_idx,
):
    """エポック終了時にサンプル翻訳を表示する"""
    model.eval()
    with torch.no_grad():
        for sent in sample_sentences:
            src_ids = src_tokenizer.encode(sent, add_eos=True, max_len=max_len)
            src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
            generated = model.generate(src_tensor, bos_idx, eos_idx, max_len=max_len)
            translation = tgt_tokenizer.decode(generated[0].cpu().tolist())
            logger.info(f"  EN: {sent}")
            logger.info(f"  JA: {translation}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    save_dir: str = "checkpoints",
    model_name: str = "model",
    warmup_steps: int = 1000,
    grad_accumulation_steps: int = 1,
    sample_sentences=None,
    src_tokenizer=None,
    tgt_tokenizer=None,
    sample_max_len: int = 64,
) -> TrainingMetrics:
    """モデル学習のメイン関数"""
    logger.info(
        f"Starting training: {epochs} epochs, lr={learning_rate}, device={device}"
    )

    os.makedirs(save_dir, exist_ok=True)

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
    )
    total_steps = len(train_loader) * epochs // grad_accumulation_steps
    scheduler = LearningRateScheduler(
        optimizer, warmup_steps, total_steps, learning_rate
    )
    metrics = TrainingMetrics()
    best_val_loss = float("inf")

    start_time = time.time()
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_start = time.time()

        train_loss, current_lr = train_epoch(
            model, train_loader, optimizer, device, scheduler, grad_accumulation_steps
        )
        val_loss, val_perplexity = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        metrics.add_epoch(train_loss, val_loss, val_perplexity, current_lr, epoch_time)

        logger.info(
            f"Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}, "
            f"ppl={val_perplexity:.2f}, lr={current_lr:.2e}, time={epoch_time:.1f}s"
        )

        if sample_sentences and src_tokenizer and tgt_tokenizer:
            from data_loader import BOS_IDX, EOS_IDX

            logger.info(f"--- Sample translations (epoch {epoch + 1}) ---")
            _show_sample_translations(
                model,
                sample_sentences,
                src_tokenizer,
                tgt_tokenizer,
                device,
                sample_max_len,
                BOS_IDX,
                EOS_IDX,
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(save_dir, f"{model_name}_best.pt")
            torch.save(checkpoint, path)
            logger.info(f"New best model saved: {path}")

        if (epoch + 1) % 5 == 0:
            path = os.path.join(save_dir, f"{model_name}_epoch_{epoch + 1}.pt")
            torch.save(checkpoint, path)

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f}s")

    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    metrics.save(metrics_path)
    return metrics


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CPU")
    return device
