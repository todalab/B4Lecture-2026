"""
Training Utilities for Ex6 B4 Lecture - 完成版
適切なロギングと学習ループを提供
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time
import math
from tqdm import tqdm
import os
import json
from typing import Dict, Tuple, Optional

# ロギング設定
def setup_logging(log_file: str = "training.log", level=logging.INFO):
    """ロギングを設定"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


class LearningRateScheduler:
    """学習率スケジューラー（Warmup + Cosine Decay）"""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.step_count = 0

    def step(self):
        """学習率を更新"""
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine decay phase
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


class TrainingMetrics:
    """学習メトリクスを記録"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.val_perplexities = []
        self.learning_rates = []
        self.epoch_times = []

    def add_epoch(self, train_loss: float, val_loss: float, val_perplexity: float,
                  lr: float, epoch_time: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_perplexities.append(val_perplexity)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)

    def save(self, filepath: str):
        """メトリクスを保存"""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_perplexities': self.val_perplexities,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times
        }
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {filepath}")


def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                scheduler: Optional[LearningRateScheduler] = None,
                grad_accumulation_steps: int = 1) -> Tuple[float, float]:
    """1エポックの学習"""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, (x, y) in enumerate(progress_bar):
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits, loss = model(x, y)

        # Gradient accumulation
        loss = loss / grad_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % grad_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            if scheduler:
                current_lr = scheduler.step()
            else:
                current_lr = optimizer.param_groups[0]['lr']

        total_loss += loss.item() * grad_accumulation_steps
        num_batches += 1

        # プログレスバーの更新
        if batch_idx % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item() * grad_accumulation_steps:.3f}',
                'lr': f'{current_lr:.2e}'
            })

    avg_loss = total_loss / num_batches
    current_lr = optimizer.param_groups[0]['lr']

    logger.info(f"Training - Average Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

    return avg_loss, current_lr


def evaluate(model: nn.Module,
             val_loader: DataLoader,
             device: torch.device) -> Tuple[float, float]:
    """評価"""
    model.eval()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    logger.info(f"Validation - Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")

    return avg_loss, perplexity


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int,
                learning_rate: float,
                device: torch.device,
                save_dir: str = "checkpoints",
                model_name: str = "model",
                warmup_steps: int = 1000,
                grad_accumulation_steps: int = 1) -> TrainingMetrics:
    """モデル学習のメイン関数"""

    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Model: {model_name}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Device: {device}")

    # ディレクトリ作成
    os.makedirs(save_dir, exist_ok=True)

    # オプティマイザー
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # スケジューラー
    total_steps = len(train_loader) * epochs // grad_accumulation_steps
    scheduler = LearningRateScheduler(optimizer, warmup_steps, total_steps, learning_rate)

    # メトリクス記録
    metrics = TrainingMetrics()

    # 最良モデル保存用
    best_val_loss = float('inf')

    # 学習開始時間
    start_time = time.time()

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_start_time = time.time()

        # 学習
        train_loss, current_lr = train_epoch(
            model, train_loader, optimizer, device, scheduler, grad_accumulation_steps
        )

        # 評価
        val_loss, val_perplexity = evaluate(model, val_loader, device)

        # メトリクス記録
        epoch_time = time.time() - epoch_start_time
        metrics.add_epoch(train_loss, val_loss, val_perplexity, current_lr, epoch_time)

        # 結果表示
        logger.info(f"Epoch {epoch + 1} Results:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val Perplexity: {val_perplexity:.2f}")
        logger.info(f"  Learning Rate: {current_lr:.2e}")
        logger.info(f"  Epoch Time: {epoch_time:.1f}s")

        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_perplexity': val_perplexity,
                'train_loss': train_loss,
                'learning_rate': current_lr
            }

            model_path = os.path.join(save_dir, f"{model_name}_best.pt")
            torch.save(checkpoint, model_path)
            logger.info(f"New best model saved: {model_path}")

        # 定期的なチェックポイント保存
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pt")
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    # 総学習時間
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time:.1f} seconds")

    # メトリクス保存
    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    metrics.save(metrics_path)

    return metrics


def count_parameters(model: nn.Module) -> int:
    """モデルのパラメータ数をカウント"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    return device


def print_gpu_memory():
    """GPU メモリ使用量を表示"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


if __name__ == "__main__":
    # テスト用のコード
    print("=== Training Utilities Test ===")

    # デバイステスト
    device = get_device()
    print(f"Device: {device}")

    # スケジューラーテスト
    dummy_optimizer = optim.Adam([torch.randn(10, requires_grad=True)], lr=1e-3)
    scheduler = LearningRateScheduler(dummy_optimizer, 100, 1000, 1e-3)

    print("Learning rate schedule test:")
    for step in [50, 100, 500, 1000]:
        scheduler.step_count = step
        lr = scheduler.step()
        print(f"  Step {step}: LR = {lr:.2e}")

    # メトリクステスト
    metrics = TrainingMetrics()
    metrics.add_epoch(2.5, 2.8, 16.4, 1e-4, 120.0)
    metrics.add_epoch(2.1, 2.5, 12.2, 8e-5, 118.0)
    print(f"Metrics test: {len(metrics.train_losses)} epochs recorded")

    print("Training utilities test completed!")