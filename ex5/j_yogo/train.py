"""
train.py
model_id ごとに独立した AnomalyDetector（CNN Encoder + Real-NVP）を学習する.
学習: abundant/model_XX_normal のみ
評価: dev/model_XX_normal + dev/model_XX_anomaly
"""

import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import hydra
from omegaconf import DictConfig

from flow_model import AnomalyDetector
from dataset import NormalDataset, EvalDataset

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Python / NumPy / PyTorch の乱数シードを固定して再現性を確保する."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: AnomalyDetector, data_cfg, model_id: str,
             device: torch.device) -> tuple:
    """
    Returns:
        auc         : float
        normal_scores : list of float
        anomaly_scores: list of float
    """
    eval_ds = EvalDataset(data_cfg, model_id)
    if len(eval_ds) == 0:
        return 0.0, [], []

    loader = DataLoader(eval_ds, batch_size=32, shuffle=False)
    model.eval()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, labels in loader:
            scores = model.anomaly_score(x.to(device))
            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(labels.tolist())

    model.train()

    normal_scores  = [s for s, l in zip(all_scores, all_labels) if l == 0]
    anomaly_scores = [s for s, l in zip(all_scores, all_labels) if l == 1]
    auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) == 2 else 0.0

    return auc, normal_scores, anomaly_scores


def train_one_model(model_id: str, cfg: DictConfig,
                    device: torch.device, writer: SummaryWriter) -> float:
    """
    指定した model_id の AnomalyDetector を学習し、最良の AUC を返す.

    Returns:
        best_auc: float（データなしの場合は None）
    """
    log.info(f"\n{'='*50}")
    log.info(f"Training model_{model_id}")

    train_ds = NormalDataset(cfg.data, model_id)
    if len(train_ds) == 0:
        log.warning(f"model_{model_id}: no training data, skipping")
        return None
    log.info(f"  {len(train_ds)} train samples")

    loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=len(train_ds) > cfg.train.batch_size,
    )

    model = AnomalyDetector(
        channels=list(cfg.model.channels),
        emb_dim=cfg.model.emb_dim,
        flow_layers=cfg.model.flow_layers,
        flow_hidden_dim=cfg.model.flow_hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )

    best_auc    = 0.0
    global_step = 0

    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            x    = batch.to(device)
            loss = model.loss(x)  # -log_likelihood

            optimizer.zero_grad()
            loss.backward()
            # 勾配クリッピング（Flowは勾配が不安定になりやすい）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1
            writer.add_scalar(f"loss_step/model_{model_id}", loss.item(), global_step)

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        writer.add_scalar(f"loss_epoch/model_{model_id}", avg_loss, epoch)
        writer.add_scalar(f"lr/model_{model_id}", scheduler.get_last_lr()[0], epoch)

        if (epoch + 1) % cfg.train.val_interval == 0:
            auc, normal_scores, anomaly_scores = evaluate(
                model, cfg.data, model_id, device
            )
            writer.add_scalar(f"auc/model_{model_id}", auc, epoch)

            if normal_scores:
                writer.add_histogram(f"anomaly_score/model_{model_id}_normal",
                                     torch.tensor(normal_scores), epoch)
                writer.add_scalar(f"score_mean/model_{model_id}_normal",
                                  np.mean(normal_scores), epoch)
            if anomaly_scores:
                writer.add_histogram(f"anomaly_score/model_{model_id}_anomaly",
                                     torch.tensor(anomaly_scores), epoch)
                writer.add_scalar(f"score_mean/model_{model_id}_anomaly",
                                  np.mean(anomaly_scores), epoch)

            log.info(
                f"  epoch {epoch+1:3d} | loss {avg_loss:.4f} | AUC {auc:.4f} | "
                f"normal_score {np.mean(normal_scores):.4f} | "
                f"anomaly_score {np.mean(anomaly_scores):.4f}"
            )
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), f"model_{model_id}_best.pt")
                log.info(f"  → saved best (AUC {best_auc:.4f})")
        else:
            log.info(f"  epoch {epoch+1:3d} | loss {avg_loss:.4f}")

    log.info(f"model_{model_id} done. Best AUC: {best_auc:.4f}")
    return best_auc


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """全 model_id の学習を実行し、最終 AUC をログ出力する."""
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")

    writer  = SummaryWriter(log_dir="runs")
    results = {}

    for model_id in cfg.data.target_models:
        auc = train_one_model(model_id, cfg, device, writer)
        if auc is not None:
            results[model_id] = auc

    writer.close()

    log.info("\n" + "="*50)
    log.info("Final Results:")
    for mid, auc in results.items():
        log.info(f"  model_{mid}: AUC {auc:.4f}")
    if results:
        log.info(f"  Mean AUC : {np.mean(list(results.values())):.4f}")


if __name__ == "__main__":
    main()