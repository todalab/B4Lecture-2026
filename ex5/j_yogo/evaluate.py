"""
evaluate.py
学習済みモデルの最終評価・閾値決定.
閾値はdevデータでF1を最大化するように最適化する.
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from dataset import EvalDataset
from flow_model import AnomalyDetector
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def find_best_threshold(all_scores: list, all_labels: list) -> tuple:
    """
    ROC曲線の全閾値候補からF1を最大化する閾値を探す.

    Returns:
        best_threshold: float
        best_f1       : float
    """
    _, _, thresholds = roc_curve(all_labels, all_scores)

    best_f1 = 0.0
    best_threshold = thresholds[0]

    for threshold in thresholds:
        preds = [1 if s > threshold else 0 for s in all_scores]
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold), float(best_f1)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """学習済みモデルを評価し、最適閾値を threshold_XX.pt に保存する."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_aucs = {}
    all_f1s = {}

    for model_id in cfg.data.target_models:
        ckpt_path = Path(f"model_{model_id}_best.pt")
        if not ckpt_path.exists():
            log.warning(f"model_{model_id}: checkpoint not found, skipping")
            continue

        model = AnomalyDetector(
            channels=list(cfg.model.channels),
            emb_dim=cfg.model.emb_dim,
            flow_layers=cfg.model.flow_layers,
            flow_hidden_dim=cfg.model.flow_hidden_dim,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        eval_ds = EvalDataset(cfg.data, model_id)
        loader = DataLoader(eval_ds, batch_size=32, shuffle=False)

        all_scores, all_labels = [], []
        with torch.no_grad():
            for x, labels in loader:
                scores = model.anomaly_score(x.to(device))
                all_scores.extend(scores.cpu().tolist())
                all_labels.extend(labels.tolist())

        if len(set(all_labels)) < 2:
            log.warning(f"model_{model_id}: ラベルが1種類のみ、スキップ")
            continue

        normal_scores = [s for s, lbl in zip(all_scores, all_labels) if lbl == 0]
        anomaly_scores = [s for s, lbl in zip(all_scores, all_labels) if lbl == 1]

        auc = roc_auc_score(all_labels, all_scores)
        best_threshold, best_f1 = find_best_threshold(all_scores, all_labels)

        preds = [1 if s > best_threshold else 0 for s in all_scores]
        cm = confusion_matrix(all_labels, preds)

        log.info(f"\nmodel_{model_id}")
        log.info(f"  AUC            : {auc:.4f}")
        log.info(f"  F1  (best)     : {best_f1:.4f}")
        log.info(f"  threshold      : {best_threshold:.4f}")
        log.info(
            f"  normal_score   mean={np.mean(normal_scores):.4f}"
            f" std={np.std(normal_scores):.4f}"
        )
        log.info(f"  anomaly_score  mean={np.mean(anomaly_scores):.4f}")
        log.info(f"  confusion matrix:\n{cm}")

        torch.save(
            {"threshold": best_threshold, "model_id": model_id},
            f"threshold_{model_id}.pt",
        )
        all_aucs[model_id] = auc
        all_f1s[model_id] = best_f1

    # 調和平均を計算
    log.info("\n" + "=" * 40)
    log.info("Summary:")
    for mid in all_aucs:
        log.info(f"  model_{mid}: AUC {all_aucs[mid]:.4f}  F1 {all_f1s[mid]:.4f}")
    if all_aucs:
        mean_auc = np.mean(list(all_aucs.values()))
        # F1の調和平均
        harmonic_f1 = len(all_f1s) / sum(1 / f1 for f1 in all_f1s.values() if f1 > 0)
        log.info(f"  Mean AUC     : {mean_auc:.4f}")
        log.info(f"  Harmonic F1  : {harmonic_f1:.4f}")


if __name__ == "__main__":
    main()
