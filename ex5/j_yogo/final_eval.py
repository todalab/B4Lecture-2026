"""eval_final.py.

Eval-groundtruth-labeled/ を使った最終評価.
手法・モデル・閾値は変更しない.
evaluate.py で決定した threshold_XX.pt をそのまま使う.
"""

import logging
import re
from pathlib import Path

import hydra
import numpy as np
import torch
from audio_processor import process_file
from flow_model import AnomalyDetector
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


def extract_model_id(filename: str) -> str:
    """ファイル名から model_id（2桁文字列）を抽出する."""
    m = re.match(r"model_(\d+)_", filename)
    return m.group(1) if m else "00"


class EvalGroundTruthDataset(Dataset):
    """Eval-groundtruth-labeled/ のデータセット."""

    def __init__(self, data_cfg, model_id: str):
        """データ設定と model_id を受け取り初期化する."""
        self.cfg = data_cfg
        self.samples = self._collect(model_id)

    def _collect(self, model_id: str) -> list:
        """eval-groundtruth-labeled/ から (tensor, label) リストを収集する."""
        gt_dir = Path(self.cfg.data_dir) / "eval-groundtruth-labeled"
        samples = []
        for p in sorted(gt_dir.glob(f"model_{model_id}_*.wav")):
            label = 1 if "_anomaly_" in p.name else 0
            for t in process_file(
                str(p),
                sr=self.cfg.sr,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                n_mels=self.cfg.n_mels,
                segment_sec=self.cfg.segment_sec,
            ):
                samples.append((t, label))
        return samples

    def __len__(self) -> int:
        """データセットのサンプル数を返す."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """指定インデックスの (tensor, label) を返す."""
        return self.samples[idx]


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """eval-groundtruth-labeled/ を用いて最終的な AUC / F1 を評価する."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_aucs = {}
    all_f1s = {}

    for model_id in cfg.data.target_models:
        ckpt_path = Path(f"model_{model_id}_best.pt")
        thr_path = Path(f"threshold_{model_id}.pt")

        if not ckpt_path.exists():
            log.warning(f"model_{model_id}: checkpoint not found, skipping")
            continue
        if not thr_path.exists():
            log.warning(f"model_{model_id}: threshold not found, run evaluate.py first")
            continue

        # モデルロード
        model = AnomalyDetector(
            channels=list(cfg.model.channels),
            emb_dim=cfg.model.emb_dim,
            flow_layers=cfg.model.flow_layers,
            flow_hidden_dim=cfg.model.flow_hidden_dim,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # 閾値ロード（evaluate.py で決定したものをそのまま使う）
        threshold = torch.load(thr_path)["threshold"]

        # eval-groundtruth-labeled/ を評価
        eval_ds = EvalGroundTruthDataset(cfg.data, model_id)
        if len(eval_ds) == 0:
            log.warning(f"model_{model_id}: no eval data found")
            continue

        loader = DataLoader(eval_ds, batch_size=32, shuffle=False)

        all_scores, all_labels = [], []
        with torch.no_grad():
            for x, labels in loader:
                scores = model.anomaly_score(x.to(device))
                all_scores.extend(scores.cpu().tolist())
                all_labels.extend(labels.tolist())

        if len(set(all_labels)) < 2:
            log.warning(f"model_{model_id}: only one label type found")
            continue

        normal_scores = [s for s, lbl in zip(all_scores, all_labels) if lbl == 0]
        anomaly_scores = [s for s, lbl in zip(all_scores, all_labels) if lbl == 1]

        auc = roc_auc_score(all_labels, all_scores)
        preds = [1 if s > threshold else 0 for s in all_scores]
        f1 = f1_score(all_labels, preds, zero_division=0)
        cm = confusion_matrix(all_labels, preds)

        log.info(f"\nmodel_{model_id}")
        log.info(f"  AUC            : {auc:.4f}")
        log.info(f"  F1             : {f1:.4f}")
        log.info(f"  threshold      : {threshold:.4f}")
        log.info(
            f"  normal_score   mean={np.mean(normal_scores):.4f}"
            f" std={np.std(normal_scores):.4f}"
        )
        log.info(f"  anomaly_score  mean={np.mean(anomaly_scores):.4f}")
        log.info(f"  confusion matrix:\n{cm}")

        all_aucs[model_id] = auc
        all_f1s[model_id] = f1

    # 最終指標
    log.info("\n" + "=" * 40)
    log.info("Final Results (eval-groundtruth-labeled):")
    for mid in all_aucs:
        log.info(f"  model_{mid}: AUC {all_aucs[mid]:.4f}  F1 {all_f1s[mid]:.4f}")
    if all_f1s:
        mean_auc = np.mean(list(all_aucs.values()))
        valid_f1s = [f1 for f1 in all_f1s.values() if f1 > 0]
        harmonic_f1 = (
            len(valid_f1s) / sum(1 / f1 for f1 in valid_f1s) if valid_f1s else 0.0
        )
        log.info(f"  Mean AUC     : {mean_auc:.4f}")
        log.info(f"  Harmonic F1  : {harmonic_f1:.4f}  ← 最終指標")


if __name__ == "__main__":
    main()
