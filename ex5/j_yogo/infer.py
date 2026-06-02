"""infer.py.

Eval/ の全ファイルに異常スコアを付けて CSV 出力する.
"""

import csv
import logging
import re
from pathlib import Path

import hydra
import torch
from audio_processor import process_file
from flow_model import AnomalyDetector
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def extract_model_id(filename: str) -> str:
    """ファイル名から model_id（2桁文字列）を抽出する."""
    m = re.match(r"model_(\d+)_", filename)
    return m.group(1) if m else "00"


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """eval/ の全 wav ファイルに異常スコアと予測ラベルを付けて CSV に出力する."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models, thresholds = {}, {}
    for model_id in cfg.data.target_models:
        ckpt_path = Path(f"model_{model_id}_best.pt")
        thr_path = Path(f"threshold_{model_id}.pt")
        if not ckpt_path.exists():
            continue
        m = AnomalyDetector(
            channels=list(cfg.model.channels),
            emb_dim=cfg.model.emb_dim,
            flow_layers=cfg.model.flow_layers,
            flow_hidden_dim=cfg.model.flow_hidden_dim,
        ).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        m.eval()
        models[model_id] = m
        thresholds[model_id] = (
            torch.load(thr_path)["threshold"] if thr_path.exists() else None
        )

    eval_dir = Path(cfg.data.data_dir) / "eval"
    wav_files = sorted(eval_dir.glob("*.wav"))
    log.info(f"{len(wav_files)} files in {eval_dir}")

    results = []
    with torch.no_grad():
        for wav_path in wav_files:
            mid = extract_model_id(wav_path.name)
            if mid not in models:
                continue
            segs = process_file(
                str(wav_path),
                sr=cfg.data.sr,
                n_fft=cfg.data.n_fft,
                hop_length=cfg.data.hop_length,
                n_mels=cfg.data.n_mels,
                segment_sec=cfg.data.segment_sec,
            )
            batch = torch.stack(segs).to(device)
            score = models[mid].anomaly_score(batch).mean().item()
            thr = thresholds[mid]
            pred = int(score > thr) if thr is not None else -1
            results.append(
                {
                    "filename": wav_path.name,
                    "model_id": mid,
                    "score": f"{score:.6f}",
                    "threshold": f"{thr:.6f}" if thr else "N/A",
                    "prediction": pred,
                }
            )

    out_path = Path("predictions.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "model_id", "score", "threshold", "prediction"]
        )
        writer.writeheader()
        writer.writerows(results)
    log.info(f"Saved {out_path} ({len(results)} rows)")


if __name__ == "__main__":
    main()
