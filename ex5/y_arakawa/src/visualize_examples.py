"""Visualize representative normal/anomaly reconstructions and save audio."""

from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
import torch
import yaml

from dataloaders.dataloader import MelSpectrogramDataset
from models.autoencoder import Autoencoder
from utils.visualize import plot_recon_pair

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "outputs/debug_thresholds/debug_thresholds.csv"
CONFIG_PATH = BASE_DIR / "configs/config_optuna_for_comp.yaml"
OPTUNA_RESULTS_PATH = BASE_DIR / "logs/mel_ae_optuna_for_comp/optimization_results.yaml"
OUT_DIR = BASE_DIR / "outputs/debug_thresholds/examples"


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_model(cfg, device):
    with open(OPTUNA_RESULTS_PATH, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)
    best = opt.get("best_params", {})
    h1 = int(best.get("model.hidden_channels1", cfg["model"].get("hidden_channels1", 32)))
    h2 = int(best.get("model.hidden_channels2", cfg["model"].get("hidden_channels2", 16)))
    lr = float(best.get("train.learning_rate", cfg["train"]["learning_rate"]))
    log_dir = Path(cfg["hydra"]["sweep"]["dir"])
    ckpt = log_dir / Path(f"hidden{h1}_hidden{h2}_lr{round(lr, 4)}") / "ckpts" / "model_epoch_0009.pt"
    model = Autoencoder(
        in_channels=1, hidden_channels1=h1, hidden_channels2=h2, latent_channels=cfg["model"]["latent_channels"]
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    return model


def _mel_to_wav(mel_db_tensor, cfg):
    mel_power = librosa.db_to_power(mel_db_tensor.cpu().numpy())
    wav = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=cfg["dataset"]["sample_rate"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        win_length=cfg["dataset"]["n_fft"],
        n_iter=32,
        power=2.0,
    )
    return wav.astype("float32")


def main():
    cfg = _load_config(CONFIG_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # choose indices
    # anomalies: label==1, top 3 by normalized_mse
    anom_idx = df[df["label"] == 1].sort_values("normalized_mse", ascending=False).head(3).index.tolist()
    # normals: label==0, bottom 3 by normalized_mse (most normal)
    norm_idx = df[df["label"] == 0].sort_values("normalized_mse", ascending=True).head(3).index.tolist()
    chosen = [("anomaly", i) for i in anom_idx] + [("normal", i) for i in norm_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # compute db stats from training list used earlier
    data_config = cfg["dataset_normal_and_anomaly"]
    train_list = BASE_DIR / data_config["train_list"]
    data_dir = BASE_DIR / data_config["data_dir"]
    db_min, db_max = MelSpectrogramDataset.compute_db_min_max(
        train_list,
        data_dir,
        sample_rate=cfg["dataset"]["sample_rate"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        n_mels=cfg["dataset"]["n_mels"],
        target_frames=cfg["dataset"]["target_frames"],
        device=torch.device("cpu"),
    )

    # dataloader (same order as CSV)
    val_dataset = MelSpectrogramDataset(
        file_list_path=BASE_DIR / data_config["eval_list"],
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

    model = _load_model(cfg, device)
    model.eval()

    for tag, idx in chosen:
        mel_norm, label, model_id = val_dataset[idx]
        # add batch dim
        mel_norm_b = mel_norm.unsqueeze(0).to(device)
        with torch.no_grad():
            recon_norm_b = model(mel_norm_b)

        # denormalize
        mel_db = val_dataset.denormalize(mel_norm.squeeze(0)).squeeze(0).cpu()
        recon_db = val_dataset.denormalize(recon_norm_b.squeeze(0).cpu()).squeeze(0).cpu()

        # plot
        fig = plot_recon_pair(mel_db, recon_db)
        out_png = OUT_DIR / f"{tag}_{idx}_{model_id}_mel.png"
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        fig.clf()
        print(f"Saved {out_png}")

        # save audio
        wav_recon = _mel_to_wav(recon_db, cfg)
        wav_in = _mel_to_wav(mel_db, cfg)
        sf.write(OUT_DIR / f"{tag}_{idx}_{model_id}_input.wav", wav_in, cfg["dataset"]["sample_rate"])
        sf.write(OUT_DIR / f"{tag}_{idx}_{model_id}_recon.wav", wav_recon, cfg["dataset"]["sample_rate"])
        print(f"Saved audio for idx {idx}: input/recon")


if __name__ == "__main__":
    main()
