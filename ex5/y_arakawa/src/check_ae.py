"""Run a trained AE on a single wav file and reconstruct audio from the output mel."""

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

from dataloaders.dataloader import MelSpectrogramDataset, denormalize_mel_db, normalize_mel_db
from models.autoencoder import Autoencoder
from utils.visualize import plot_recon_pair

BASE_DIR = Path(__file__).resolve().parents[1]
# User-editable paths.
WAV_FILE_PATH = BASE_DIR.parents[0] / "data/dev/abundant/model_00_normal_90000000.wav"
CONFIG_PATH = BASE_DIR / "configs/config_optuna_for_comp.yaml"
OPTUNA_RESULTS_PATH = BASE_DIR / "logs/mel_ae_optuna_for_comp/optimization_results.yaml"
OUTPUT_DIR = BASE_DIR / "outputs/check_ae"


def _pad_or_trim_mel(mel: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Pad or trim a mel spectrogram to the target frame length."""
    frames = mel.shape[-1]
    if frames > target_frames:
        return mel[..., :target_frames]
    if frames < target_frames:
        pad_amount = target_frames - frames
        return torch.nn.functional.pad(mel, (0, pad_amount))
    return mel


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_model(cfg: dict, optuna_results_path: Path, device: torch.device) -> Autoencoder:
    with open(optuna_results_path, "r", encoding="utf-8") as f:
        optuna_results = yaml.safe_load(f)

    best_params = optuna_results.get("best_params", {})
    hidden_channels1 = int(best_params.get("model.hidden_channels1", cfg["model"].get("hidden_channels1", 32)))
    hidden_channels2 = int(best_params.get("model.hidden_channels2", cfg["model"].get("hidden_channels2", 16)))
    learning_rate = float(best_params.get("train.learning_rate", cfg["train"]["learning_rate"]))

    log_dir = Path(cfg["hydra"]["sweep"]["dir"])
    ckpt_path = log_dir / Path(f"lr{round(learning_rate, 4)}") / "ckpts" / "model_epoch_0009.pt"

    model = Autoencoder(
        in_channels=1,
        hidden_channels1=hidden_channels1,
        hidden_channels2=hidden_channels2,
        latent_channels=cfg["model"]["latent_channels"],
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def _wav_to_mel(wav_path: Path, cfg: dict, device: torch.device) -> torch.Tensor:
    """Convert a wav file into a padded mel spectrogram tensor."""
    waveform_np, _ = librosa.load(wav_path, sr=cfg["dataset"]["sample_rate"])
    waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(device)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg["dataset"]["sample_rate"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        n_mels=cfg["dataset"]["n_mels"],
        power=2.0,
    ).to(device)
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power").to(device)

    mel = db_transform(mel_transform(waveform))
    mel = _pad_or_trim_mel(mel, cfg["dataset"]["target_frames"])
    return mel


def _load_normalization_stats(cfg: dict) -> tuple[float, float]:
    data_dir = cfg["dataset"].get("data_dir")
    data_dir_path = Path(data_dir) if data_dir is not None else None
    return MelSpectrogramDataset.compute_db_min_max(
        Path(cfg["dataset"]["train_list"]),
        data_dir_path,
        sample_rate=cfg["dataset"]["sample_rate"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        n_mels=cfg["dataset"]["n_mels"],
        target_frames=cfg["dataset"]["target_frames"],
        device=torch.device("cpu"),
    )


def _mel_to_wav(mel_db: torch.Tensor, cfg: dict) -> np.ndarray:
    """Invert a dB-scaled mel spectrogram back to waveform with Griffin-Lim."""
    mel_power = librosa.db_to_power(mel_db.cpu().numpy())
    waveform = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=cfg["dataset"]["sample_rate"],
        n_fft=cfg["dataset"]["n_fft"],
        hop_length=cfg["dataset"]["hop_length"],
        win_length=cfg["dataset"]["n_fft"],
        n_iter=32,
        power=2.0,
    )
    return waveform.astype(np.float32)


def main() -> None:
    cfg = _load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = _load_model(cfg, OPTUNA_RESULTS_PATH, device)
    model.eval()

    db_min, db_max = _load_normalization_stats(cfg)

    input_mel_db = _wav_to_mel(WAV_FILE_PATH, cfg, device)
    input_mel = normalize_mel_db(input_mel_db, db_min, db_max).unsqueeze(0)
    with torch.no_grad():
        recon_mel = model(input_mel)

    input_mel_cpu = denormalize_mel_db(input_mel.squeeze(0), db_min, db_max).squeeze(0).cpu()
    recon_mel_cpu = denormalize_mel_db(recon_mel.squeeze(0), db_min, db_max).squeeze(0).cpu()

    figure = plot_recon_pair(input_mel_cpu, recon_mel_cpu)
    figure.savefig(OUTPUT_DIR / "mel_reconstruction.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    recon_waveform = _mel_to_wav(recon_mel_cpu, cfg)
    input_restored_waveform = _mel_to_wav(input_mel_cpu, cfg)
    sf.write(OUTPUT_DIR / "input_restored.wav", input_restored_waveform, cfg["dataset"]["sample_rate"])
    sf.write(OUTPUT_DIR / "reconstructed.wav", recon_waveform, cfg["dataset"]["sample_rate"])

    print(f"input wav: {WAV_FILE_PATH}")
    print(f"reconstructed wav: {OUTPUT_DIR / 'reconstructed.wav'}")
    print(f"mel plot: {OUTPUT_DIR / 'mel_reconstruction.png'}")


if __name__ == "__main__":
    main()
