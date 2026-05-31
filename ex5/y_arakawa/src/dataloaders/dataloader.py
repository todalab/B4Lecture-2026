"""Data loading utilities for mel-spectrogram autoencoder training."""

from collections.abc import Sequence
from pathlib import Path

import librosa
import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from utils.seed import make_seed_worker, make_torch_generator


def normalize_mel_db(mel_db: torch.Tensor, db_min: float, db_max: float) -> torch.Tensor:
    """Normalize dB-scaled mel spectrograms to the 0-1 range."""
    db_range = db_max - db_min
    if db_range <= 0:
        return torch.zeros_like(mel_db)
    return torch.clamp((mel_db - db_min) / db_range, 0.0, 1.0)


def denormalize_mel_db(mel_norm: torch.Tensor, db_min: float, db_max: float) -> torch.Tensor:
    """Restore 0-1 normalized mel spectrograms back to dB scale."""
    return mel_norm * (db_max - db_min) + db_min


class MelSpectrogramDataset(Dataset):
    """Dataset for mel spectrogram reconstruction."""

    def __init__(
        self,
        file_list_path: str | Path | Sequence[str | Path],
        data_dir_path: str | Path | None,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        target_frames: int,
        db_min: float | None = None,
        db_max: float | None = None,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.files = self._load_file_list(file_list_path, data_dir_path)
        self.sample_rate = sample_rate
        self.target_frames = target_frames
        self.db_min = db_min
        self.db_max = db_max
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        ).to(self.device)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power").to(self.device)

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _pad_or_trim_mel_tensor(mel: torch.Tensor, target_frames: int) -> torch.Tensor:
        frames = mel.shape[-1]
        if frames > target_frames:
            return mel[..., :target_frames]
        if frames < target_frames:
            pad_amount = target_frames - frames
            return F.pad(mel, (0, pad_amount))
        return mel

    @staticmethod
    def _load_file_list(
        file_list_path: str | Path | Sequence[str | Path], data_dir_path: str | Path | None
    ) -> list[Path]:
        """Load a text file containing one audio path per line.

        Parameters
        ----------
        file_list_path : str | Path
            Path to the list file.

        data_dir_path : str | Path
            データが格納されているフォルダのパス.

        Returns
        -------
        files : list[Path]
            Resolved audio paths (relative paths are resolved to the list file directory).
        """
        files: list[Path] = []
        if isinstance(file_list_path, (str, Path)):
            list_path = Path(file_list_path)
            base_dir = Path(data_dir_path) if data_dir_path is not None else list_path.parent
            with open(list_path, "r") as f:
                content = f.read()
                for line in content.splitlines():
                    path = Path(line)
                    files.append(path if path.is_absolute() else (base_dir / path))
        else:
            base_dir = Path(data_dir_path) if data_dir_path is not None else None
            for file_item in file_list_path:
                path = Path(file_item)
                files.append(path if path.is_absolute() or base_dir is None else (base_dir / path))
        return files

    @classmethod
    def compute_db_min_max(
        cls,
        file_list_path: str | Path | Sequence[str | Path],
        data_dir_path: str | Path | None,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        target_frames: int,
        device: torch.device | None = None,
    ) -> tuple[float, float]:
        """Compute the global dB min/max over a file list."""
        compute_device = device if device is not None else torch.device("cpu")
        files = cls._load_file_list(file_list_path, data_dir_path)
        if not files:
            raise ValueError("file_list_path does not contain any audio files")

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        ).to(compute_device)
        db_transform = torchaudio.transforms.AmplitudeToDB(stype="power").to(compute_device)

        min_db = float("inf")
        max_db = float("-inf")
        with torch.no_grad():
            for file_path in files:
                waveform_np, _ = librosa.load(file_path, sr=sample_rate)
                waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(compute_device)
                mel_db = db_transform(mel_transform(waveform))
                mel_db = cls._pad_or_trim_mel_tensor(mel_db, target_frames)
                min_db = min(min_db, float(mel_db.amin().item()))
                max_db = max(max_db, float(mel_db.amax().item()))

        return min_db, max_db

    def _pad_or_trim_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """メルスペクトログラムを既定のサイズに調節する.

        Parameters
        ----------
        mel : torch.Tensor
            Input mel spectrogram with shape (1, n_mels, frames).

        Returns
        -------
        mel : torch.Tensor
            Mel spectrogram padded or trimmed to target_frames.
        """
        return self._pad_or_trim_mel_tensor(mel, self.target_frames)

    def _normalize(self, mel_db: torch.Tensor) -> torch.Tensor:
        if self.db_min is None or self.db_max is None:
            raise ValueError("db_min and db_max must be provided for normalization")
        return normalize_mel_db(mel_db, self.db_min, self.db_max)

    def denormalize(self, mel_norm: torch.Tensor) -> torch.Tensor:
        if self.db_min is None or self.db_max is None:
            raise ValueError("db_min and db_max must be provided for normalization")
        return denormalize_mel_db(mel_norm, self.db_min, self.db_max)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        """Load one audio file, convert to mel, and return metadata.

        Parameters
        ----------
        index : int
            Index into the file list.

        Returns
        -------
        sample : tuple[torch.Tensor, int, str]
            (mel, 正常か異常か（0 or 1）, model番号) where mel is (1, n_mels, frames).
        """
        file_path = self.files[index]
        _, model, is_normal, _ = file_path.stem.split("_")

        # 正常か異常かを正常：0、異常：1で表す
        is_normal = 0 if is_normal == "normal" else 1

        # Load audio
        waveform_np, _ = librosa.load(file_path, sr=self.sample_rate)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(self.device)  # 先頭に新たな次元を追加する

        # メルスペクトログラムを計算
        mel_db = self.mel(waveform)  # torchの関数を使っている
        mel_db = self.to_db(mel_db)  # デシベル単位に変換
        mel_db = self._pad_or_trim_mel(mel_db)
        return self._normalize(mel_db), is_normal, model


def create_dataloader(
    file_list_path: str | Path | Sequence[str | Path],
    data_dir_path: str | Path | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    sample_rate: int = 8000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 40,
    target_frames: int = 40,
    seed: int = 42,
    db_min: float | None = None,
    db_max: float | None = None,
    device: torch.device | None = None,
) -> DataLoader:
    """Build a DataLoader for mel-spectrogram reconstruction.

    Parameters
    ----------
    file_list_path : str | Path
        音声ファイルを指定しているテキストファイルの相対パス.
    data_dir_path : str | Path
        音声ファイルがあるフォルダの相対パス.
    batch_size : int, optional
        Batch size.
    shuffle : bool, optional
        Whether to shuffle samples each epoch.
    sample_rate : int, optional
        Target sampling rate used by librosa.
    n_fft : int, optional
        FFT size used for mel spectrogram.
    hop_length : int, optional
        Hop length between frames.
    n_mels : int, optional
        Number of mel bins.
    target_frames : int, optional
        Fixed number of frames after padding/trim.
    seed : int, optional
        Seed for deterministic shuffling and workers.

    Returns
    -------
    dataloader : DataLoader
        Configured DataLoader instance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if db_min is None or db_max is None:
        db_min, db_max = MelSpectrogramDataset.compute_db_min_max(
            file_list_path,
            data_dir_path,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            target_frames=target_frames,
            device=torch.device("cpu"),
        )

    generator = make_torch_generator(seed)
    seed_worker = make_seed_worker(seed)
    dataset = MelSpectrogramDataset(
        file_list_path,
        data_dir_path,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        target_frames=target_frames,
        db_min=db_min,
        db_max=db_max,
        device=device,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, worker_init_fn=seed_worker)
