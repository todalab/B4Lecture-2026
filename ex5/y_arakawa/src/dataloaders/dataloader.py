"""Data loading utilities for mel-spectrogram autoencoder training."""

from pathlib import Path

import librosa
import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from utils.seed import make_seed_worker, make_torch_generator


class MelSpectrogramDataset(Dataset):
    """Dataset for mel spectrogram reconstruction."""

    def __init__(
        self,
        file_list_path: str | Path,
        data_dir_path: str | Path,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        target_frames: int,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.files = self._load_file_list(file_list_path, data_dir_path)
        self.sample_rate = sample_rate
        self.target_frames = target_frames
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

    # def _get_file_paths(self, folder_path: str | Path) -> list[Path]:
    #     file_paths: list[Path] = []
    #     path = Path(folder_path)
    #     for root, _, files in os.walk(path):
    #         for file in files:
    #             file = Path(file)
    #             file_paths.append(file if file.is_absolute() else (Path(root) / file))
    #     return file_paths

    def _load_file_list(self, file_list_path: str | Path, data_dir_path: str | Path) -> list[Path]:
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
        list_path = Path(file_list_path)
        base_dir = Path(data_dir_path)
        with open(list_path, "r") as f:
            content = f.read()
            for line in content.splitlines():
                path = Path(line)
                files.append(path if path.is_absolute() else (base_dir / path))
        return files

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
        frames = mel.shape[-1]
        if frames > self.target_frames:
            return mel[..., : self.target_frames]
        if frames < self.target_frames:
            pad_amount = self.target_frames - frames
            return F.pad(mel, (0, pad_amount))
        return mel

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
        mel = self.mel(waveform)  # torchの関数を使っている
        mel = self.to_db(mel)  # デシベル単位に変換
        return self._pad_or_trim_mel(mel), is_normal, model


def create_dataloader(
    file_list_path: str | Path,
    data_dir_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    sample_rate: int = 8000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 40,
    target_frames: int = 40,
    seed: int = 42,
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
        device=device,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, worker_init_fn=seed_worker)
