"""Dataset for loading fan audio and log-Mel spectrograms."""

from pathlib import Path

import torchaudio
from mel import build_mel, log_mel
from torch.utils.data import Dataset


class Fan(Dataset):
    """Dataset of fan audio files represented as log-Mel spectrogrammes."""

    def __init__(self, data_folders: list[str]):
        """Load audio paths and initialise spectrogramme parameters."""
        self.paths = sorted(
            path for folder in data_folders for path in Path(folder).glob("*.wav")
        )

        # Assume all audio have the same sample rate
        _, sample_rate = torchaudio.load(self.paths[0])
        self.sample_rate = sample_rate
        self.mel = build_mel(sample_rate)
        self.labels = [int("anomaly" in p.name) for p in self.paths]

    def __len__(self):
        """Return the number of audio samples in the dataset."""
        return len(self.paths)

    def __getitem__(self, i):
        """Return a log-Mel spectrogramme and its binary label."""
        waveform, _ = torchaudio.load(self.paths[i])
        log_spectrogramme = log_mel(waveform, self.sample_rate, self.mel)
        label = self.labels[i]
        return log_spectrogramme, label
