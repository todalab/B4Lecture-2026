"""Utilities for computing log-Mel spectrogrammes from audio waveforms."""

import torch
from torch import Tensor
from torchaudio.pipelines._tts.interface import Optional
from torchaudio.transforms import MelSpectrogram

EPS = 1e-6


def build_mel(sample_rate: int) -> MelSpectrogram:
    """Build and return a Mel spectrogram transform."""
    return MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=256)


def log_mel(
    waveform: Tensor, sample_rate: int, mel: Optional[MelSpectrogram] = None
) -> Tensor:
    """Convert a waveform to a log-Mel spectrogram."""
    mel = mel or build_mel(sample_rate)
    return torch.log(mel(waveform) + EPS)
