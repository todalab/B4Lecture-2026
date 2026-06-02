"""
audio_processor.py
音声ファイル → log-Mel スペクトログラム Tensor に変換する.
外部依存: librosa, numpy, torch
"""

from typing import Optional

import librosa
import numpy as np
import torch


def load_and_segment(
    path: str,
    sr: int,
    segment_sec: float,
    hop_length_sec: Optional[float] = None,
) -> list:
    """
    wav ファイルを読み込み、固定長セグメントに分割する.

    Args:
        path: wav ファイルパス
        sr: サンプリングレート
        segment_sec: 1セグメントの長さ（秒）
        hop_length_sec: セグメントのホップ長（秒）.None なら非重複

    Returns:
        list of (segment_samples,) ndarray
    """
    wave, _ = librosa.load(path, sr=sr, mono=True)
    seg_len = int(sr * segment_sec)
    hop_len = int(sr * hop_length_sec) if hop_length_sec else seg_len

    segments = []
    start = 0
    while start + seg_len <= len(wave):
        segments.append(wave[start : start + seg_len])
        start += hop_len

    # 最低1セグメント保証（ファイルが短い場合はゼロパディング）
    if len(segments) == 0:
        padded = np.zeros(seg_len, dtype=np.float32)
        padded[: len(wave)] = wave
        segments.append(padded)

    return segments


def wave_to_logmel(
    wave: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> np.ndarray:
    """
    波形 → log-Mel スペクトログラム

    Returns:
        (n_mels, T) float32 ndarray
    """
    mel = librosa.feature.melspectrogram(
        y=wave,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return log_mel  # (n_mels, T)


def normalize(log_mel: np.ndarray) -> np.ndarray:
    """per-clip 正規化（mean=0, std=1）"""
    mu = log_mel.mean()
    sigma = log_mel.std() + 1e-8
    return (log_mel - mu) / sigma


def process_file(
    path: str,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    segment_sec: float,
) -> list:
    """
    wav ファイル 1 本 → Tensor リスト

    Returns:
        list of Tensor (1, n_mels, T)  ← 1ch 画像として扱う
    """
    segments = load_and_segment(path, sr, segment_sec)
    tensors = []
    for seg in segments:
        log_mel = wave_to_logmel(seg, sr, n_fft, hop_length, n_mels)
        log_mel = normalize(log_mel)
        t = torch.from_numpy(log_mel).unsqueeze(0)  # (1, n_mels, T)
        tensors.append(t)
    return tensors
