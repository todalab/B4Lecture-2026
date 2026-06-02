"""
dataset.py
  学習 : abundant/model_XX_normal のみ
  評価 : dev/model_XX_normal + dev/model_XX_anomaly
"""

import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

from audio_processor import process_file


def extract_model_id(filename: str) -> str:
    """ファイル名から model_id（2桁文字列）を抽出する."""
    m = re.match(r"model_(\d+)_", filename)
    return m.group(1) if m else "00"


class NormalDataset(Dataset):
    """
    学習用：abundant/ の正常音のみ.
    model_id でフィルタリング.
    """

    def __init__(self, data_cfg, model_id: str):
        self.cfg      = data_cfg
        self.model_id = model_id
        self.tensors  = self._load_all()

    def _load_all(self) -> list:
        """abundant/ 配下の正常音ファイルを全て読み込み Tensor リストを返す."""
        abundant_dir = Path(self.cfg.data_dir) / "dev" / "abundant"
        tensors = []
        for p in sorted(abundant_dir.glob(f"model_{self.model_id}_normal_*.wav")):
            tensors.extend(self._process(str(p)))
        return tensors

    def _process(self, path: str) -> list:
        """wav ファイルを Tensor リストに変換する."""
        return process_file(
            path,
            sr=self.cfg.sr,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            segment_sec=self.cfg.segment_sec,
        )

    def __len__(self) -> int:
        """データセットのサンプル数を返す."""
        return len(self.tensors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """idx 番目の Tensor (1, n_mels, T) を返す."""
        return self.tensors[idx]


class EvalDataset(Dataset):
    """
    評価用：dev/ の正常 + 異常.
    model_id でフィルタリング.
    """

    def __init__(self, data_cfg, model_id: str):
        self.cfg      = data_cfg
        self.model_id = model_id
        self.samples  = self._collect()  # list of (tensor, label)

    def _collect(self) -> list:
        """dev/ 配下の正常・異常ファイルを収集し (tensor, label) リストを返す."""
        dev_dir = Path(self.cfg.data_dir) / "dev"
        samples = []
        for p in sorted(dev_dir.glob(f"model_{self.model_id}_*.wav")):
            if p.parent.name == "abundant":
                continue
            label = 1 if "_anomaly_" in p.name else 0
            for t in self._process(str(p)):
                samples.append((t, label))
        return samples

    def _process(self, path: str) -> list:
        """wav ファイルを Tensor リストに変換する."""
        return process_file(
            path,
            sr=self.cfg.sr,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            segment_sec=self.cfg.segment_sec,
        )

    def __len__(self) -> int:
        """データセットのサンプル数を返す."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """idx 番目の (tensor, label) を返す."""
        tensor, label = self.samples[idx]
        return tensor, label