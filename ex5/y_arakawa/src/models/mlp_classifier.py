"""AEの潜在特徴量を入力とするMLP分類器ヘッド。"""

from collections.abc import Sequence

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """AEの潜在ベクトルを1つのロジットに写像する単純なMLP。

    Parameters
    ----------
    input_dim : int
        平坦化した潜在ベクトルの次元数。
    hidden_dims : Sequence[int]
        隠れ層のサイズ。空のシーケンスを渡すと単一のLinear層になる。
    dropout : float
        各隠れ層の活性化後に適用するDropoutの確率。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """順伝播。

        Parameters
        ----------
        z : torch.Tensor
            形状 (B, ...) の潜在特徴量テンソル。dim>=1 で平坦化される。

        Returns
        -------
        logits : torch.Tensor
            形状 (B,) のロジット。確率を得るには外部でsigmoidを適用すること。
        """
        z_flat = z.flatten(start_dim=1)
        return self.net(z_flat).squeeze(-1)
