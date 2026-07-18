"""Affine coupling flow primitives."""

from __future__ import annotations

import torch
from torch import nn

from nf_assignment.flows.transforms import Transform


class AffineCouplingTransform(Transform):
    """Shared affine coupling transform for toy and sequence flows."""

    def __init__(self, conditioner: nn.Module):
        """Store the neural network that predicts shift and log-scale parameters."""

        super().__init__()
        self.conditioner = conditioner

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply affine coupling in the data-to-latent direction.

        Args:
            x: Input tensor shaped ``[batch, channels]`` for toy data or
                ``[batch, channels, frames]`` for speech features. The channel
                axis is split into identity and transformed halves.
            mask: Optional speech mask shaped ``[batch, 1, frames]`` where one
                marks valid frames and zero marks padding.
            condition: Optional conditioner input shaped ``[batch, cond_channels,
                frames]`` on the same frame grid as ``x``.

        Returns:
            A pair ``(y, log_det)`` where ``y`` has the same axes as ``x`` and
            ``log_det`` is shaped ``[batch]``.
        """

        # channel を前半（恒等）と後半（変換）に分割する channel-split coupling。
        # コンディショナーは恒等部分を入力に、変換部分と同じ形状の a, b を出力する。
        if x.ndim == 2:
            channels = x.shape[1]
            half = channels // 2
            x_identity = x[:, :half]  # [batch, half] そのまま通す
            x_transform = x[:, half:]  # [batch, channels - half] affine 変換対象
            a_b = self.conditioner(x_identity)  # [batch, 2 * (channels - half)]
            # コンディショナーは [shift, log_scale] の順で連結して返す
            b, a = a_b.chunk(2, dim=1)  # b=shift, a=log_scale  [batch, channels - half]
            y_transform = x_transform * torch.exp(a) + b  # [batch, channels - half]
            y = torch.cat([x_identity, y_transform], dim=1)  # [batch, channels]
            log_det = a.sum(dim=1)  # [batch] 対角成分（log-scale）の和

        elif x.ndim == 3:
            channels = x.shape[1]
            half = channels // 2
            x_identity = x[:, :half, :]  # [batch, half, frames]
            x_transform = x[:, half:, :]  # [batch, channels - half, frames]
            a_b = self.conditioner(
                x_identity, condition=condition, mask=mask
            )  # [batch, 2 * (channels - half), frames]
            b, a = a_b.chunk(2, dim=1)  # b=shift, a=log_scale  [batch, channels - half, frames]
            y_transform = x_transform * torch.exp(a) + b  # [batch, channels - half, frames]
            y = torch.cat([x_identity, y_transform], dim=1)  # [batch, channels, frames]
            if mask is not None:
                y = y * mask  # パディングフレームを 0 に戻す
                log_det = (a * mask).sum(dim=(1, 2))  # [batch] 有効フレームのみ加算
            else:
                log_det = a.sum(dim=(1, 2))  # [batch]
        else:
            raise ValueError("AffineCouplingTransform only supports 2D or 3D inputs.")

        return y, log_det

    def inverse(
        self,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply affine coupling in the latent-to-data direction.

        Args:
            y: Latent tensor shaped ``[batch, channels]`` for toy data or
                ``[batch, channels, frames]`` for speech features.
            mask: Optional speech mask shaped ``[batch, 1, frames]``.
            condition: Optional conditioner input shaped ``[batch, cond_channels,
                frames]``.

        Returns:
            A pair ``(x, log_det)`` where ``x`` has the same axes as ``y`` and
            ``log_det`` is shaped ``[batch]``.
        """

        # forward と同じ channel-split。恒等部分は不変なので y の前半をそのまま使える。
        if y.ndim == 2:
            channels = y.shape[1]
            half = channels // 2
            y_identity = y[:, :half]  # [batch, half]
            y_transform = y[:, half:]  # [batch, channels - half]
            a_b = self.conditioner(y_identity)  # [batch, 2 * (channels - half)]
            b, a = a_b.chunk(2, dim=1)  # b=shift, a=log_scale  [batch, channels - half]
            x_transform = (y_transform - b) * torch.exp(-a)  # 加算→減算、乗算→除算
            x = torch.cat([y_identity, x_transform], dim=1)  # [batch, channels]
            log_det = -a.sum(dim=1)  # [batch] 逆変換なので符号反転

        elif y.ndim == 3:
            channels = y.shape[1]
            half = channels // 2
            y_identity = y[:, :half, :]  # [batch, half, frames]
            y_transform = y[:, half:, :]  # [batch, channels - half, frames]
            a_b = self.conditioner(
                y_identity, condition=condition, mask=mask
            )  # [batch, 2 * (channels - half), frames]
            b, a = a_b.chunk(2, dim=1)  # b=shift, a=log_scale  [batch, channels - half, frames]
            x_transform = (y_transform - b) * torch.exp(-a)  # [batch, channels - half, frames]
            x = torch.cat([y_identity, x_transform], dim=1)  # [batch, channels, frames]
            if mask is not None:
                x = x * mask
                log_det = -(a * mask).sum(dim=(1, 2))  # [batch]
            else:
                log_det = -a.sum(dim=(1, 2))  # [batch]
        else:
            raise ValueError("AffineCouplingTransform only supports 2D or 3D inputs.")
        return x, log_det


class AffineCouplingBlock(Transform):
    """Toy affine coupling layer using the shared coupling transform."""

    def __init__(self, param_map: nn.Module):
        """Wrap a toy conditioner that maps ``[batch, channels / 2]`` to affine params."""

        super().__init__()
        self.transform = AffineCouplingTransform(param_map)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform toy samples from data to latent axes.

        Args:
            x: Tensor shaped ``[batch, 2]`` where axis 0 is sample index and
                axis 1 contains the two toy coordinates.

        Returns:
            A pair ``(z, log_det)`` with ``z`` shaped ``[batch, 2]`` and
            ``log_det`` shaped ``[batch]``.
        """

        return self.transform(x)

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform toy latent samples back to data axes.

        Args:
            y: Tensor shaped ``[batch, 2]`` where axis 0 is sample index and
                axis 1 contains latent coordinates.

        Returns:
            A pair ``(x, log_det)`` with ``x`` shaped ``[batch, 2]`` and
            ``log_det`` shaped ``[batch]``.
        """

        return self.transform.inverse(y)


class SequenceAffineCoupling(Transform):
    """Speech affine coupling layer using the shared coupling transform."""

    def __init__(self, channels: int, conditioner: nn.Module):
        """Create a channel-split speech coupling transform."""

        super().__init__()
        if channels % 2 != 0:
            raise ValueError("channels must be even for channel-split coupling.")
        self.channels = int(channels)
        self.conditioner = conditioner
        self.transform = AffineCouplingTransform(conditioner)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform speech features from data to latent axes.

        Args:
            x: Tensor shaped ``[batch, coded_sp_channels, frames]``.
            mask: Optional mask shaped ``[batch, 1, frames]``.
            condition: Optional condition tensor shaped ``[batch, cond_channels,
                frames]``.

        Returns:
            A pair ``(z, log_det)`` with ``z`` shaped like ``x`` and ``log_det``
            shaped ``[batch]``.
        """

        return self.transform(x, mask=mask, condition=condition)

    def inverse(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform speech latent features back to data axes.

        Args:
            z: Tensor shaped ``[batch, coded_sp_channels, frames]``.
            mask: Optional mask shaped ``[batch, 1, frames]``.
            condition: Optional condition tensor shaped ``[batch, cond_channels,
                frames]``.

        Returns:
            A pair ``(x, log_det)`` with ``x`` shaped like ``z`` and ``log_det``
            shaped ``[batch]``.
        """

        return self.transform.inverse(z, mask=mask, condition=condition)
