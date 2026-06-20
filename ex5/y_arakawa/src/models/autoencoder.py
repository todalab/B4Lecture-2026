"""Lightweight convolutional autoencoder for mel spectrograms."""

import torch
from torch import nn


class Encoder1(nn.Module):
    """Convolutional encoder for mel spectrograms."""

    def __init__(
        self, in_channels: int = 1, hidden_channels1: int = 32, hidden_channels2: int = 16, latent_channels: int = 8
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Encoder down-samples by a total factor of 4 (H/4, W/4)
            nn.Conv2d(in_channels, hidden_channels1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # /2
            # reduce by another factor of 2 using stride in conv (total /4)
            nn.Conv2d(hidden_channels1, hidden_channels2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # project to latent without further spatial reduction
            nn.Conv2d(hidden_channels2, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input mel spectrograms into latent feature maps.

        Parameters
        ----------
        x : torch.Tensor
            Input mel spectrograms with shape (B, 1, n_mels, frames).

        Returns
        -------
        latent : torch.Tensor
            Latent feature maps with shape (B, latent_channels, H/4, W/4).
        """
        return self.net(x)


class Decoder1(nn.Module):
    """Convolutional decoder for mel spectrograms."""

    def __init__(
        self, out_channels: int = 1, hidden_channels1: int = 32, hidden_channels2: int = 16, latent_channels: int = 8
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, hidden_channels2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_channels2, hidden_channels1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels1, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent feature maps back to mel spectrograms.

        Parameters
        ----------
        x : torch.Tensor
            Latent feature maps. (B, latent_channels, H/4, W/4)

        Returns
        -------
        recon : torch.Tensor
            Reconstructed mel spectrograms. (B, 1, n_mels, frames)
        """
        return self.net(x)  # (B, out_channels, H, W)


class Encoder2(nn.Module):
    """畳み込みエンコーダー.BatchNormを使用する.

    線形層の形状（Flatten後5120 -> 128）を一定に保つため、入力の n_mels が
    64 より大きい場合は H 方向のみ stride=2 でダウンサンプリングする中間層を
    追加する。
    """

    BASE_FEAT_H = 8
    BASE_FEAT_W = 5
    BOTTLENECK_CHANNELS = 128

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels1: int = 32,
        hidden_channels2: int = 16,
        latent_channels: int = 8,
        n_mels: int = 64,
        target_frames: int = 40,
    ) -> None:
        super().__init__()
        # 3回の stride=2 ダウンサンプリングで H, W は 1/8 になる
        h_after_base = n_mels // 8
        if h_after_base < self.BASE_FEAT_H or h_after_base % self.BASE_FEAT_H != 0:
            raise ValueError(
                f"n_mels={n_mels} は BASE_FEAT_H={self.BASE_FEAT_H} の 8 の倍数倍 (64, 128, ...) である必要があります"
            )
        # H を BASE_FEAT_H まで落とすために必要な追加 stride=2 (H 方向のみ) の段数
        self.extra_h_down = 0
        h_remaining = h_after_base
        while h_remaining > self.BASE_FEAT_H:
            h_remaining //= 2
            self.extra_h_down += 1

        flatten_size = self.BOTTLENECK_CHANNELS * self.BASE_FEAT_H * self.BASE_FEAT_W

        base_layers: list[nn.Module] = [
            # 入力: (1, n_mels, frames) -> 出力: (32, n_mels/2, frames/2)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 入力: (32, n_mels/2, frames/2) -> 出力: (32, n_mels/2, frames/2)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 入力: (32, n_mels/2, frames/2) -> 出力: (64, n_mels/4, frames/4)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 入力: (64, n_mels/4, frames/4) -> 出力: (128, n_mels/8, frames/8)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ]
        # H 方向のみ stride=2 で追加ダウンサンプリング
        for _ in range(self.extra_h_down):
            base_layers.extend(
                [
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                ]
            )
        base_layers.extend(
            [
                # ここで2次元のマップを1次元に平坦化 (128 * 8 * 5 = 5120次元)
                nn.Flatten(),
                # 潜在空間（ボトルネック）へ圧縮
                nn.Linear(in_features=flatten_size, out_features=128),
            ]
        )
        self.net = nn.Sequential(*base_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input mel spectrograms into latent feature maps.

        Parameters
        ----------
        x : torch.Tensor
            Input mel spectrograms with shape (B, 1, n_mels, frames).

        Returns
        -------
        latent : torch.Tensor
            Latent feature maps with shape (B, latent_channels).
        """
        return self.net(x)


class Decoder2(nn.Module):
    """畳み込みデコーダー.BatchNormを使用する.

    Encoder2 と対称になるよう、n_mels が 64 より大きい場合は H 方向のみの
    ConvTranspose アップサンプリング層を追加する。
    """

    BASE_FEAT_H = 8
    BASE_FEAT_W = 5
    BOTTLENECK_CHANNELS = 128

    def __init__(
        self,
        out_channels: int = 1,
        hidden_channels1: int = 32,
        hidden_channels2: int = 16,
        latent_channels: int = 8,
        n_mels: int = 64,
        target_frames: int = 40,
    ) -> None:
        super().__init__()
        h_after_base = n_mels // 8
        if h_after_base < self.BASE_FEAT_H or h_after_base % self.BASE_FEAT_H != 0:
            raise ValueError(
                f"n_mels={n_mels} は BASE_FEAT_H={self.BASE_FEAT_H} の 8 の倍数倍 (64, 128, ...) である必要があります"
            )
        self.extra_h_up = 0
        h_remaining = h_after_base
        while h_remaining > self.BASE_FEAT_H:
            h_remaining //= 2
            self.extra_h_up += 1

        flatten_size = self.BOTTLENECK_CHANNELS * self.BASE_FEAT_H * self.BASE_FEAT_W

        # 潜在空間から平坦化された特徴量へ逆変換
        self.linear = nn.Linear(in_features=128, out_features=flatten_size)
        self.activation = nn.ReLU()

        deconv_layers: list[nn.Module] = []
        # H 方向のみ stride=2 のアップサンプリング (Encoder2 の追加層と対称)
        for _ in range(self.extra_h_up):
            deconv_layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels=128, out_channels=128, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)
                    ),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                ]
            )
        deconv_layers.extend(
            [
                # 入力: (128, n_mels/8, frames/8) -> 出力: (64, n_mels/4, frames/4)
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # 入力: (64, n_mels/4, frames/4) -> 出力: (32, n_mels/2, frames/2)
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # 入力: (32, n_mels/2, frames/2) -> 出力: (32, n_mels/2, frames/2)
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # 入力: (32, n_mels/2, frames/2) -> 出力: (1, n_mels, frames)
                nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),  # 出力を0-1の範囲に制限
            ]
        )
        self.deconv = nn.Sequential(*deconv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent feature maps back to mel spectrograms.

        Parameters
        ----------
        x : torch.Tensor
            Latent feature maps. (B, latent_channels)

        Returns
        -------
        recon : torch.Tensor
            Reconstructed mel spectrograms. (B, 1, n_mels, frames)
        """
        x = self.linear(x)
        x = self.activation(x)
        x = x.view(-1, self.BOTTLENECK_CHANNELS, self.BASE_FEAT_H, self.BASE_FEAT_W)
        x = self.deconv(x)
        return x


class Autoencoder(nn.Module):
    """Convolutional autoencoder for mel spectrograms.

    Parameters
    ----------
    variant : str
        'fc' uses the fully-connected bottleneck (`Encoder2`/`Decoder2`),
        'conv' uses spatial conv bottleneck (`Encoder1`/`Decoder1`).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels1: int = 16,
        hidden_channels2: int = 8,
        latent_channels: int = 4,
        variant: str = "fc",
        n_mels: int = 64,
        target_frames: int = 40,
    ) -> None:
        super().__init__()
        self.variant = variant
        if variant == "conv":
            # conv variant: Encoder1/Decoder1 keep spatial maps
            self.encoder = Encoder1(
                in_channels=in_channels,
                hidden_channels1=hidden_channels1,
                hidden_channels2=hidden_channels2,
                latent_channels=latent_channels,
            )
            self.decoder = Decoder1(
                out_channels=in_channels,
                hidden_channels1=hidden_channels1,
                hidden_channels2=hidden_channels2,
                latent_channels=latent_channels,
            )
        else:
            # default: fully-connected bottleneck
            self.encoder = Encoder2(
                in_channels=in_channels,
                hidden_channels1=hidden_channels1,
                hidden_channels2=hidden_channels2,
                latent_channels=latent_channels,
                n_mels=n_mels,
                target_frames=target_frames,
            )
            self.decoder = Decoder2(
                out_channels=in_channels,
                hidden_channels1=hidden_channels1,
                hidden_channels2=hidden_channels2,
                latent_channels=latent_channels,
                n_mels=n_mels,
                target_frames=target_frames,
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input mel spectrograms into latent feature maps.

        Parameters
        ----------
        x : torch.Tensor
            Input mel spectrograms. (B, 1, n_mels, frames)

        Returns
        -------
        latent : torch.Tensor
            Latent feature maps. (B, latent_channels, H/4, W/4)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent feature maps back to mel spectrograms.

        Parameters
        ----------
        z : torch.Tensor
            Latent feature maps. (B, latent_channels, H/4, W/4)

        Returns
        -------
        recon : torch.Tensor
            Reconstructed mel spectrograms. (B, 1, n_mels, frames)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run encode-decode pass and return reconstruction.

        Parameters
        ----------
        x : torch.Tensor
            Input mel spectrograms. (B, 1, n_mels, frames)

        Returns
        -------
        recon : torch.Tensor
            Reconstructed mel spectrograms. (B, 1, n_mels, frames)
        """
        z = self.encode(x)
        recon = self.decode(z)
        # Keep the same residual blending behavior; works for both variants
        return torch.clamp(x + 0.1 * recon, 0.0, 1.0)
