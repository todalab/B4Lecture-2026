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
    """畳み込みエンコーダー.BatchNormを使用する."""

    def __init__(
        self, in_channels: int = 1, hidden_channels1: int = 32, hidden_channels2: int = 16, latent_channels: int = 8
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # 入力: (1, 64, 40) -> 出力: (32, 32, 20)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 入力: (32, 32, 20) -> 出力: (32, 32, 20)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 入力: (32, 32, 20) -> 出力: (64, 16, 10)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 入力: (64, 16, 10) -> 出力: (128, 8, 5)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # ここで2次元のマップを1次元に平坦化 (128 * 8 * 5 = 5120次元)
            nn.Flatten(),
            # 潜在空間（ボトルネック）へ圧縮
            nn.Linear(in_features=128 * 8 * 5, out_features=128),
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
            Latent feature maps with shape (B, latent_channels).
        """
        return self.net(x)


class Decoder2(nn.Module):
    """畳み込みデコーダー.BatchNormを使用する."""

    def __init__(
        self, out_channels: int = 1, hidden_channels1: int = 32, hidden_channels2: int = 16, latent_channels: int = 8
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # 潜在空間から平坦化された特徴量へ逆変換
            nn.Linear(in_features=128, out_features=128 * 8 * 5),
            nn.ReLU(),
            # 入力: (128, 8, 5) -> 出力: (64, 16, 10)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 入力: (64, 16, 10) -> 出力: (32, 32, 20)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 入力: (32, 32, 20) -> 出力: (32, 32, 20)
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 入力: (32, 32, 20) -> 出力: (1, 64, 40)
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # 出力を0-1の範囲に制限
        )

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
        x = self.net[0](x)  # Linear層を通す
        x = x.view(-1, 128, 8, 5)  # 平坦化された特徴量を2次元マップに変換
        x = self.net[1:](x)  # 残りのConvTranspose層を通す
        return x


class Autoencoder(nn.Module):
    """Convolutional autoencoder for mel spectrograms."""

    def __init__(
        self, in_channels: int = 1, hidden_channels1: int = 16, hidden_channels2: int = 8, latent_channels: int = 4
    ) -> None:
        super().__init__()
        self.encoder = Encoder2(
            in_channels=in_channels,
            hidden_channels1=hidden_channels1,
            hidden_channels2=hidden_channels2,
            latent_channels=latent_channels,
        )
        self.decoder = Decoder2(
            out_channels=in_channels,
            hidden_channels1=hidden_channels1,
            hidden_channels2=hidden_channels2,
            latent_channels=latent_channels,
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
        return torch.clamp(x + 0.1 * recon, 0.0, 1.0)
