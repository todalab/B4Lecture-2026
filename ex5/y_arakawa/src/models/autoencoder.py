"""Lightweight convolutional autoencoder for mel spectrograms."""

import torch
from torch import nn


class Encoder(nn.Module):
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


class Decoder(nn.Module):
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


class Autoencoder(nn.Module):
    """Convolutional autoencoder for mel spectrograms."""

    def __init__(
        self, in_channels: int = 1, hidden_channels1: int = 16, hidden_channels2: int = 8, latent_channels: int = 4
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels1=hidden_channels1,
            hidden_channels2=hidden_channels2,
            latent_channels=latent_channels,
        )
        self.decoder = Decoder(
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
        return recon
