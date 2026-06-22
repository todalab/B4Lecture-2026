# -*- coding: utf-8 -*-
"""VAE の解答例。VAE.py の #TODO を埋めた完全実装。"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MNIST_SIZE = 28


class VAE(nn.Module):
    """Variational Autoencoder (VAE) — 実装版"""

    def __init__(self, z_dim: int, h_dim: int, drop_rate: float):
        super().__init__()
        self.eps = np.spacing(1)
        self.x_dim = MNIST_SIZE * MNIST_SIZE
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Encoder 層
        self.enc_fc1 = nn.Linear(self.x_dim, h_dim)
        self.enc_fc2 = nn.Linear(h_dim, h_dim // 2)
        self.enc_fc3_mean = nn.Linear(h_dim // 2, z_dim)
        self.enc_fc3_logvar = nn.Linear(h_dim // 2, z_dim)

        # Decoder 層
        self.dec_fc1 = nn.Linear(z_dim, h_dim // 2)
        self.dec_fc2 = nn.Linear(h_dim // 2, h_dim)
        self.dec_drop = nn.Dropout(drop_rate)
        self.dec_fc3 = nn.Linear(h_dim, self.x_dim)

    def encoder(self, x: torch.Tensor):
        x = x.view(-1, self.x_dim)
        x = F.relu(self.enc_fc1(x))
        x = F.relu(self.enc_fc2(x))
        mean, log_var = self.enc_fc3_mean(x), self.enc_fc3_logvar(x)
        return mean, log_var

    def sample_z(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        epsilon = torch.randn_like(mean)
        z = mean + epsilon * torch.exp(0.5 * log_var)
        return z

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = self.dec_drop(z)
        y = torch.sigmoid(self.dec_fc3(z))
        return y

    def kld(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        sigma_sq = torch.exp(log_var)
        elbo_kl  = 0.5 * torch.sum(1 + log_var - mean**2 - sigma_sq)
        return elbo_kl

    def forward(self, x: torch.Tensor):
        mean, log_var = self.encoder(x)
        z = self.sample_z(mean, log_var)
        y = self.decoder(z)

        elbo_kl = self.kld(mean, log_var)
        elbo_rec = torch.sum(
            x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        )
        return [elbo_kl, elbo_rec], z, y
