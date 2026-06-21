# -*- coding: utf-8 -*-
"""VAE の実装ファイル。#TODO の箇所を埋めてください。"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MNIST_SIZE = 28


class VAE(nn.Module):
    """Variational Autoencoder (VAE)."""

    def __init__(self, z_dim: int, h_dim: int, drop_rate: float):
        """
        Parameters
        ----------
        z_dim : int
            潜在変数の次元数
        h_dim : int
            中間層の次元数
        drop_rate : float
            Dropout 率
        """
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
        """入力画像を潜在変数の分布パラメータに変換する。

        Parameters
        ----------
        x : torch.Tensor
            入力画像 (batch, x_dim)

        Returns
        -------
        mean : torch.Tensor
            潜在変数の平均 μ (batch, z_dim)
        log_var : torch.Tensor
            潜在変数の対数分散 log σ² (batch, z_dim)

        参照: "Auto-Encoding Variational Bayes" Appendix C.1
        """
        # TODO: x を (batch, x_dim) に平坦化し、enc_fc1 → ReLU → enc_fc2 → ReLU を通す。
        #       その後 enc_fc3_mean と enc_fc3_logvar に通して mean と log_var を返す。
        raise NotImplementedError

    def sample_z(
        self, mean: torch.Tensor, log_var: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Reparametrization trick で潜在変数をサンプリングする。

        Parameters
        ----------
        mean : torch.Tensor
            平均 μ (batch, z_dim)
        log_var : torch.Tensor
            対数分散 log σ² (batch, z_dim)
        device : torch.device

        Returns
        -------
        z : torch.Tensor
            サンプリングされた潜在変数 (batch, z_dim)

        参照: "Auto-Encoding Variational Bayes" Section 2.4, Eq. (4)
             z = μ + ε ⊙ exp(0.5 * log σ²),   ε ~ N(0, I)
        """
        # TODO: mean と同じ形状の ε を標準正規分布からサンプリングし、z を返す。
        raise NotImplementedError

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """潜在変数から再構成画像を生成する。

        Parameters
        ----------
        z : torch.Tensor
            潜在変数 (batch, z_dim)

        Returns
        -------
        y : torch.Tensor
            再構成画像 (batch, x_dim), 値域 [0, 1]

        参照: "Auto-Encoding Variational Bayes" Appendix C.1
        """
        # TODO: dec_fc1 → ReLU → dec_fc2 → ReLU → dec_drop → dec_fc3 → Sigmoid の順に通す。
        raise NotImplementedError

    def forward(self, x: torch.Tensor, device: torch.device):
        """ELBO の各項を計算してフォワードパスを実行する。

        Parameters
        ----------
        x : torch.Tensor
            入力画像 (batch, x_dim)
        device : torch.device

        Returns
        -------
        [elbo_kl, elbo_rec] : list[torch.Tensor]
            elbo_kl  = 0.5 * Σ(1 + log σ² - μ² - σ²)    ← -KL[q||p],  ≤ 0
            elbo_rec = Σ[x log(ŷ+ε) + (1-x) log(1-ŷ+ε)] ← 再構成対数尤度, ≤ 0
        z : torch.Tensor
            潜在変数 (batch, z_dim)
        y : torch.Tensor
            再構成画像 (batch, x_dim)

        Note:
            loss = -(elbo_kl + elbo_rec)  を最小化することで ELBO を最大化する。

        参照: "Auto-Encoding Variational Bayes" Eq. (3), Appendix B (KL の解析解)
        """
        # TODO: x.to(device) → encoder → sample_z → decoder の順に呼び出す。
        #       その後 elbo_kl と elbo_rec を計算して返す。
        #
        #   elbo_kl  = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
        #   elbo_rec = torch.sum(x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps))
        raise NotImplementedError
