# -*- coding: utf-8 -*-
"""
Ex7 B4講義 — VAE（変分自己符号化器）
このファイルは VAE モデルの実装を行うためのコードです。
以下のメソッドの #TODO と記載されている箇所を実装してください:
    - VAE.encoder
    - VAE.sample_z
    - VAE.decoder
    - VAE.forward
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MNIST_SIZE = 28


class VAE(nn.Module):
    """Variational Autoencoder (VAE).

    Args:
        z_dim(int):      潜在変数の次元数
        h_dim(int):      中間層の次元数
        drop_rate(float): Dropout 率

    Functions:
        __init__(z_dim, h_dim, drop_rate): 初期化コンストラクタ
        encoder(x):                        入力画像 → (μ, log σ²)
        sample_z(mean, log_var, device):   Reparametrization trick で z をサンプリング
        decoder(z):                        潜在変数 → 再構成画像
        forward(x, device):                ELBO の各項を計算してフォワードパスを実行

    （ヒント）VAE の ELBO（変分下限）:
        ELBO = -KL[q(z|x) || p(z)] + E_q[log p(x|z)]
        loss = -ELBO  を最小化することで ELBO を最大化する

    Encoder のアーキテクチャ:
        x_dim(784) → h_dim → h_dim//2 → z_dim (mean)
                                        → z_dim (log_var)

    Decoder のアーキテクチャ:
        z_dim → h_dim//2 → h_dim → x_dim(784)  [最終層: Sigmoid]
    """

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
        """入力画像を潜在変数の分布パラメータに変換する。

        Args:
            x(batch, x_dim): 入力画像（DataLoader により平坦化済み）

        Returns:
            mean(batch, z_dim):    潜在変数の平均 μ
            log_var(batch, z_dim): 潜在変数の対数分散 log σ²

        参照: "Auto-Encoding Variational Bayes" Appendix C.1
        """
        x = x.view(-1, self.x_dim)
        #TODO: enc_fc1 → ReLU → enc_fc2 → ReLU の順に通す。
        #      その後 enc_fc3_mean と enc_fc3_logvar に通して mean と log_var を返す。
        raise NotImplementedError("VAE.encoder の TODO を実装してください")
        return mean, log_var

    def sample_z(
        self, mean: torch.Tensor, log_var: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Reparametrization trick で潜在変数をサンプリングする。

        Args:
            mean(batch, z_dim):    平均 μ
            log_var(batch, z_dim): 対数分散 log σ²
            device(torch.device):  使用デバイス

        Returns:
            z(batch, z_dim): サンプリングされた潜在変数

        （ヒント）Reparametrization trick:
            ε ~ N(0, I)
            z = μ + ε ⊙ exp(0.5 * log σ²)

        参照: "Auto-Encoding Variational Bayes" Section 2.4, Eq. (4)
        """
        #TODO: mean と同じ形状の ε を標準正規分布からサンプリングし、z を計算して返す。
        #      ε のサンプリングには torch.randn(..., device=device) を使う。
        raise NotImplementedError("VAE.sample_z の TODO を実装してください")
        return z

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """潜在変数から再構成画像を生成する。

        Args:
            z(batch, z_dim): 潜在変数

        Returns:
            y(batch, x_dim): 再構成画像、値域 [0, 1]

        参照: "Auto-Encoding Variational Bayes" Appendix C.1
        """
        #TODO: dec_fc1 → ReLU → dec_fc2 → ReLU → dec_drop → dec_fc3 → Sigmoid の順に通す。
        raise NotImplementedError("VAE.decoder の TODO を実装してください")
        return y

    def forward(self, x: torch.Tensor, device: torch.device):
        """ELBO の各項を計算してフォワードパスを実行する。

        Args:
            x(batch, x_dim): 入力画像（DataLoader により平坦化済み）
            device(torch.device): 使用デバイス

        Returns:
            [elbo_kl, elbo_rec]: ELBO の2項
                elbo_kl(scalar):  KL 項（= -KL[q||p]、≤ 0）
                elbo_rec(scalar): 再構成項（≤ 0）
            z(batch, z_dim): サンプリングされた潜在変数
            y(batch, x_dim): 再構成画像

        （ヒント）各項の計算式:
            elbo_kl  = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
            elbo_rec = torch.sum(x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps))
            loss     = -(elbo_kl + elbo_rec)  ← main.py で最小化

        参照: "Auto-Encoding Variational Bayes" Eq. (3), Appendix B（KL の解析解）
        """
        x = x.to(device)
        #TODO: encoder → sample_z → decoder の順に呼び出す。
        #      その後 elbo_kl と elbo_rec を計算して [elbo_kl, elbo_rec], z, y を返す。
        raise NotImplementedError("VAE.forward の TODO を実装してください")
        return [elbo_kl, elbo_rec], z, y
