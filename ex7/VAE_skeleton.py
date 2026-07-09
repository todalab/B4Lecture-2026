# -*- coding: utf-8 -*-
"""
Ex7 B4講義 — VAE（変分自己符号化器）
このファイルは VAE モデルの実装を行うためのコードです。
以下のメソッドの #TODO と記載されている箇所を実装してください:
    - VAE.encoder
    - VAE.reparametrization_trick
    - VAE.decoder
    - VAE.kld
    - VAE.forward
"""

import numpy as np
import torch
import torch.nn as nn

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
        reparametrization_trick(mean, log_var):           Reparametrization trick で z をサンプリング
        decoder(z):                        潜在変数 → 再構成画像
        kld(mean, log_var):                KL[q(z|x) || p(z)] の解析解
        forward(x):                        ELBO の各項を計算してフォワードパスを実行

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
        # TODO: enc_fc1 → ReLU → enc_fc2 → ReLU の順に通す。
        #      その後 enc_fc3_mean と enc_fc3_logvar に通して mean と log_var を返す。
        x = torch.relu(self.enc_fc1(x))
        x = torch.relu(self.enc_fc2(x))
        mean = self.enc_fc3_mean(x)
        log_var = self.enc_fc3_logvar(x)
        return mean, log_var

    def reparametrization_trick(
        self, mean: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparametrization trick で潜在変数をサンプリングする。

        Args:
            mean(batch, z_dim):    平均 μ
            log_var(batch, z_dim): 対数分散 log σ²

        Returns:
            z(batch, z_dim): サンプリングされた潜在変数

        （ヒント）Reparametrization trick:
            ε ~ N(0, I)
            z = μ + ε ⊙ exp(0.5 * log σ²)

        参照: "Auto-Encoding Variational Bayes" Section 2.4, Eq. (4)
        """
        # TODO: mean と同じ形状の ε を標準正規分布からサンプリングし、z を計算して返す。
        eps = torch.randn(mean.size(), device=mean.device)
        z = mean + eps * torch.exp(0.5 * log_var)
        return z

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """潜在変数から再構成画像を生成する。

        Args:
            z(batch, z_dim): 潜在変数

        Returns:
            y(batch, x_dim): 再構成画像、値域 [0, 1]

        参照: "Auto-Encoding Variational Bayes" Appendix C.1
        """
        # TODO: dec_fc1 → ReLU → dec_fc2 → ReLU → dec_drop → dec_fc3 → Sigmoid の順に通す。
        x = torch.relu(self.dec_fc1(z))
        x = torch.relu(self.dec_fc2(x))
        x = self.dec_drop(x)
        y = torch.sigmoid(self.dec_fc3(x))
        return y

    def kld(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """KL ダイバージェンス KL[q(z|x) || p(z)] を計算する。

        Args:
            mean(batch, z_dim):    近似事後分布の平均 μ
            log_var(batch, z_dim): 近似事後分布の対数分散 log σ²

        Returns:
            kl(scalar): KL[q(z|x) || p(z)]（常に ≥ 0）

        参照: "Auto-Encoding Variational Bayes" Appendix B
        """
        # TODO: KL[q(z|x) || p(z)] の解析解を実装する。
        #       torch.distributions.kl_divergence() の使用は禁止。
        # main.pyでバッチ数で割るので、ここではバッチ平均は取らずに sum で返す。
        kl = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
        return kl

    def forward(self, x: torch.Tensor):
        """ELBO の各項を計算してフォワードパスを実行する。

        Args:
            x(batch, x_dim): 入力画像（DataLoader により平坦化済み、値域 [0, 1]）

        Returns:
            [elbo_kl, elbo_rec]: ELBO の2項
                elbo_kl(scalar):  KL 項（≤ 0）
                elbo_rec(scalar): 再構成項 — ベルヌーイ対数尤度（≤ 0）
            z(batch, z_dim): サンプリングされた潜在変数
            y(batch, x_dim): 再構成画像（値域 [0, 1]、decoder の Sigmoid 出力）

        参照: "Auto-Encoding Variational Bayes" Eq. (3), Appendix C.1
        """
        # TODO: encoder → reparametrization_trick → decoder の順に呼び出す。
        #      elbo_kl = -self.kld(mean, log_var) で KL 項と（符号に注意）、
        #      elbo_rec を計算して [elbo_kl, elbo_rec], z, y を返す。
        mean, log_var = self.encoder(x)
        z = self.reparametrization_trick(mean, log_var)
        y = self.decoder(z)
        elbo_kl = -self.kld(mean, log_var)
        # 再構成項の計算（ベルヌーイ対数尤度：ln L(p) = Σ x ln p + (1-x) ln (1-p)）

        # eps を足して log(0) を避ける
        # main.py でバッチ数で割るので、ここではバッチ平均は取らずに sum で返す。
        elbo_rec = torch.sum(
            x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        )
        return [elbo_kl, elbo_rec], z, y
