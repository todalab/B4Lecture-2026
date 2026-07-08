# -*- coding: utf-8 -*-
"""
Ex8 B4講義 — Diffusion Model（拡散モデル）
このファイルは DiffusionModel の実装を行うためのコードです。
以下のメソッドの #TODO と記載されている箇所を実装してください:
    - DiffusionModel.q_sample
    - DiffusionModel.p_sample
    - DiffusionModel.training_step
"""

from typing import Any, Dict, Literal

import torch
import torch.nn as nn
from tqdm import tqdm


class DiffusionModel(nn.Module):
    """Denoising Diffusion Probabilistic Model (DDPM).

    Args:
        model (nn.Module):                      ノイズ推定モデル（例: UNet2DModel）
        criterion (nn.Module):                  損失関数
        num_timesteps (int):                    拡散過程のタイムステップ数 T
        noise_schedule (Literal["linear"]):     ノイズスケジュールの種類
        noise_schedule_kwargs (Dict[str, Any]): ノイズスケジュールの引数（start, end）

    Functions:
        __init__:                              初期化・α, β, ᾱ バッファを登録（実装済み）
        q_sample(x0, t, noise):               順拡散過程 x_t ~ q(x_t|x_0)
        p_sample(x, t):                       逆拡散過程 x_{t-1} ~ p_θ(x_{t-1}|x_t)
        training_step(images):                1バッチの損失を計算
        generate(num_timesteps, shape):       サンプル生成（実装済み）
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        num_timesteps: int,
        noise_schedule: Literal["linear", "cosine"],
        noise_schedule_kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the diffusion model."""
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.num_timesteps = num_timesteps

        if noise_schedule == "linear":
            beta = torch.linspace(
                noise_schedule_kwargs["start"],
                noise_schedule_kwargs["end"],
                num_timesteps,
            )
        elif noise_schedule == "cosine":
            # 参照: Nichol & Dhariwal, "Improved Denoising Diffusion
            # Probabilistic Models" Eq. (17)
            s = noise_schedule_kwargs.get("s", 0.008)
            t = torch.linspace(0, num_timesteps, num_timesteps + 1) / num_timesteps
            alpha_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            beta = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)

        alpha = 1.0 - beta
        alpha_prod = alpha.cumprod(dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_prod", alpha_prod)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ノイズ推定モデルを呼び出す。（実装済み・変更不要）

        Args:
            x (B, C, H, W): ノイズ付き画像 x_t
            t (B,):          タイムステップ

        Returns:
            noise_pred (B, C, H, W): 推定ノイズ ε_θ(x_t, t)
        """
        return self.model(x, t).sample

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """順拡散過程（Forward process）: x_0 に t ステップ分のノイズを加える。

        Args:
            x0 (B, C, H, W):    クリーン画像 x_0
            t (B,):             タイムステップ
            noise (B, C, H, W): ノイズ ε

        Returns:
            x_t (B, C, H, W): ノイズ付き画像

        参照: "Denoising Diffusion Probabilistic Models" Eq. (4)
        """
        # TODO
        alpha_prod_t = self.alpha_prod[t].view(-1, 1, 1, 1)
        sqrt_alpha_prod_t = alpha_prod_t.sqrt()
        sqrt_one_minus_alpha_prod_t = (1.0 - alpha_prod_t).sqrt()
        return sqrt_alpha_prod_t * x0 + sqrt_one_minus_alpha_prod_t * noise

    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """逆拡散過程（Reverse process）: x_t から x_{t-1} を推定する。

        Args:
            x (B, C, H, W): ノイズ付き画像 x_t
            t (B,):          タイムステップ（全要素が同じ値）

        Returns:
            x_prev (B, C, H, W): 1ステップ前の画像 x_{t-1}

        参照: "Denoising Diffusion Probabilistic Models" Algorithm 2, Eq. (11)
        """
        # TODO
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        alpha_prod_t = self.alpha_prod[t].view(-1, 1, 1, 1)

        noise_pred = self.forward(x, t)

        mean = (1.0 / alpha_t.sqrt()) * (
            x - (beta_t / (1.0 - alpha_prod_t).sqrt()) * noise_pred
        )

        z = torch.randn_like(x)
        z[t == 0] = 0.0

        return mean + beta_t.sqrt() * z

    def training_step(self, images: torch.Tensor) -> torch.Tensor:
        """1バッチの損失を計算する。

        Args:
            images (B, C, H, W): クリーン画像

        Returns:
            loss (scalar): 訓練損失

        参照: "Denoising Diffusion Probabilistic Models" Algorithm 1
        """
        # TODO
        device = images.device
        batch_size = images.size(0)

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(images)

        x_t = self.q_sample(images, t, noise)
        noise_pred = self.forward(x_t, t)

        return self.criterion(noise_pred, noise)

    def generate(self, num_timesteps: int, shape: tuple) -> torch.Tensor:
        """サンプルを生成する。（実装済み・変更不要）"""
        device = self.alpha.device
        x = torch.randn(shape, device=device)
        for step in tqdm(range(num_timesteps - 1, -1, -1)):
            t = torch.full((x.size(0),), step, dtype=torch.long, device=device)
            x = self.p_sample(x, t)
        return x
