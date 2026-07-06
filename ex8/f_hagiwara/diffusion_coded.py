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
from torch.utils.tensorboard import SummaryWriter
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
        noise_schedule: Literal["linear"],
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
            alpha = 1.0 - beta
            alpha_prod = alpha.cumprod(dim=0)
            self.register_buffer("beta", beta)
            self.register_buffer("alpha", alpha)
            self.register_buffer("alpha_prod", alpha_prod)

        # TensorBoard用
        self.writer = SummaryWriter(log_dir="runs/diffusion")
        self.global_step = 0

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

        memo:
            x_t=√(α) x_0+√(1-α) 𝜀
        """
        # DONE
        # alpha_prodはalpha.cumprodによって作られたハイパーパラメータαの累積積を格納した1次元テンソル
        # viewは指定したサイズに変形する関数。先頭引数を-1にすることで自動的に調整される
        # 今回の場合(B,1,1,1)になる
        # これにより(B,C,H,W)の画像とブロードキャストして計算できるように
        alpha_prod_t = self.alpha_prod[t].view(-1, 1, 1, 1)

        x_t = torch.sqrt(alpha_prod_t) * x0 + torch.sqrt(1.0 - alpha_prod_t) * noise
        return x_t

    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """逆拡散過程（Reverse process）: x_t から x_{t-1} を推定する。

        Args:
            x (B, C, H, W): ノイズ付き画像 x_t
            t (B,):          タイムステップ（全要素が同じ値）

        Returns:
            x_prev (B, C, H, W): 1ステップ前の画像 x_{t-1}

        参照: "Denoising Diffusion Probabilistic Models" Algorithm 2, Eq. (11)
        """
        # DONE
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_prod_t = self.alpha_prod[t].view(-1, 1, 1, 1)

        # PyTorchでは model(input) の形で model.forward(input)が呼ばれる
        # selfはDiffusionModelのインスタンス
        # self(x,t)はノイズ付き画像とタイムステップを与えself.forwardの出力である推定ノイズを返す
        noise_pred = self(x, t)

        mu = (1.0 / torch.sqrt(alpha_t)) * (
            x - beta_t / torch.sqrt(1.0 - alpha_prod_t) * noise_pred
        )

        # t=0のときのみノイズを加えない
        # 生成ではバッチ内の全画像が同じ時刻なので一つを見ればよい
        if t[0] == 0:
            return mu

        noise = torch.randn_like(x)

        return mu + torch.sqrt(beta_t) * noise

    def training_step(self, images: torch.Tensor) -> torch.Tensor:
        """1バッチの損失を計算する。

        Args:
            images (B, C, H, W): クリーン画像

        Returns:
            loss (scalar): 訓練損失

        参照: "Denoising Diffusion Probabilistic Models" Algorithm 1

        memo:
            1. 画像を読み込む
            2. 1からTまでのある時刻tを選ぶ
            3. 正規分布に従うノイズを作る
            4. q_sampleで作成したノイズ付き画像を作成
            5. UNetでノイズ推定
            6. 本物のノイズとの差をLossにする
        """
        # DONE

        device = images.device
        batch_size = images.size(0)

        # ランダムなタイムステップ
        t = torch.randint(
            0,
            self.num_timesteps,
            (batch_size,),
            device=device,
        )

        # 正解ノイズ
        noise = torch.randn_like(images)

        # ノイズ付き画像
        noisy_images = self.q_sample(images, t, noise)

        # UNetによるノイズ推定
        noise_pred = self(noisy_images, t)

        # ノイズ推定誤差
        loss = self.criterion(noise_pred, noise)

        # TensorBoardへ記録
        self.writer.add_scalar("train_loss", loss.item(), self.global_step)
        self.global_step += 1

        return loss

    def generate(self, num_timesteps: int, shape: tuple) -> torch.Tensor:
        """サンプルを生成する。（実装済み・変更不要）"""
        device = self.alpha.device
        x = torch.randn(shape, device=device)
        for step in tqdm(range(num_timesteps - 1, -1, -1)):
            t = torch.full((x.size(0),), step, dtype=torch.long, device=device)
            x = self.p_sample(x, t)
        return x
