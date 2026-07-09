# -*- coding: utf-8 -*-
"""
Ex8 B4講義 — Diffusion Model（拡散モデル）
このファイルは DiffusionModel の実装を行うためのコードです。
以下のメソッドの #TODO と記載されている箇所を実装してください:
    - DiffusionModel.q_sample
    - DiffusionModel.p_sample
    - DiffusionModel.training_step
"""

from typing import Any, Dict, List, Literal, Optional

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
            )  # デフォルトの設定だと付加されるノイズはだんだん大きくなる
            alpha = 1.0 - beta
            alpha_prod = alpha.cumprod(dim=0)  # そのインデックスまでのαの累積積を計算
            self.register_buffer(
                "beta", beta
            )  # 勾配減少方向への更新は受けないが、モデルの状態としてセーブ＆ロードされる
            self.register_buffer("alpha", alpha)
            self.register_buffer("alpha_prod", alpha_prod)
        elif noise_schedule == "cosine":
            # cosineスケジュール
            start_angle = torch.arccos(
                torch.sqrt(torch.tensor(noise_schedule_kwargs["start"]))
            )
            end_angle = torch.arccos(
                torch.sqrt(torch.tensor(noise_schedule_kwargs["end"]))
            )
            t = torch.linspace(0, 1, num_timesteps)
            beta = torch.cos(start_angle + t * (end_angle - start_angle))
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
        # 論文中でバー付きのαは累積積を表すので、self.alpha_prodを使用する
        # 係数は (B,) なので (B, 1, 1, 1) に reshape して画像とブロードキャストする
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
        # TODO
        # forward メソッドを使用してノイズ推定モデルを呼び出す
        noise_pred = self(x, t)
        # 係数は (B,) なので (B, 1, 1, 1) に reshape して画像とブロードキャストする
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        alpha_prod_t = self.alpha_prod[t].view(-1, 1, 1, 1)
        # t=0 のときはノイズを加えない（決定的）
        z = torch.randn_like(x)
        z[t == 0] = 0.0
        # 論文中では実験的にσ^2 = β_t としているので、ここでは sqrt(β_t) を使用する
        x_prev = (1.0 / torch.sqrt(alpha_t)) * (
            x - ((beta_t / torch.sqrt(1.0 - alpha_prod_t)) * noise_pred)
        ) + torch.sqrt(beta_t) * z
        return x_prev

    def ddim_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM の逆過程: x_t から x_{t_prev} を推定する（サブサンプリング対応）。

        DDPM の p_sample が隣接ステップ (t → t-1) しか扱えないのに対し、
        DDIM は任意の t_prev (< t) へ直接ジャンプできるため、ステップを
        サブサンプリングして少ないステップ数で高速に生成できる。

        Args:
            x (B, C, H, W):  ノイズ付き画像 x_t
            t (B,):          現在のタイムステップ（全要素が同じ値）
            t_prev (B,):     次のタイムステップ（t より小さい。全要素が同じ値）
            eta (float):     確率性の度合い。0 で決定的（DDIM）、1 で DDPM 相当

        Returns:
            x_prev (B, C, H, W): タイムステップ t_prev の画像

        参照: "Denoising Diffusion Implicit Models" Eq. (12)
        """
        noise_pred = self(x, t)
        # 係数は (B,) なので (B, 1, 1, 1) に reshape してブロードキャストする
        alpha_prod_t = self.alpha_prod[t].view(-1, 1, 1, 1)
        # t_prev < 0（最終ステップ）のときは ᾱ_{-1} = 1 とする
        alpha_prod_t_prev = torch.where(
            (t_prev >= 0).view(-1, 1, 1, 1),
            self.alpha_prod[t_prev.clamp(min=0)].view(-1, 1, 1, 1),
            torch.ones_like(alpha_prod_t),
        )

        # x_0 の予測値（predicted x_0）
        pred_x0 = (x - torch.sqrt(1.0 - alpha_prod_t) * noise_pred) / torch.sqrt(
            alpha_prod_t
        )

        # σ_t: eta=0 で 0（決定的）、eta=1 で DDPM の分散に一致
        sigma_t = eta * torch.sqrt(
            (1.0 - alpha_prod_t_prev)
            / (1.0 - alpha_prod_t)
            * (1.0 - alpha_prod_t / alpha_prod_t_prev)
        )

        # x_t を指す方向（direction pointing to x_t）
        dir_xt = torch.sqrt(1.0 - alpha_prod_t_prev - sigma_t**2) * noise_pred

        # t_prev = -1（最終ステップ）ではノイズを加えない
        z = torch.randn_like(x)
        z[t_prev < 0] = 0.0

        x_prev = torch.sqrt(alpha_prod_t_prev) * pred_x0 + dir_xt + sigma_t * z
        return x_prev

    def training_step(self, images: torch.Tensor) -> torch.Tensor:
        """1バッチの損失を計算する。

        Args:
            images (B, C, H, W): クリーン画像

        Returns:
            loss (scalar): 訓練損失

        参照: "Denoising Diffusion Probabilistic Models" Algorithm 1
        """
        # TODO
        # x_0はどうやって生成する？N(0,1)で固定？
        # uniformは一様分布を意味する
        # バッチサイズ分のランダムなタイムステップ t を一様サンプリング
        t = torch.randint(
            0, self.num_timesteps, (images.size(0),), device=images.device
        )
        # 画像と同じ形状のノイズ ε ~ N(0, I) をサンプリング
        noise = torch.randn_like(images)
        # 順拡散過程でノイズ付き画像 x_t を作成
        x_t = self.q_sample(images, t, noise)
        # forward メソッドを使用してノイズ推定モデルで ε_θ(x_t, t) を推定
        noise_pred = self(x_t, t)
        # 損失を計算
        loss = self.criterion(noise_pred, noise)
        return loss

    def generate(
        self,
        num_timesteps: int,
        shape: tuple,
        use_ddim: bool = False,
        ddim_steps: Optional[int] = None,
        ddim_eta: float = 0.0,
        gif_path: Optional[str] = None,
        gif_every_n_steps: int = 1,
        gif_nrow: Optional[int] = None,
        gif_duration: int = 50,
    ) -> torch.Tensor:
        """サンプルを生成する。

        Args:
            num_timesteps (int):              逆拡散過程のステップ数 T
            shape (tuple):                    生成する画像の shape (B, C, H, W)
            use_ddim (bool):                  True で DDIM サンプリングを使用する
            ddim_steps (Optional[int]):       DDIM 使用時のサブサンプリングステップ数
                                              （None の場合は num_timesteps を使用）
            ddim_eta (float):                 DDIM の確率性。0 で決定的、1 で DDPM 相当
            gif_path (Optional[str]):         指定すると逆拡散過程を GIF として保存する
            gif_every_n_steps (int):          GIF に何ステップごとにフレームを追加するか
            gif_nrow (Optional[int]):         GIF グリッドの1行あたりの画像数
                                              （None の場合はバッチサイズの平方根）
            gif_duration (int):               GIF の各フレームの表示時間 [ms]

        Returns:
            x (B, C, H, W): 生成された画像 x_0
        """
        device = self.alpha.device
        x = torch.randn(shape, device=device)

        frames: List[torch.Tensor] = []
        record_gif = gif_path is not None

        if use_ddim:
            # タイムステップ列を一様確率で除去（サブサンプリング）する。
            # 例: T=500, ddim_steps=50 なら [0, 10, 20, ..., 490] を使用
            if ddim_steps is None:
                ddim_steps = num_timesteps
            step_indices = torch.linspace(
                0, num_timesteps - 1, ddim_steps, device=device
            ).long()
            # 大きいステップから小さいステップへ（逆順）
            timesteps = list(reversed(step_indices.tolist()))
            # 各ステップの1つ前の遷移先（最後は -1 = x_0 相当）
            timesteps_prev = timesteps[1:] + [-1]

            for i, (step, step_prev) in enumerate(zip(tqdm(timesteps), timesteps_prev)):
                t = torch.full((x.size(0),), step, dtype=torch.long, device=device)
                t_prev = torch.full(
                    (x.size(0),), step_prev, dtype=torch.long, device=device
                )
                x = self.ddim_sample(x, t, t_prev, eta=ddim_eta)
                if record_gif and (i % gif_every_n_steps == 0 or step_prev < 0):
                    frames.append(x.detach().cpu())
        else:
            for step in tqdm(range(num_timesteps - 1, -1, -1)):
                t = torch.full((x.size(0),), step, dtype=torch.long, device=device)
                x = self.p_sample(x, t)
                # 指定ステップごと、および最終ステップで現在の画像を記録
                if record_gif and (step % gif_every_n_steps == 0 or step == 0):
                    frames.append(x.detach().cpu())

        if record_gif:
            self._save_gif(frames, gif_path, gif_nrow, gif_duration)

        return x

    @staticmethod
    def _save_gif(
        frames: List[torch.Tensor],
        gif_path: str,
        nrow: Optional[int],
        duration: int,
    ) -> None:
        """逆拡散過程のフレームリストを GIF として保存する。

        Args:
            frames (List[Tensor]): 各要素が (B, C, H, W) の画像テンソル
            gif_path (str):        保存先パス
            nrow (Optional[int]):  グリッド1行あたりの画像数
            duration (int):        各フレームの表示時間 [ms]
        """
        import math
        import os

        import numpy as np
        from PIL import Image
        from torchvision.utils import make_grid

        batch_size = frames[0].size(0)
        if nrow is None:
            nrow = max(1, int(math.sqrt(batch_size)))

        pil_frames = []
        for frame in frames:
            # [-1, 1] → [0, 1] に正規化してからグリッド化
            img = (frame + 1) / 2
            img = img.clamp(0, 1)
            grid = make_grid(img, nrow=nrow)
            # (C, H, W) → (H, W, C) の uint8 配列へ変換
            array = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(array))

        out_dir = os.path.dirname(gif_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
        )
