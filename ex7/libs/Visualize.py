# -*- coding: utf-8 -*-
"""VAE の学習結果を可視化するクラス。実装済み・変更不要。"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import ArtistAnimation


class Visualize:
    """VAE の可視化クラス。

    生成される画像:
        reconstruction/  : 入力画像と再構成画像の比較
        latent_space/    : 潜在空間の散布図（クラス色分け）
        lattice_point/   : 格子点からの生成画像（z_dim=2 のみ）
        walkthrough/     : 潜在空間の補間 GIF アニメーション（z_dim=2 のみ）
    """

    def __init__(self, z_dim, h_dim, dataloader_test, model, device):
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.dataloader_test = dataloader_test
        self.model = model
        self.device = device

    def createDirectories(self):
        for d in ["reconstruction", "latent_space", "lattice_point", "walkthrough"]:
            os.makedirs(f"./images/{d}", exist_ok=True)

    def reconstruction(self):
        """入力画像と再構成画像を並べて保存する。"""
        for batch_idx, (x, _) in enumerate(self.dataloader_test):
            fig, axes = plt.subplots(2, 10, figsize=(20, 4))
            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])

            for i, im in enumerate(x.view(-1, 28, 28)[:10]):
                axes[0][i].imshow(im, "gray")

            _, _, y = self.model(x.to(self.device))
            y = y.cpu().detach().numpy().reshape(-1, 28, 28)
            for i, im in enumerate(y[:10]):
                axes[1][i].imshow(im, "gray")

            fig.savefig(f"./images/reconstruction/z{self.z_dim}_{batch_idx}.png")
            plt.close(fig)

    def latent_space(self):
        """潜在空間をクラスごとに色分けして散布図として保存する。"""
        cm = plt.get_cmap("tab10")
        for batch_idx, (x, labels) in enumerate(self.dataloader_test):
            _, z, _ = self.model(x.to(self.device))
            z = z.cpu().detach().numpy()

            fig_plot, ax_plot = plt.subplots(figsize=(9, 9))
            fig_scatter, ax_scatter = plt.subplots(figsize=(9, 9))

            for k in range(10):
                idx = np.where(labels.cpu().numpy() == k)[0]
                ax_plot.plot(z[idx, 0], z[idx, 1], "o", ms=4, color=cm(k), label=str(k))
                ax_scatter.scatter(z[idx, 0], z[idx, 1], marker=f"${k}$", color=cm(k))

            ax_plot.legend(loc="upper right")
            fig_plot.savefig(
                f"./images/latent_space/z{self.z_dim}_{batch_idx}_plot.png"
            )
            fig_scatter.savefig(
                f"./images/latent_space/z{self.z_dim}_{batch_idx}_scatter.png"
            )
            plt.close(fig_plot)
            plt.close(fig_scatter)

    def lattice_point(self):
        """2D 潜在空間の格子点からデコードした画像を保存する（z_dim=2 専用）。"""
        n = 25
        xs = np.linspace(-2, 2, n)
        ys = np.linspace(-2, 2, n)
        zx, zy = np.meshgrid(xs, ys)
        Z = (
            torch.tensor(np.stack([zx, zy], axis=-1), dtype=torch.float)
            .to(self.device)
            .reshape(-1, self.z_dim)
        )
        imgs = self.model.decoder(Z).cpu().detach().numpy().reshape(-1, 28, 28)

        fig, axes = plt.subplots(n, n, figsize=(9, 9))
        for i in range(n):
            for j in range(n):
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
                axes[i][j].imshow(imgs[n * (n - 1 - i) + j], "gray")
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(f"./images/lattice_point/z{self.z_dim}.png")
        plt.close(fig)

    def walkthrough(self):
        """潜在空間上の2点間を線形補間して GIF アニメーションを生成する（z_dim=2 専用）。"""
        step = 50
        endpoints = [
            (torch.tensor([-3.0, 0.0]), torch.tensor([3.0, 0.0])),
            (torch.tensor([-3.0, 3.0]), torch.tensor([3.0, -3.0])),
            (torch.tensor([0.0, 3.0]), torch.tensor([0.0, -3.0])),
            (torch.tensor([3.0, 3.0]), torch.tensor([-3.0, -3.0])),
        ]
        for n, (z1, z2) in enumerate(endpoints):
            z_path = torch.stack(
                [z1 * (1 - t / step) + z2 * (t / step) for t in range(step)]
            ).to(self.device)
            frames = (
                self.model.decoder(z_path).cpu().detach().numpy().reshape(-1, 28, 28)
            )

            fig, ax = plt.subplots(figsize=(4, 4))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_xticks([])
            ax.set_yticks([])
            images = [[ax.imshow(im, "gray")] for im in frames]
            anim = ArtistAnimation(
                fig, images, interval=100, blit=True, repeat_delay=1000
            )
            anim.save(f"./images/walkthrough/z{self.z_dim}_{n}.gif", writer="pillow")
            plt.close(fig)
