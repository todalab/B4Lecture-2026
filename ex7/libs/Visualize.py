# -*- coding: utf-8 -*-
"""VAE の学習結果を可視化するクラス。"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import ArtistAnimation
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


class Visualize:
    """VAE の可視化クラス。

    生成される画像:
        reconstruction/  : 入力画像と再構成画像の比較
        latent_space/    : 潜在空間の散布図（クラス色分け）
        lattice_point/   : 格子点からの生成画像
                           （z_dim=2 は潜在空間、z_dim>2 は t-SNE 2D 空間）
        walkthrough/     : 潜在空間の補間 GIF アニメーション
                           （z_dim=2 は潜在空間、z_dim>2 は t-SNE 2D 空間）
    """

    def __init__(self, z_dim, h_dim, dataloader_test, model, device):
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.dataloader_test = dataloader_test
        self.model = model
        self.device = device
        self._tsne_cache = None

    def createDirectories(self):
        for d in ["reconstruction", "latent_space", "lattice_point", "walkthrough"]:
            os.makedirs(f"./images/{d}", exist_ok=True)

    def reconstruction(self, tag=""):
        """入力画像と再構成画像を並べて保存する。

        Parameters
        ----------
        tag : str
            ファイル名に付与する識別子（例: "best", "final"）。
        """
        suffix = f"_{tag}" if tag else ""
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

            fig.savefig(
                f"./images/reconstruction/z{self.z_dim}{suffix}_{batch_idx}.png"
            )
            plt.close(fig)

    def latent_space(self, tag=""):
        """潜在空間をクラスごとに色分けして散布図として保存する。

        Parameters
        ----------
        tag : str
            ファイル名に付与する識別子（例: "best", "final"）。
        """
        suffix = f"_{tag}" if tag else ""
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
                f"./images/latent_space/z{self.z_dim}{suffix}_{batch_idx}_plot.png"
            )
            fig_scatter.savefig(
                f"./images/latent_space/z{self.z_dim}{suffix}_{batch_idx}_scatter.png"
            )
            plt.close(fig_plot)
            plt.close(fig_scatter)

    def lattice_point(self):
        """格子点からデコードした画像を保存する。

        z_dim=2 のときは潜在空間そのものを、z_dim>2 のときは t-SNE で
        2 次元に落とした空間を格子状に走査してデコードする。
        """
        if self.z_dim == 2:
            self._lattice_point_2d()
        else:
            self._lattice_point_tsne()

    def walkthrough(self):
        """2 点間を線形補間して GIF アニメーションを生成する。

        z_dim=2 のときは潜在空間上で、z_dim>2 のときは t-SNE 2D 空間上で
        補間し、逆写像で潜在ベクトルに戻してからデコードする。
        """
        if self.z_dim == 2:
            self._walkthrough_2d()
        else:
            self._walkthrough_tsne()

    def _lattice_point_2d(self):
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

    def _walkthrough_2d(self):
        """潜在空間上の2点間を線形補間して GIF を生成する（z_dim=2 専用）。"""
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

    # ------------------------------------------------------------------
    # z_dim > 2 用：t-SNE による 2 次元可視化
    # ------------------------------------------------------------------
    def _collect_latents(self, max_samples=3000):
        """テストデータをエンコードし、潜在ベクトル（平均 μ）とラベルを集める。"""
        self.model.eval()
        zs, ys = [], []
        count = 0
        with torch.no_grad():
            for x, labels in self.dataloader_test:
                mean, _ = self.model.encoder(x.to(self.device))
                zs.append(mean.cpu().numpy())
                ys.append(labels.cpu().numpy())
                count += x.size(0)
                if count >= max_samples:
                    break
        Z = np.concatenate(zs)[:max_samples]
        y = np.concatenate(ys)[:max_samples]
        return Z, y

    def _prepare_tsne(self):
        """t-SNE 埋め込みを計算してキャッシュする。

        Returns
        -------
        E : np.ndarray, shape (N, 2)
            t-SNE による 2 次元埋め込み。
        Z : np.ndarray, shape (N, z_dim)
            元の潜在ベクトル（逆写像の参照に使う）。
        labels : np.ndarray, shape (N,)
            各サンプルのクラスラベル。
        """
        if self._tsne_cache is not None:
            return self._tsne_cache

        Z, labels = self._collect_latents()
        perplexity = min(30, max(5, (len(Z) - 1) // 3))
        tsne = TSNE(
            n_components=2,
            init="pca",
            perplexity=perplexity,
            random_state=42,
        )
        E = tsne.fit_transform(Z)
        self._tsne_cache = (E, Z, labels)
        return self._tsne_cache

    def _inverse_map(self, query, E, Z, k=10):
        """t-SNE 2D 空間の点を潜在ベクトルに逆写像する。

        t-SNE は逆変換を持たないため、t-SNE 埋め込み E と元の潜在ベクトル Z の
        対応から、k 近傍の逆距離加重 (IDW) 平均で潜在ベクトルを推定する。
        """
        k = min(k, len(E))
        nn = NearestNeighbors(n_neighbors=k).fit(E)
        dist, idx = nn.kneighbors(query)
        weights = 1.0 / (dist + 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        # (M, k) x (M, k, z_dim) -> (M, z_dim)
        z_query = np.einsum("mk,mkd->md", weights, Z[idx])
        return z_query

    def _decode_np(self, z_np):
        """numpy の潜在ベクトルをデコードして 28x28 画像配列を返す。"""
        z = torch.tensor(z_np, dtype=torch.float, device=self.device)
        with torch.no_grad():
            imgs = self.model.decoder(z).cpu().numpy()
        return imgs.reshape(-1, 28, 28)

    def _lattice_point_tsne(self):
        """t-SNE 2D 空間を格子状に走査し、逆写像してデコードした画像を保存する。"""
        E, Z, _ = self._prepare_tsne()
        n = 25
        xs = np.linspace(E[:, 0].min(), E[:, 0].max(), n)
        ys = np.linspace(E[:, 1].min(), E[:, 1].max(), n)
        gx, gy = np.meshgrid(xs, ys)
        grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
        z_grid = self._inverse_map(grid, E, Z)
        imgs = self._decode_np(z_grid)

        fig, axes = plt.subplots(n, n, figsize=(9, 9))
        for i in range(n):
            for j in range(n):
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
                axes[i][j].imshow(imgs[n * (n - 1 - i) + j], "gray")
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(f"./images/lattice_point/z{self.z_dim}_tsne.png")
        plt.close(fig)
        print(f"t-SNE lattice_point → ./images/lattice_point/z{self.z_dim}_tsne.png")

    def _walkthrough_tsne(self):
        """t-SNE 2D 空間でクラス重心間を補間し、逆写像してデコードした GIF を生成する。"""
        E, Z, labels = self._prepare_tsne()
        step = 50

        # 各クラスの t-SNE 上の重心を計算
        centroids = {}
        for c in range(10):
            idx = np.where(labels == c)[0]
            if len(idx) > 0:
                centroids[c] = E[idx].mean(axis=0)

        available = sorted(centroids.keys())
        # 重心間を結ぶ補間ペア（存在するクラスから 4 本を選ぶ）
        candidate_pairs = [(0, 1), (2, 3), (4, 7), (8, 9)]
        pairs = [
            (a, b) for a, b in candidate_pairs if a in centroids and b in centroids
        ]
        if not pairs and len(available) >= 2:
            pairs = [(available[0], available[-1])]

        for n, (a, b) in enumerate(pairs):
            e1, e2 = centroids[a], centroids[b]
            path2d = np.stack(
                [e1 * (1 - t / step) + e2 * (t / step) for t in range(step)]
            )
            z_path = self._inverse_map(path2d, E, Z)
            frames = self._decode_np(z_path)

            fig, ax = plt.subplots(figsize=(4, 4))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_xticks([])
            ax.set_yticks([])
            images = [[ax.imshow(im, "gray")] for im in frames]
            anim = ArtistAnimation(
                fig, images, interval=100, blit=True, repeat_delay=1000
            )
            anim.save(
                f"./images/walkthrough/z{self.z_dim}_tsne_{n}.gif", writer="pillow"
            )
            plt.close(fig)
        print(f"t-SNE walkthrough → ./images/walkthrough/z{self.z_dim}_tsne_*.gif")
