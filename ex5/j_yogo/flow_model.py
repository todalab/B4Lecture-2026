"""
flow_model.py
CNN Encoder + Normalizing Flow (Real-NVP) による異常検知.

学習：正常音の log-likelihood を最大化
推論：log-likelihood が低い → 異常スコアが高い
"""

import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
#  CNN Encoder                                                         #
# ------------------------------------------------------------------ #

class CNNEncoder(nn.Module):
    """
    log-Mel スペクトログラム → 埋め込みベクトル

    Input : (B, 1, n_mels, T)
    Output: (B, emb_dim)
    """

    def __init__(self, channels: list, emb_dim: int):
        super().__init__()
        layers = []
        in_ch  = 1
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.cnn  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(channels[-1], emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        return x


# ------------------------------------------------------------------ #
#  Real-NVP Affine Coupling Layer                                      #
# ------------------------------------------------------------------ #

class AffineCouplingLayer(nn.Module):
    """
    入力 z を前半 z_a / 後半 z_b に分割し、
    z_b を z_a で条件付けて変換する.

    z_b' = z_b * exp(s(z_a)) + t(z_a)
    z_a' = z_a  （変化なし）

    → 逆変換：z_b = (z_b' - t(z_a)) * exp(-s(z_a))
    → log|det J| = sum(s(z_a))
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim_a = dim // 2
        self.dim_b = dim - self.dim_a

        # s と t を同じネットワークから出力
        self.net = nn.Sequential(
            nn.Linear(self.dim_a, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.dim_b * 2),  # s と t を結合
        )
        # s の出力を小さく初期化（学習初期の安定性）
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, z: torch.Tensor) -> tuple:
        """
        Returns:
            z_out    : (B, dim) 変換後
            log_det_J: (B,) log|det Jacobian|
        """
        z_a, z_b = z[:, :self.dim_a], z[:, self.dim_a:]
        st       = self.net(z_a)
        s, t     = st[:, :self.dim_b], st[:, self.dim_b:]
        s        = torch.tanh(s)  # s を [-1, 1] に制限して数値安定化

        z_b_out   = z_b * torch.exp(s) + t
        log_det_J = s.sum(dim=1)

        return torch.cat([z_a, z_b_out], dim=1), log_det_J

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """変換後の z から元の z を復元する（逆変換）."""
        z_a, z_b = z[:, :self.dim_a], z[:, self.dim_a:]
        st       = self.net(z_a)
        s, t     = st[:, :self.dim_b], st[:, self.dim_b:]
        s        = torch.tanh(s)

        z_b_orig = (z_b - t) * torch.exp(-s)
        return torch.cat([z_a, z_b_orig], dim=1)


# ------------------------------------------------------------------ #
#  Real-NVP                                                            #
# ------------------------------------------------------------------ #

class RealNVP(nn.Module):
    """
    複数の AffineCouplingLayer を積み重ねた Normalizing Flow.
    奇数層と偶数層で分割方向を交互に変えることで全次元を変換する.
    """

    def __init__(self, dim: int, n_layers: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim, hidden_dim)
            for _ in range(n_layers)
        ])
        # 奇数層では次元を入れ替えて分割方向を変える
        self.perms = [
            torch.arange(dim) if i % 2 == 0
            else torch.cat([torch.arange(dim // 2, dim), torch.arange(dim // 2)])
            for i in range(n_layers)
        ]

    def forward(self, z: torch.Tensor) -> tuple:
        """
        Returns:
            z_out        : (B, dim) 標準正規分布に近い変換後ベクトル
            total_log_det: (B,) 全層の log|det J| の合計
        """
        total_log_det = torch.zeros(z.size(0), device=z.device)
        for layer, perm in zip(self.layers, self.perms):
            z = z[:, perm.to(z.device)]
            z, log_det = layer(z)
            total_log_det += log_det
        return z, total_log_det

    def log_likelihood(self, z: torch.Tensor) -> torch.Tensor:
        """
        入力 z の log-likelihood を返す.

        log p(z) = log p_N(f(z)) + log|det J|
        p_N: 標準正規分布

        Returns: (B,) 各サンプルの log-likelihood
        """
        z_out, log_det = self.forward(z)

        # 標準正規分布の log-likelihood
        dim       = z_out.shape[1]
        log_p_z   = -0.5 * (z_out ** 2 + torch.log(torch.tensor(2 * torch.pi))).sum(dim=1)

        return log_p_z + log_det  # (B,)


# ------------------------------------------------------------------ #
#  AnomalyDetector: Encoder + Flow をまとめたモデル                    #
# ------------------------------------------------------------------ #

class AnomalyDetector(nn.Module):
    """
    CNN Encoder で音声を埋め込み、Real-NVP で正常分布を学習する.

    学習： -log_likelihood を最小化（正常音の確率を最大化）
    推論：  anomaly_score = -log_likelihood（高いほど異常）
    """

    def __init__(self, channels: list, emb_dim: int,
                 flow_layers: int, flow_hidden_dim: int):
        super().__init__()
        self.encoder = CNNEncoder(channels, emb_dim)
        self.flow    = RealNVP(emb_dim, flow_layers, flow_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: (B,) log-likelihood（高いほど正常）
        """
        z  = self.encoder(x)
        ll = self.flow.log_likelihood(z)
        return ll

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: (B,) 異常スコア（高いほど異常）
        """
        return -self.forward(x)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """学習用Loss: 負の平均 log-likelihood"""
        return -self.forward(x).mean()