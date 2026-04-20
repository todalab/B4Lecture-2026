"""
answer/02_gmm.py  ―  3-2 GMM 実装
EMアルゴリズムによる混合ガウスモデルのモジュール．
他のスクリプトから `from answer.gmm import GMM` のようにインポートして使う．

直接実行した場合は動作確認用のサンプルを走らせる．

Usage:
    python answer/02_gmm.py
"""

import numpy as np


# ============================================================
# 多変量正規分布の確率密度関数
# ============================================================

def gaussian_pdf(X, mu, Sigma):
    """多変量正規分布の確率密度を計算する．

    Parameters
    ----------
    X     : ndarray, shape (N, d)
    mu    : ndarray, shape (d,)
    Sigma : ndarray, shape (d, d)

    Returns
    -------
    ndarray, shape (N,)
    """
    d = X.shape[1]
    diff = X - mu                          # (N, d)
    Sigma_inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    det = max(det, 1e-300)                 # アンダーフロー対策
    coeff = 1.0 / np.sqrt((2 * np.pi) ** d * det)
    # einsum で (X-mu)^T Sigma^{-1} (X-mu) をベクトル化
    mahal = np.einsum("ni,ij,nj->n", diff, Sigma_inv, diff)
    return coeff * np.exp(-0.5 * mahal)


# ============================================================
# GMM クラス
# ============================================================

class GMM:
    """EMアルゴリズムによる混合ガウスモデル．

    Parameters
    ----------
    K           : int   クラスター数
    max_iter    : int   最大反復回数
    tol         : float 収束判定の閾値（対数尤度の変化量）
    reg         : float 共分散行列の正則化パラメータ（対角成分に加算）
    random_state: int or None  乱数シード
    """

    def __init__(self, K, max_iter=300, tol=1e-6, reg=1e-6, random_state=None):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.rng = np.random.default_rng(random_state)

        # フィット後に参照できる属性
        self.pi = None      # 混合係数 (K,)
        self.mu = None      # 平均      (K, d)
        self.Sigma = None   # 共分散    (K, d, d)
        self.r_ = None      # 負担率    (N, K)
        self.log_likelihoods_ = None  # 各反復での対数尤度

    # ----------------------------------------------------------
    # 初期化
    # ----------------------------------------------------------

    def _init_params(self, X):
        """パラメータをランダムに初期化する．
        平均: データからランダムに K 点選ぶ
        共分散: 全データの共分散行列（正則化済み）
        混合係数: 一様
        """
        N, d = X.shape
        self.pi = np.ones(self.K) / self.K
        idx = self.rng.choice(N, self.K, replace=False)
        self.mu = X[idx].copy()                              # (K, d)
        cov0 = np.cov(X.T) + self.reg * np.eye(d)
        self.Sigma = np.stack([cov0.copy() for _ in range(self.K)])  # (K, d, d)

    # ----------------------------------------------------------
    # Eステップ
    # ----------------------------------------------------------

    def _e_step(self, X):
        """負担率 r[n, k] を計算する．

        r[n, k] = pi_k * N(x_n | mu_k, Sigma_k)
                  / sum_j pi_j * N(x_n | mu_j, Sigma_j)

        Returns
        -------
        r : ndarray, shape (N, K)
        """
        N = X.shape[0]
        # 各クラスターの重み付き確率密度を計算
        r = np.column_stack([
            self.pi[k] * gaussian_pdf(X, self.mu[k], self.Sigma[k])
            for k in range(self.K)
        ])                                   # (N, K)

        # 正規化（行方向の和が 1 になるように）
        r_sum = r.sum(axis=1, keepdims=True)
        r_sum = np.where(r_sum < 1e-300, 1e-300, r_sum)
        return r / r_sum

    # ----------------------------------------------------------
    # Mステップ
    # ----------------------------------------------------------

    def _m_step(self, X, r):
        """負担率 r をもとにパラメータを更新する．"""
        N, d = X.shape
        Nk = r.sum(axis=0)                   # 各クラスターの有効サンプル数 (K,)

        for k in range(self.K):
            rk = r[:, k]                     # (N,)

            # 平均の更新
            self.mu[k] = (rk[:, np.newaxis] * X).sum(axis=0) / Nk[k]

            # 共分散の更新
            diff = X - self.mu[k]            # (N, d)
            self.Sigma[k] = (rk[:, np.newaxis] * diff).T @ diff / Nk[k]
            self.Sigma[k] += self.reg * np.eye(d)   # 正則化

        # 混合係数の更新
        self.pi = Nk / N

    # ----------------------------------------------------------
    # 対数尤度
    # ----------------------------------------------------------

    def _log_likelihood(self, X):
        """対数尤度 sum_n log( sum_k pi_k * N(x_n | mu_k, Sigma_k) ) を計算する．"""
        # 各クラスターの重み付き確率密度の行列 (N, K)
        weighted = np.column_stack([
            self.pi[k] * gaussian_pdf(X, self.mu[k], self.Sigma[k])
            for k in range(self.K)
        ])
        # クラスターについて和をとってから log
        mixture_pdf = weighted.sum(axis=1)           # (N,)
        mixture_pdf = np.where(mixture_pdf < 1e-300, 1e-300, mixture_pdf)
        return np.sum(np.log(mixture_pdf))

    # ----------------------------------------------------------
    # フィット
    # ----------------------------------------------------------

    def fit(self, X, verbose=True):
        """EMアルゴリズムで GMM を学習する．

        Parameters
        ----------
        X       : ndarray, shape (N, d)
        verbose : bool  収束情報を表示するか

        Returns
        -------
        self
        """
        self._init_params(X)
        self.log_likelihoods_ = []
        prev_ll = -np.inf

        for i in range(self.max_iter):
            r = self._e_step(X)
            self._m_step(X, r)
            ll = self._log_likelihood(X)
            self.log_likelihoods_.append(ll)

            if abs(ll - prev_ll) < self.tol:
                if verbose:
                    print(f"  収束: {i + 1} 反復  (最終対数尤度: {ll:.4f})")
                break
            prev_ll = ll
        else:
            if verbose:
                print(f"  最大反復数 {self.max_iter} に到達  (最終対数尤度: {ll:.4f})")

        # 最終的な負担率を保存
        self.r_ = self._e_step(X)
        return self

    # ----------------------------------------------------------
    # 予測・情報量基準
    # ----------------------------------------------------------

    def predict(self, X):
        """各データ点を負担率最大のクラスターに割り当てる．"""
        r = self._e_step(X)
        return np.argmax(r, axis=1)

    def n_params(self, d):
        """自由パラメータ数を返す．
        (K-1) 混合係数 + K*d 平均 + K*d*(d+1)/2 共分散
        """
        return (self.K - 1) + self.K * d + self.K * d * (d + 1) // 2

    def aic(self, X):
        """AIC = -2 * log_likelihood + 2 * n_params"""
        ll = self._log_likelihood(X)
        p = self.n_params(X.shape[1])
        return -2 * ll + 2 * p

    def bic(self, X):
        """BIC = -2 * log_likelihood + n_params * log(N)"""
        ll = self._log_likelihood(X)
        p = self.n_params(X.shape[1])
        return -2 * ll + p * np.log(X.shape[0])


# ============================================================
# 動作確認（直接実行したとき）
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.multivariate_normal([0, 0], np.eye(2), 100),
        rng.multivariate_normal([5, 5], np.eye(2), 100),
    ])

    print("GMM 動作確認 (K=2, 2クラスター生成データ)")
    gmm = GMM(K=2, random_state=0)
    gmm.fit(X)

    labels = gmm.predict(X)
    print(f"クラスター割り当て: {np.bincount(labels)}")
    print(f"推定平均:\n{gmm.mu}")
    print(f"AIC: {gmm.aic(X):.2f},  BIC: {gmm.bic(X):.2f}")
