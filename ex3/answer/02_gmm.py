"""
answer/02_gmm.py  ―  3-2 GMM 実装（動作確認スクリプト）

GMM の実装本体は answer/gmm.py にある．
このスクリプトはその動作確認として，人工データで学習・予測を行う．

Usage:
    python answer/02_gmm.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from answer.gmm import GMM

rng = np.random.default_rng(0)

# 2クラスターの人工データ
X = np.vstack([
    rng.multivariate_normal([0, 0], np.eye(2), 100),
    rng.multivariate_normal([5, 5], np.eye(2), 100),
])

print("=== GMM 動作確認（K=2, 2クラスター生成データ）===")
gmm = GMM(K=2, random_state=0)
gmm.fit(X)

labels = gmm.predict(X)
print(f"\nクラスター割り当て（サイズ）: {np.bincount(labels)}")
print(f"推定混合係数: {gmm.pi.round(3)}")
print(f"推定平均:\n{gmm.mu.round(3)}")
print(f"\nAIC: {gmm.aic(X):.2f}")
print(f"BIC: {gmm.bic(X):.2f}")
print(f"パラメータ数 (d=2, K=2): {gmm.n_params(d=2)}")
