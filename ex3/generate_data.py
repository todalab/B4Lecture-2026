"""
generate_data.py
data/ ディレクトリに data1.csv, data2.csv, data3.csv を生成する．
"""

import numpy as np
import os

os.makedirs("data", exist_ok=True)

rng = np.random.default_rng(42)

# --------------------------------------------------
# data1: 3クラスター，比較的明瞭に分離
# --------------------------------------------------
means1 = [[-4, 0], [4, 0], [0, 5]]
covs1 = [
    [[1.0, 0.3], [0.3, 0.8]],
    [[0.8, -0.2], [-0.2, 1.0]],
    [[1.2, 0.0], [0.0, 0.6]],
]
sizes1 = [120, 100, 80]

data1 = np.vstack([
    rng.multivariate_normal(m, c, n)
    for m, c, n in zip(means1, covs1, sizes1)
])
rng.shuffle(data1)
np.savetxt("data/data1.csv", data1, delimiter=",")
print(f"data1.csv: {data1.shape}  (K=3 想定)")

# --------------------------------------------------
# data2: 4クラスター，うち2つが近接
# --------------------------------------------------
means2 = [[-5, -3], [-5, 3], [3, -2], [5, 3]]
covs2 = [
    [[1.5, 0.5], [0.5, 0.8]],
    [[0.9, -0.4], [-0.4, 1.2]],
    [[1.0, 0.6], [0.6, 1.0]],
    [[0.7, 0.0], [0.0, 1.5]],
]
sizes2 = [90, 90, 100, 80]

data2 = np.vstack([
    rng.multivariate_normal(m, c, n)
    for m, c, n in zip(means2, covs2, sizes2)
])
rng.shuffle(data2)
np.savetxt("data/data2.csv", data2, delimiter=",")
print(f"data2.csv: {data2.shape}  (K=4 想定，うち2つが近接)")

# --------------------------------------------------
# data3: 2クラスター，強い相関・細長い形状
# --------------------------------------------------
means3 = [[-3, -3], [3, 3]]
covs3 = [
    [[3.0, 2.5], [2.5, 2.5]],
    [[2.5, -2.0], [-2.0, 2.0]],
]
sizes3 = [150, 120]

data3 = np.vstack([
    rng.multivariate_normal(m, c, n)
    for m, c, n in zip(means3, covs3, sizes3)
])
rng.shuffle(data3)
np.savetxt("data/data3.csv", data3, delimiter=",")
print(f"data3.csv: {data3.shape}  (K=2 想定，相関が強い楕円形状)")

print("\nデータ生成完了: data/ ディレクトリを確認してください．")
