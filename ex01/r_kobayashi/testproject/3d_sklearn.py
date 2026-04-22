# 3次元配列

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# # 1. データの作成
path_3d = "../../../ex1/data/sample3d.csv"

with open(path_3d) as f:
    sample3 = np.loadtxt(path_3d, delimiter=",", skiprows=1)

X1 = sample3[:, 0]
X2 = sample3[:, 1]
y = sample3[:, 2]

# 特徴量を結合
X = np.column_stack((X1, X2))

# 2. 線形回帰モデルを学習
model = LinearRegression()
model.fit(X, y)

# 3. 回帰平面の描写用メッシュグリッド
X1_grid, X2_grid = np.meshgrid(
    np.linspace(X1.min(), X1.max(), 20), np.linspace(X2.min(), X2.max(), 20)
)
X_grid = np.column_stack((X1_grid.ravel(), X2_grid.ravel()))
y_pred_grid = model.predict(X_grid).reshape(X1_grid.shape)

# 平面でない場合を考える

# 学習率の設定
lr = LinearRegression()

# 多項式特徴量の生成 (今回は2次の多項式と3次の多項式を考える)
quadratic = PolynomialFeatures(degree=2, include_bias=False)
cubic = PolynomialFeatures(degree=3, include_bias=False)
# 特徴量を変換
X_quadratic = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)
# 2次の多項式回帰モデルの学習
lr.fit(X_quadratic, y)
y_pred_quadratic = lr.predict(X_quadratic)
# 3次の多項式回帰モデルの学習
lr.fit(X_cubic, y)
y_pred_cubic = lr.predict(X_cubic)

mse_quadratic = mean_squared_error(y, y_pred_quadratic)
mse_cubic = mean_squared_error(y, y_pred_cubic)
r2_quadratic = r2_score(y, y_pred_quadratic)
r2_cubic = r2_score(y, y_pred_cubic)
print(f"2次の多項式回帰 - MSE: {mse_quadratic:.2f}, R^2: {r2_quadratic:.2f}")
print(f"3次の多項式回帰 - MSE: {mse_cubic:.2f}, R^2: {r2_cubic:.2f}")
# 2次の多項式回帰 - MSE: XXXX, R^2: XXXX
# 3次の多項式回帰 - MSE: XXXX, R^2: XXXX


# グリッド生成用の元データを取得
x1 = X[:, 0]
x2 = X[:, 1]

# 描画用グリッド作成
x1_range = np.linspace(x1.min(), x1.max(), 30)
x2_range = np.linspace(x2.min(), x2.max(), 30)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]

# 2次元特徴量に変換
X_grid_quad = quadratic.transform(X_grid)
X_grid_cubic = cubic.transform(X_grid)

# 2つの別々のモデルを使う
lr_quad = LinearRegression()
lr_quad.fit(X_quadratic, y)
z_quad = lr_quad.predict(X_grid_quad).reshape(x1_grid.shape)

lr_cub = LinearRegression()
lr_cub.fit(X_cubic, y)
z_cub = lr_cub.predict(X_grid_cubic).reshape(x1_grid.shape)

# 3Dプロット
fig = plt.figure(figsize=(18, 8))

# 線形回帰のプロット
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.scatter(X1, X2, y, color="blue", label="data", alpha=0.5)
ax.plot_surface(X1_grid, X2_grid, y_pred_grid, color="red", alpha=0.6)
ax.set_title("Multiple Linear Regression (2 Features)")
ax.set_xlabel("Feature 1 (X1)")
ax.set_ylabel("Feature 2 (X2)")
ax.set_zlabel("Target (y)", labelpad=0)

# 2次のプロット
ax1 = fig.add_subplot(1, 3, 2, projection="3d")
ax1.scatter(x1, x2, y, color="blue", label="data", alpha=0.5)
ax1.plot_surface(x1_grid, x2_grid, z_quad, color="orange", alpha=0.6)
ax1.set_title("2nd-degree Polynomial Regression")
ax1.set_xlabel("Feature 1 (X1)")
ax1.set_ylabel("Feature 2 (X2)")
ax1.set_zlabel("Target (y)", labelpad=0)

# 3次のプロット
ax2 = fig.add_subplot(1, 3, 3, projection="3d")
ax2.scatter(x1, x2, y, color="blue", label="data", alpha=0.5)
ax2.plot_surface(x1_grid, x2_grid, z_cub, color="green", alpha=0.6)
ax2.set_title("3rd-degree Polynomial Regression")
ax2.set_xlabel("Feature 1 (X1)")
ax2.set_ylabel("Feature 2 (X2)")
ax2.set_zlabel("Target (y)", labelpad=0)

plt.tight_layout()
# plt.show()
plt.savefig("3d_sklearn.png")
print("3d_sklearn.pngを出力しました。")
