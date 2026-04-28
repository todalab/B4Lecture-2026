# -*- coding: utf-8 -*-
"""課題2 PDA LDA"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typing

matplotlib.rcParams["font.family"] = "sans-serif"

def make2d_scatter_figure(
    x: pd.Series,
    y: pd.Series,
    title: str
    ) -> None:
  """2次元散布図を作成する．"""
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.scatter(x, y, label="data")
  plt.legend(loc="upper left")
  plt.title(title)
  plt.ylabel("y")
  plt.xlabel("x")
  plt.savefig(f"output/{title}.png")
  plt.close()


def make2d_scatter_axis_figure(
    x: pd.Series,
    y: pd.Series,
    coefficients: np.array,
    ave_x:float,
    ave_y:float,
    title: str
    ) -> None:
  """主成分表示付きの2次元散布図を作成する．"""
  print(f"中心:\n({ave_x}, {ave_y})")
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.scatter(x, y, label="data")
  colors = ["tab:red", "tab:blue"]
  for i in range(2):
    direction = coefficients[:, i]
    slope = direction[1] / direction[0] if direction[0] != 0 else np.inf
    ax.axline(
      (ave_x, ave_y),
      slope=slope,
      color=colors[i],
      label=f"PC{i + 1}"
    )
    print(f"PC{i+1}\n slope={slope}")
  plt.legend(loc="upper left")
  plt.title(title)
  plt.ylabel("y")
  plt.xlabel("x")
  plt.savefig(f"output/{title}.png")
  plt.close()


def ave_center_2d(
    X:np.ndarray
) -> tuple[np.ndarray, float, float]:
  """2次元データを平均中心化する．"""
  ave_x = np.average(X[:,0])
  ave_y = np.average(X[:,1])
  ave = [ave_x, ave_y]
  # 平均中心化
  return (X - ave, ave_x, ave_y)


def pca_2d(
    X:np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float] :
  """2次元データのPCA．"""
  X_centered, ave_x, ave_y = ave_center_2d(X)
  # 共分散行列
  X_cov = np.cov(X_centered.T)
  print("X_cov:\n",X_cov)
  # 固有値、固有ベクトルを求める
  eig = np.linalg.eigh(X_cov)[0]
  eigvec = np.linalg.eigh(X_cov)[1]
  # 大きい順に並べ替え
  idx = np.argsort(eig)[::-1]
  eig = eig[idx]
  eigvec = eigvec[:, idx]
  print("idx\n", idx)
  print("eig:\n",eig)
  print("eigvec:\n",eigvec)
  # 寄与率
  cr = eig/sum(eig)
  print("寄与率:\n",cr)
  # 累積寄与率
  ccr = np.cumsum(cr)
  print("累積寄与率:\n", ccr)
  return (eig, eigvec, ave_x, ave_y)


def make3d_scatter_figure(
    x: pd.Series,
    y: pd.Series,
    z: pd.Series,
    title: str
    ) -> None:
  """3次元散布図を作成する．"""
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  ax.scatter(x, y, z, label="data")
  ax.legend(loc="upper left")
  ax.set_title(title)
  ax.set_zlabel("z")
  ax.set_ylabel("y")
  ax.set_xlabel("x")
  plt.savefig(f"output/{title}.png")
  plt.close()


def ave_center_3d(
    X:np.ndarray
) -> tuple[np.ndarray, float, float, float]:
  """3次元データを平均中心化する．"""
  ave_x = np.average(X[:,0])
  ave_y = np.average(X[:,1])
  ave_z = np.average(X[:,2])
  ave = [ave_x, ave_y, ave_z]
  # 平均中心化
  return (X - ave, ave_x, ave_y, ave_z)


def pca_3d(
    X:np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float] :
  """3次元データのPCA．"""
  X_centered, ave_x, ave_y, ave_z = ave_center_3d(X)
  # 共分散行列
  X_cov = np.cov(X_centered.T)
  print("X_cov:\n",X_cov)
  # 固有値、固有ベクトルを求める
  eig = np.linalg.eigh(X_cov)[0]
  eigvec = np.linalg.eigh(X_cov)[1]
  # 大きい順に並べ替え
  idx = np.argsort(eig)[::-1]
  eig = eig[idx]
  eigvec = eigvec[:, idx]
  print("idx\n", idx)
  print("eig:\n",eig)
  print("eigvec:\n",eigvec)
  # 寄与率
  cr = eig/sum(eig)
  print("寄与率:\n",cr)
  # 累積寄与率
  ccr = np.cumsum(cr)
  print("累積寄与率:\n", ccr)
  return (eig, eigvec, ave_x, ave_y, ave_z)


def make3d_scatter_axis_figure(
    x: pd.Series,
    y: pd.Series,
    z: pd.Series,
    coefficients: np.array,
    ave_x:float,
    ave_y:float,
    ave_z:float,
    title: str
    ) -> None:
  """主成分の軸表示付きの3次元散布図を作成する．"""
  print(f"中心:\n({ave_x}, {ave_y}, {ave_z})")
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  ax.scatter(x, y, z, label="data")
  colors = ["tab:red", "tab:blue", "tab:green"]
  axis_length = 1.5
  for i in range(3):
    direction = coefficients[:, i]
    slope_y = direction[1] / direction[0] if direction[0] != 0 else np.inf
    slope_z = direction[2] / direction[0] if direction[0] != 0 else np.inf
    norm = np.linalg.norm(direction)
    if norm == 0:
      continue
    unit_direction = direction / norm
    line_x = [ave_x - axis_length * unit_direction[0], ave_x + axis_length * unit_direction[0]]
    line_y = [ave_y - axis_length * unit_direction[1], ave_y + axis_length * unit_direction[1]]
    line_z = [ave_z - axis_length * unit_direction[2], ave_z + axis_length * unit_direction[2]]
    ax.plot(line_x, line_y, line_z, color=colors[i], label=f"PC{i + 1}")
    print(f"PC{i+1}\n slope_y={slope_y}, slope_z={slope_z}")
  ax.legend(loc="upper left")
  ax.set_title(title)
  ax.set_zlabel("z")
  ax.set_ylabel("y")
  ax.set_xlabel("x")
  plt.savefig(f"output/{title}.png")
  plt.close()


def PC_show(
    X: np.ndarray,
    eigvec: np.ndarray,
    title:str
) -> None:
  """主成分の計算と表示．第1・第2主成分スコアの2次元散布図．"""
  # 主成分への変換
  t1 = np.dot(X, eigvec[:,0])
  t2 = np.dot(X, eigvec[:,1])
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(t1, t2, label="data")
  ax.legend(loc="upper left")
  ax.set_title(title)
  ax.set_ylabel("t2")
  ax.set_xlabel("t1")
  plt.savefig(f"output/{title}.png")
  plt.close()


def ave_center_100d(
    X:np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """100次元データを平均中心化する．"""
  ave = np.average(X, axis=0)
  # 平均中心化
  return (X - ave, ave)


def show_cumulative_contribution_ratio_sample_100d(
    X:np.ndarray
) -> None:
  """100次元データの累積寄与率をグラフで表示する．"""
  X_centered, ave = ave_center_100d(X)
  # 共分散行列
  X_cov = np.cov(X_centered.T)
  # 固有値、固有ベクトルを求める
  eig = np.linalg.eigh(X_cov)[0]
  # 大きい順に並べ替え
  idx = np.argsort(eig)[::-1]
  eig = eig[idx]
  # 寄与率
  cr = eig/sum(eig)
  # 累積寄与率
  ccr = np.cumsum(cr)
  plt.figure()
  plt.plot(range(1, len(ccr) + 1), ccr, marker="o")
  plt.axhline(y=0.9, color="r", linestyle="--")
  plt.title("Cumulative Contribution Ratio")
  plt.xlabel("Number of Principal Components")
  plt.ylabel("Cumulative Contribution Ratio")
  plt.grid()
  plt.savefig("output/cumulative_contribution_ratio_100d.png")
  plt.close()

def pca_100d(
    X:np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] :
  """100次元データのPCA．2次元まで圧縮する．"""
  X_centered, ave = ave_center_100d(X)
  # 共分散行列
  X_cov = np.cov(X_centered.T)
  print("X_cov:\n",X_cov)
  # 固有値、固有ベクトルを求める
  eig = np.linalg.eigh(X_cov)[0]
  eigvec = np.linalg.eigh(X_cov)[1]
  # 大きい順に並べ替え
  idx = np.argsort(eig)[::-1]
  eig = eig[idx]
  eigvec = eigvec[:, idx]
  print("idx\n", idx)
  print("eig:\n",eig)
  print("eigvec:\n",eigvec)
  # 寄与率
  cr = eig/sum(eig)
  print("寄与率:\n",cr)
  # 累積寄与率
  ccr = np.cumsum(cr)
  print("累積寄与率:\n", ccr)
  # 累積寄与率が0.9以上になる最小の次元数を求める
  n_components = np.argmax(ccr >= 0.9) + 1
  print("累積寄与率が0.9以上になる最小の次元数:", n_components)
  return (eig, eigvec, ave)


def PC_show_pca_100d(
    X: np.ndarray,
    eigvec: np.ndarray,
    title:str
) -> None:
  """主成分の計算と表示．第1・第2主成分スコアの2次元散布図．"""
  # 主成分への変換
  t1 = np.dot(X, eigvec[:,0])
  t2 = np.dot(X, eigvec[:,1])
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(t1, t2, label="data")
  ax.legend(loc="upper left")
  ax.set_title(title)
  ax.set_ylabel("t2")
  ax.set_xlabel("t1")
  plt.savefig(f"output/{title}.png")
  plt.close()


def lda(
    X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """ldaを実施．"""
  # 行列をlabelごとに分離
  X0 = X[X[:,2]==0].reshape((-1,3))[:,0:2]
  X1 = X[X[:,2]==1].reshape((-1,3))[:,0:2]
  print("X0:\n", X0)
  print("X1:\n", X1)
  # クラスごとの平均ベクトルを計算
  ave_X0 = np.average(X0, axis=0)
  ave_X1 = np.average(X1, axis=0)
  print("ave_X0:\n", ave_X0)
  print("ave_X1:\n", ave_X1)
  # クラス内分散行列
  X0 = X0 - ave_X0
  X1 = X1 - ave_X1
  SW = np.zeros((2,2))
  for x in X0:
    SW += x.reshape((2,1)) @ x.reshape((2,1)).T
  for x in X1:
    SW += x.reshape((2,1)) @ x.reshape((2,1)).T
  print("SW:\n", SW)
  # クラス間分散行列
  SB = (ave_X0 - ave_X1).reshape((2,1)) @ (ave_X0 - ave_X1).reshape((2,1)).T
  print("SB:\n", SB)
  # 固有値問題を解く
  # 固有値、固有ベクトルを求める
  eig = np.linalg.eigh(np.linalg.inv(SW) @ SB)[0]
  eigvec = np.linalg.eigh(np.linalg.inv(SW) @ SB)[1]
  # 大きい順に並べ替え
  idx = np.argsort(eig)[::-1]
  eig = eig[idx]
  eigvec = eigvec[:, idx]
  print("idx\n", idx)
  print("eig:\n",eig)
  print("eigvec:\n",eigvec)
  return (eig, eigvec, ave_X0, ave_X1)


def make_lda_scatter_axis_figure(
    x: pd.Series,
    y: pd.Series,
    coefficients: np.array,
    ave_x: float,
    ave_y: float,
    title: str
    ) -> None:
  """LDAの軸表示付きの2次元散布図を作成する．
  軸の始点をクラス間の重心の中点に設定する。"""
  print(f"中心(中点):\n({ave_x}, {ave_y})")
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.scatter(x, y, label="data")
  colors = ["tab:red"]
  for i in range(1):
    direction = coefficients[:, i]
    slope = direction[1] / direction[0] if direction[0] != 0 else np.inf
    ax.axline(
      (ave_x, ave_y),
      slope=slope,
      color=colors[i],
      label=f"LD{i + 1}"
    )
    print(f"LD{i+1}\n slope={slope}")
  plt.legend(loc="upper left")
  plt.title(title)
  plt.ylabel("y")
  plt.xlabel("x")
  plt.savefig(f"output/{title}.png")
  plt.close()


def make_lda_1dprojection_figure(
    x: pd.Series,
    y: pd.Series,
    labels: pd.Series,
    ave_x: float,
    ave_y: float,
    coefficients: np.array,
    title: str
    ) -> None:
  """LDAの軸に射影した1次元散布図を作成する．"""
  X = np.column_stack((x.to_numpy(), y.to_numpy()))
  direction = coefficients[:, 0]
  norm = np.linalg.norm(direction)
  if norm == 0:
    raise ValueError("LDAの射影方向のノルムが0です")
  unit_direction = direction / norm
  projected = (X - np.array([ave_x, ave_y])) @ unit_direction
  label_values = labels.to_numpy()
  class0 = projected[label_values == 0]
  class1 = projected[label_values == 1]
  threshold = (np.average(class0) + np.average(class1)) / 2

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(class0, np.zeros_like(class0), label="class 0")
  ax.scatter(class1, np.ones_like(class1), label="class 1")
  ax.axvline(threshold, color="tab:red", linestyle="--", label=f"threshold={threshold:.3f}")
  ax.set_title(title)
  ax.set_xlabel("LDA axis value")
  ax.set_yticks([])
  ax.legend(loc="upper right")
  plt.savefig(f"output/{title}.png")
  plt.close()

if __name__ == "__main__":
  # CSVファイルからデータを読み込む
  sample2d_raw = pd.read_csv("../data/pca_2d.csv", header=None)
  sample3d_raw = pd.read_csv("../data/pca_3d.csv", header=None)
  sample100d_raw = pd.read_csv("../data/pca_100d.csv", header=None)
  sample2c_raw = pd.read_csv("../data/lda_2class.csv")


  # pca_2d
  print("Start PCA_2d")
  make2d_scatter_figure(sample2d_raw[0], sample2d_raw[1], "pca_2d scatter plot")
  sample2d = sample2d_raw.to_numpy()
  eig, eigvec, ave_x, ave_y = pca_2d(sample2d)
  make2d_scatter_axis_figure(sample2d_raw[0], sample2d_raw[1], eigvec, ave_x, ave_y, "pca_2d scatter plot with PCA axis")


  # pca_3d
  print("Start PCA_3d")
  make3d_scatter_figure(sample3d_raw[0], sample3d_raw[1], sample3d_raw[2], "pca_3d scatter plot")
  sample3d = sample3d_raw.to_numpy()
  eig, eigvec, ave_x, ave_y, ave_z = pca_3d(sample3d)
  make3d_scatter_axis_figure(sample3d_raw[0], sample3d_raw[1], sample3d_raw[2], eigvec, ave_x, ave_y, ave_z, "pca_3d scatter plot with PCA axis")
  sample3d_centered, ave_x, ave_y, ave_z = ave_center_3d(sample3d)
  PC_show(sample3d_centered, eigvec, "pca_3d PCA scatter plot")


  # pca_100d
  print("Start PCA_100d")
  sample100d = sample100d_raw.to_numpy()
  eig, eigvec, ave = pca_100d(sample100d)
  sample100d_centered, ave = ave_center_100d(sample100d)
  PC_show_pca_100d(sample100d_centered, eigvec, "pca_100d PCA scatter plot")
  show_cumulative_contribution_ratio_sample_100d(sample100d)


  # lda
  print("Start LDA")
  sample_2c = sample2c_raw.to_numpy()
  eig, eigvec, ave_X0, ave_X1 = lda(sample_2c)
  mid = (ave_X0 + ave_X1) / 2
  ave_x_mid, ave_y_mid = mid[0], mid[1]
  make_lda_scatter_axis_figure(sample2c_raw["x1"], sample2c_raw["x2"], eigvec, ave_x_mid, ave_y_mid, "lda scatter plot with LDA axis")
  make_lda_1dprojection_figure(sample2c_raw["x1"], sample2c_raw["x2"], sample2c_raw["label"], ave_x_mid, ave_y_mid, eigvec, "lda scatter plot with LDA axis (1d projection)")