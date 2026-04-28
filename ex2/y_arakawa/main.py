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
  eig = np.linalg.eig(X_cov)[0]
  eigvec = np.linalg.eig(X_cov)[1]
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
  eig = np.linalg.eig(X_cov)[0]
  eigvec = np.linalg.eig(X_cov)[1]
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
  x_range = [-1.5, 1.5]
  y_range = [-1.5, 1.5]
  z_range = [-1.5, 1.5]
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  ax.scatter(x, y, z, label="data")
  colors = ["tab:red", "tab:blue", "tab:green"]
  for i in range(3):
    direction = coefficients[:, i]
    slope_y = direction[1] / direction[0] if direction[0] != 0 else np.inf
    slope_z = direction[2] / direction[0] if direction[0] != 0 else np.inf
    ax.quiver(
      ave_x, ave_y, ave_z,
      direction[0], direction[1], direction[2],
      color=colors[i],
      label=f"PC{i + 1}"
    )
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


# def 


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


