# -*- coding: utf-8 -*-
"""課題2 PDA LDA"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "sans-serif"


# CSVファイルからデータを読み込む
sample2d = pd.read_csv("../data/pca_2d.csv")
sample3d = pd.read_csv("../data/pca_3d.csv")
sample100d = pd.read_csv("../data/pca_100d.csv")
sample2c = pd.read_csv("../data/lda_2class.csv")

