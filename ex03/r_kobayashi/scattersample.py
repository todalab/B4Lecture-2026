"""散布図の導出."""

import matplotlib.pyplot as plt
import pandas as pd

data1 = pd.read_csv("../../ex3/data/data1.csv", header=None)
data1.describe()
data2 = pd.read_csv("../../ex3/data/data2.csv", header=None)
data2.describe()
data3 = pd.read_csv("../../ex3/data/data3.csv", header=None)
data3.describe()

fig, ax = plt.subplots(2, 2, figsize=(6, 6))

ax[0, 0].scatter(data1[0], data1[1], s=5)
ax[0, 0].set_title("data1")
ax[0, 0].set_xlabel("x")
ax[0, 0].set_ylabel("y")
ax[0, 0].grid(True)

ax[0, 1].scatter(data2[0], data2[1], s=5)
ax[0, 1].set_title("data2")
ax[0, 1].set_xlabel("x")
ax[0, 1].set_ylabel("y")
ax[0, 1].grid(True)

ax[1, 0].scatter(data3[0], data3[1], s=5)
ax[1, 0].set_title("data3")
ax[1, 0].set_xlabel("x")
ax[1, 0].set_ylabel("y")
ax[1, 0].grid(True)

plt.savefig("scatter_plot.png")
