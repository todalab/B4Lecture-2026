import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    # sample2d_1.csv の処理
    df = pd.read_csv("../data/sample_logistics.csv", header=0)
    x1, x2=df["x1"], df["x2"]
    y = df["y"]



if __name__ == "__main__":
    main()