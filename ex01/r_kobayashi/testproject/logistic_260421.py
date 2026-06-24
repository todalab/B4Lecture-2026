import math

import matplotlib.pyplot as plt
import numpy as np

path_lo = "../../../ex1/data/sample_logistic.csv"

with open(path_lo) as f:
    sample_lo = np.loadtxt(path_lo, delimiter=",", skiprows=1)
d = sample_lo.shape[0]
x1 = np.array(sample_lo[:, 0])
x2 = np.array(sample_lo[:, 1])
y = np.array(sample_lo[:, 2])
# one = np.ones(d).T
# x = np.stack([x1,x2,one], axis=1)
x = np.stack([x1, x2], axis=1)

print(x.shape)

# x = x.reshape(-1,1)
y = y.reshape(-1, 1)

# パラメータ初期化
w = np.random.randn(x.shape[1], 1)
b = np.random.randn()
learning_rate = 0.01
epochs = 10000

likelihood_list = []
loss_list = []
accuracy_list = []
threshold = 0.5

# 学習ループ
for epoch in range(epochs):
    # print(x,w)
    # bprint(x.shape, type(x), w.shape, type(w))
    z = x @ w + b
    prediction = 1 / (1 + np.exp(-z))  # sigmoid関数
    # print(prediction.shape,type(prediction),y.shape,type(y))
    error = prediction - y

    # 勾配計算
    gradient_w = x.T @ error / len(x)
    gradient_b = error.mean()
    # print(gradient_w.shape, type(gradient_w), gradient_b.shape, type(gradient_b))

    # パラメータ更新
    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b
    # print("prediction.shape: " + str(prediction.shape))

    # 1000エポックごとに尤度・損失・精度を表示
    if epoch % 100 == 0:
        likelihood = 0
        for i in range(len(x)):
            if y[i] == 1:
                # likelihood = likelihood * 1 / (1 + np.exp(-w.T @ x[i]))
                likelihood = likelihood + math.log(prediction[i, 0])
            elif y[i] == 0:
                # likelihood = likelihood * (1 - 1 / (1 + np.exp(-w.T @ x[i])))
                likelihood = likelihood + math.log(1 - prediction[i, 0])
        # print(likelihood)
        likelihood_list.append(likelihood)

        loss = -y * np.log(prediction + 1e-15) - (1 - y) * np.log(
            1 - prediction + 1e-15
        )
        # print(loss.mean())
        loss_list.append(loss.mean())

        predictions_binary = (prediction >= threshold).astype(int)
        accuracy = np.mean(predictions_binary == y)
        accuracy_list.append(accuracy)

print(type(loss_list))


fig = plt.figure(figsize=(5, 12))
fig.suptitle("logistic")

plt.subplot(3, 1, 1)
plt.plot(likelihood_list)
plt.title("Likelihood")
plt.subplot(3, 1, 2)
plt.plot(loss_list)
plt.title("Loss")
plt.subplot(3, 1, 3)
plt.plot(accuracy_list)
plt.title("Accuracy")
plt.savefig("logistic_loss.png")
print("logistic_loss.pngを出力しました。")
