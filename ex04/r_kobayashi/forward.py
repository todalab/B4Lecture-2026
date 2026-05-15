"""Forwardアルゴリズム."""

import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# print(data)

# 𝑠𝑖 i 番目の状態
# 𝑦𝑡 t 時刻目の観測
# 𝑂=(𝑦1,…,𝑦𝑇) 観測系列

# 𝜋𝑖 初期状態確率：最初に状態 𝑠𝑖 にいる確率
# 𝑎𝑖𝑗 遷移確率：状態 𝑠𝑖 → 𝑠𝑗 へ移る確率
# 𝑏(𝑠𝑖,y𝑘)出力確率（emission）：状態 𝑠𝑖 が観測 y𝑘 を出す確率


# 𝛼𝑡(𝑗)前向き確率：時刻 t に状態 𝑠𝑗 にいて、そこまでの観測が出る確率
# 𝛼1(𝑗)初期化：𝜋𝑗𝑏(𝑠𝑗,𝑦1), ∀𝑖=1,2,…,𝑀
# 𝛼𝑡(𝑗)再帰：(∑𝑖𝑎𝑖𝑗𝛼𝑡−1(𝑖))𝑏(𝑠𝑗,𝑦𝑡), ∀𝑗=1,2,…,𝑀


# https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.ap-northeast-1.amazonaws.com%2F0%2F1166480%2F4f8295b5-aeb0-08d9-44bd-548d8175415c.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=3c2e2b4b05ad26d3842f101824ef488e
def forward(observations, transition_matrix, emission_matrix, initial_distribution):
    """Forwardアルゴリズム.

    Args:
        observations (ndarray of shape (T, )): 観測系列 𝑂=(𝑦1,𝑦2,...,𝑦𝑇)
        transition_matrix (ndarray of shape (M, M)): 遷移確率 𝑎𝑖𝑗
        emission_matrix (ndarray of shape (M, K)): 出力確率 𝑏(𝑠𝑗,𝑦𝑡) (Kは観測しうる記号の数)
        initial_distribution (ndarray of shape (M,)): 初期確率 𝜋𝑖

    Returns:
        ndarray of shape (T, M): 各系列の尤度
    """
    n_timesteps = observations.shape[0]  # 観測の長さT
    n_states = transition_matrix.shape[0]  # 状態数M
    forward_prob = np.zeros(
        (n_timesteps, n_states)
    )  # 時間tに状態sjをとる確率𝛼𝑡(𝑗)を格納する行列（T×M）

    forward_prob[0, :] = (
        initial_distribution * emission_matrix[:, observations[0]]
    )  # 初期化 𝛼1(𝑖)=𝜋𝑖𝑏(𝑠𝑖,𝑦1)

    for t in range(1, n_timesteps):
        for j in range(n_states):
            forward_prob[t, j] = (
                forward_prob[t - 1]
                @ transition_matrix[:, j]
                # 𝛼𝑡−1(𝑖) のベクトルと𝑎𝑖𝑗 の列ベクトルの内積の総和
                * emission_matrix[j, observations[t]]
            )  # 再帰 𝛼𝑡(𝑗)=(∑𝑖𝑎𝑖𝑗𝛼𝑡−1(𝑖))𝑏(𝑠𝑗,𝑦𝑡)

    return forward_prob


# data1 の例
# 'output': ndarray (shape=(300,20), dtype=int64)
# 𝑂𝑛=(𝑦1,…,𝑦20) が 300 個


def compute_likelihoods(data):
    """尤度の計算.

    Args:
        data (dict): モデルや出力などのデータ一覧

    Returns:
        likelihoods (ndarray of shape (H, N)): 各モデル、各系列における尤度 (H はモデル数、N は観測系列数)
    """
    # データ取得
    outputs = data["output"]  # (300,20)
    A_all = data["models"]["A"]  # (3,4,4)
    B_all = data["models"]["B"]  # (3,4,6)
    PI_all = data["models"]["PI"]  # (3,4,1)

    n_models = A_all.shape[0]  # モデルの数
    n_sequences = outputs.shape[0]  # モデルにおける観測した長さ

    likelihoods = np.zeros((n_models, n_sequences))  # 尤度の初期化

    for k in range(n_models):
        A = A_all[k]
        B = B_all[k]
        PI = PI_all[k].squeeze()
        for n in range(n_sequences):
            obs = outputs[n]
            likelihoods[k, n] = sum(forward(obs, A, B, PI)[-1])

    # print(likelihoods.shape)
    # print(type(likelihoods))
    return likelihoods
    # 各 HMM モデルそれぞれについて、観測系列の尤度を並べたnumpy行列が得られる


def model_predict(data):
    """モデル予測.

    Args:
        data (dict): モデルや出力などのデータ一覧

    Returns:
        cm (ndarray of shape (H, H)): 予測ラベルと正解ラベルの混合行列
        accuracy (float): 正解率
    """
    likelihoods = compute_likelihoods(data)  # 尤度の計算
    predicted_models = np.argmax(likelihoods, axis=0)  # モデルの予測
    # print(predicted_models)
    answer_all = data["answer_models"]  # 正解ラベル
    # print(answer_all)
    cm = confusion_matrix(
        answer_all, predicted_models
    )  # 予測モデルと正解モデルの混合行列
    print(f"confusion matrix = \n{cm}")
    accuracy = accuracy_score(answer_all, predicted_models)
    print(f"accuracy = {accuracy}")
    return cm, accuracy


def main(select):
    """メイン関数.

    Args:
        select (int): データの選択

    Returns:
        cm (ndarray of shape (H, H)): 予測ラベルと正解ラベルの混合行列
        accuracy (float): 正解率
        execution_time (float): 実行時間
    """
    if select == 1:
        data = pickle.load(open("../../ex4/data/data1.pickle", "rb"))
        resultname = "data1_result.png"
    elif select == 2:
        data = pickle.load(open("../../ex4/data/data2.pickle", "rb"))
        resultname = "data2_result.png"
    elif select == 3:
        data = pickle.load(open("../../ex4/data/data3.pickle", "rb"))
        resultname = "data3_result.png"
    elif select == 4:
        data = pickle.load(open("../../ex4/data/data4.pickle", "rb"))
        resultname = "data4_result.png"

    start = time.perf_counter()
    cm, accuracy = model_predict(data)
    end = time.perf_counter()
    execution_time = end - start
    print(f"実行時間: {execution_time:.6f} 秒")

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    ax.spines[:].set_visible(True)
    plt.title(resultname)
    plt.savefig(resultname)
    # plt.show()
    return cm, accuracy, execution_time


if __name__ == "__main__":
    try:
        arg = int(sys.argv[1])
    except IndexError:
        print("引数が必要です")
    main(arg)
