"""ビタビアルゴリズム."""

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

# 𝑎𝑖𝑗 遷移確率：状態 𝑠𝑖 → 𝑠𝑗 へ移る確率
# 𝑏(𝑠𝑖,y𝑘)出力確率（emission）：状態 𝑠𝑖 が観測 y𝑘 を出す確率


def Viterbi(observations, transition_matrix, emission_matrix, initial_distribution):
    """viterbiアルゴリズム.

    Args:
        observations (ndarray of shape (T, )): 観測系列 𝑂=(𝑦1,𝑦2,...,𝑦𝑇)
        transition_matrix (ndarray of shape (M, M)): 遷移確率 𝑎𝑖𝑗
        emission_matrix (ndarray of shape (M, K)): 出力確率 𝑏(𝑠𝑗,𝑦𝑡) (Kは観測しうる記号の数)
        initial_distribution (ndarray of shape (M,)): 初期確率 𝜋𝑖

    Returns:
        viterbi_prob (ndarray of shape (T, M)): 最大対数尤度
        path (list): 最尤状態系列
    """
    n_timesteps = observations.shape[0]  # 観測の長さT
    n_states = transition_matrix.shape[0]  # 状態数M
    EPS = 1e-12
    viterbi_prob = np.zeros(
        (n_timesteps, n_states)
    )  # 時間tに状態sjをとる最大確率𝛿𝑡(𝑖)を格納する行列（T×M）
    backtrack = np.zeros((n_timesteps, n_states), dtype=int)

    viterbi_prob[0, :] = np.log(initial_distribution + EPS) + np.log(
        emission_matrix[:, observations[0]] + EPS
    )  # 初期化 𝛿1(𝑖)=𝜋𝑖𝑏(𝑠𝑖,𝑦1)

    for t in range(1, n_timesteps):
        for j in range(n_states):
            viterbi_prob[t, j] = np.max(
                viterbi_prob[t - 1] + np.log(transition_matrix[:, j] + EPS)
            ) + np.log(
                emission_matrix[j, observations[t]] + EPS
            )  # 再帰 𝛿𝑡(𝑗)=max⁡𝑖[𝛿𝑡−1(𝑖)⋅𝑎𝑖𝑗]⋅𝑏𝑗(𝑜𝑡)
            backtrack[t, j] = np.argmax(
                viterbi_prob[t - 1] + np.log(transition_matrix[:, j] + EPS)
            )
    # print(f"viterbi_prob = {viterbi_prob}")
    # print(f"backtrack = {backtrack}")
    last_state = np.argmax(viterbi_prob[-1])
    path = [int(last_state)]
    current_state = last_state
    for t in range(n_timesteps - 1, 0, -1):
        current_state = backtrack[t, current_state]
        path.append(int(current_state))
    path.reverse()
    # print(f"path = {path}")
    return viterbi_prob, path


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
            viterbi_mat, path = Viterbi(obs, A, B, PI)
            P_star = np.max(viterbi_mat[-1])  # 一番尤もらしい経路を選択
            likelihoods[k, n] = P_star

    # print(likelihoods.shape)
    # print(type(likelihoods))
    print(f"path = {path}")
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
        resultname = "data1v_result.png"
    elif select == 2:
        data = pickle.load(open("../../ex4/data/data2.pickle", "rb"))
        resultname = "data2v_result.png"
    elif select == 3:
        data = pickle.load(open("../../ex4/data/data3.pickle", "rb"))
        resultname = "data3v_result.png"
    elif select == 4:
        data = pickle.load(open("../../ex4/data/data4.pickle", "rb"))
        resultname = "data4v_result.png"

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
