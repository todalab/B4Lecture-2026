"""
HMM課題用データ生成スクリプト

生成されるデータ:
  data1.pickle : Left-to-Right HMM, 少数クラス (k=3モデル)
  data2.pickle : Ergodic HMM,       少数クラス (k=3モデル)
  data3.pickle : Left-to-Right HMM, 多数クラス (k=5モデル)
  data4.pickle : Ergodic HMM,       多数クラス (k=5モデル)
"""

import os
import pickle
import numpy as np


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    """各行を正規化して確率行列にする"""
    return mat / mat.sum(axis=-1, keepdims=True)


def make_left_to_right_A(n_states: int, rng: np.random.Generator) -> np.ndarray:
    """Left-to-Right HMM の状態遷移行列を生成する（上三角確率行列）.

    最終状態は自己遷移のみ（吸収状態）にして確率が 1 になるようにする．
    """
    A = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        # i 以降の状態にのみ遷移できる
        weights = rng.random(n_states - i)
        A[i, i:] = weights / weights.sum()
    # 最終状態は自己遷移のみ（吸収状態）
    A[-1, -1] = 1.0
    return A


def make_ergodic_A(n_states: int, rng: np.random.Generator) -> np.ndarray:
    """Ergodic HMM の状態遷移行列を生成する（全結合確率行列）"""
    A = rng.random((n_states, n_states))
    return normalize_rows(A)


def make_hmm_params(
    n_models: int,
    n_states: int,
    n_symbols: int,
    is_left_to_right: bool,
    rng: np.random.Generator,
) -> dict:
    """HMMパラメータ (PI, A, B) を生成する"""
    PI_list, A_list, B_list = [], [], []

    for _ in range(n_models):
        # 初期確率
        if is_left_to_right:
            # Left-to-Right は必ず状態 0 から開始
            pi = np.zeros((n_states, 1))
            pi[0, 0] = 1.0
        else:
            raw = np.clip(rng.random((n_states, 1)), 1e-10, None)
            pi = raw / raw.sum()

        # 状態遷移確率行列
        if is_left_to_right:
            A = make_left_to_right_A(n_states, rng)
        else:
            A = make_ergodic_A(n_states, rng)

        # 出力確率行列
        B = normalize_rows(rng.random((n_states, n_symbols)))

        PI_list.append(pi)
        A_list.append(A)
        B_list.append(B)

    return {
        "PI": np.array(PI_list),  # [k, l, 1]
        "A": np.array(A_list),   # [k, l, l]
        "B": np.array(B_list),   # [k, l, n]
    }


def generate_sequence(pi: np.ndarray, A: np.ndarray, B: np.ndarray,
                       seq_len: int, rng: np.random.Generator) -> np.ndarray:
    """1つのHMMから出力系列を生成する"""
    n_states = A.shape[0]
    n_symbols = B.shape[1]

    def _safe_p(arr):
        arr = np.clip(arr, 0, None)
        return arr / arr.sum()

    # 初期状態をサンプリング
    state = rng.choice(n_states, p=_safe_p(pi[:, 0]))
    sequence = []

    for _ in range(seq_len):
        # 出力記号をサンプリング
        obs = rng.choice(n_symbols, p=_safe_p(B[state]))
        sequence.append(obs)
        # 次の状態をサンプリング
        state = rng.choice(n_states, p=_safe_p(A[state]))

    return np.array(sequence)


def generate_dataset(
    models: dict,
    n_sequences: int,
    seq_len: int,
    rng: np.random.Generator,
) -> dict:
    """データセットを生成する"""
    k = models["PI"].shape[0]
    outputs = []
    answer_models = []

    for _ in range(n_sequences):
        model_idx = rng.integers(0, k)
        seq = generate_sequence(
            models["PI"][model_idx],
            models["A"][model_idx],
            models["B"][model_idx],
            seq_len,
            rng,
        )
        outputs.append(seq)
        answer_models.append(model_idx)

    return {
        "answer_models": np.array(answer_models),  # [p,]
        "output": np.array(outputs),               # [p, t]
        "models": models,
    }


def main():
    os.makedirs("data", exist_ok=True)
    seed = 42

    configs = [
        # (filename,  is_ltr, n_models, n_states, n_symbols, n_seq, seq_len)
        ("data1.pickle", True,  3, 4, 6, 300, 20),   # Left-to-Right, 少数
        ("data2.pickle", False, 3, 4, 6, 300, 20),   # Ergodic, 少数
        ("data3.pickle", True,  5, 5, 8, 500, 30),   # Left-to-Right, 多数
        ("data4.pickle", False, 5, 5, 8, 500, 30),   # Ergodic, 多数
    ]

    for fname, is_ltr, k, l, n, p, t in configs:
        rng = np.random.default_rng(seed)
        models = make_hmm_params(k, l, n, is_ltr, rng)
        dataset = generate_dataset(models, p, t, rng)
        out_path = os.path.join("data", fname)
        with open(out_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Generated {out_path}:")
        print(f"  models={k}, states={l}, symbols={n}, sequences={p}, seq_len={t}")
        print(f"  Type: {'Left-to-Right' if is_ltr else 'Ergodic'}")
        print()


if __name__ == "__main__":
    main()
