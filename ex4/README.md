# 第6回B4輪講課題

## 概要

> [!WARNING]
> - 本課題でのコーディングエージェントやAIツールの利用禁止（コンペまではAIツール無しで，考えて実装する力を養成するため）
> - numpyの行列演算を使って実装すること

本課題では，隠れマルコフモデル（HMM）を用いて，出力系列の尤度計算と，各出力系列を生成したモデルの推定を行う．

## データセット `data/` について

- `data1.pickle`：Left-to-Right HMM から生成されたデータ（少数クラス）
- `data2.pickle`：Ergodic HMM から生成されたデータ（少数クラス）
- `data3.pickle`：Left-to-Right HMM から生成されたデータ（多数クラス）
- `data4.pickle`：Ergodic HMM から生成されたデータ（多数クラス）

pickleデータの読み込み方法：

```python
import pickle
data = pickle.load(open("data/data1.pickle", "rb"))
```

### dataの階層構造

```
data
├─ answer_models   # 出力系列を生成したモデル（正解ラベル） [p,]
├─ output          # 出力系列 [p, t]
└─ models          # 定義済みHMM
   ├─ PI           # 初期確率 [k, l, 1]
   ├─ A            # 状態遷移確率行列 [k, l, l]
   └─ B            # 出力確率 [k, l, n]
```

- $k$：モデル数，$l$：状態数，$n$：出力記号数，$p$：出力系列数，$t$：系列長

### 具体的な中身の例

- `answer_models = [1, 3, 3, 4, 0, 2, ...]`：出力系列 0 は $m_1$ から，出力系列 1 は $m_3$ から生成された
- `output[0] = [0, 4, 2, ..., 4, 0, 0]`：出力系列 0 の出力記号列は $o_0, o_4, o_2, \ldots$ だった
- `PI[0] = [[1], [0], [0]]`：$m_0$ の初期状態が $s_0$ である確率が 1，$s_1, s_2$ は 0

## 課題

### 6-1 Forwardアルゴリズムによる尤度計算とモデル推定

- Forward アルゴリズムを実装し，各出力系列 $O$ と各モデル $m_k$ に対して尤度 $P(O \mid m_k)$ を計算せよ．
- 尤度が最大となるモデルを推定モデルとして，正解ラベルと比較せよ．
- 混同行列（Confusion Matrix）と正解率（Accuracy）を算出・表示せよ．
- アルゴリズムの計算時間を測定せよ．

### 6-2 Viterbiアルゴリズムによるモデル推定

- Viterbi アルゴリズムを実装し，各出力系列 $O$ と各モデル $m_k$ に対して最適パスの対数確率を計算せよ．
- 対数確率が最大となるモデルを推定モデルとして，正解ラベルと比較せよ．
- 混同行列（Confusion Matrix）と正解率（Accuracy）を算出・表示せよ．
- アルゴリズムの計算時間を測定せよ．

### 6-3 性能比較

- ForwardアルゴリズムとViterbiアルゴリズムの結果（正解率，計算時間）を比較・考察せよ．
- data1〜data4 の各データセットで実験を行い，Left-to-Right HMM と Ergodic HMM の違いについて考察せよ．

## 出力例
`data/data1.pickle`, `data/data2.pickle` に対して Forward, Viterbi アルゴリズムを実行した結果を以下に記載．

![data/data1.pickleの結果](./fig/data1_result.png)

![data/data2.pickleの結果](./fig/data2_result.png)

## アルゴリズムの概要

### Forwardアルゴリズム

前向き確率 $\alpha_t(i)$ を以下のように定義する：

$$\alpha_t(i) = P(o_1, o_2, \ldots, o_t, q_t = s_i \mid \lambda)$$

**初期化：**
$$\alpha_1(i) = \pi_i b_i(o_1)$$

**漸化式：**
$$\alpha_{t+1}(j) = \left[\sum_{i=1}^{N} \alpha_t(i) a_{ij}\right] b_j(o_{t+1})$$

**尤度：**
$$P(O \mid \lambda) = \sum_{i=1}^{N} \alpha_T(i)$$

### Viterbiアルゴリズム

最大確率 $\delta_t(i)$ を以下のように定義する：

$$\delta_t(i) = \max_{q_1,\ldots,q_{t-1}} P(q_1,\ldots,q_{t-1}, q_t=s_i, o_1,\ldots,o_t \mid \lambda)$$

**初期化：**
$$\delta_1(i) = \pi_i b_i(o_1)$$

**漸化式（対数域での計算を推奨）：**
$$\delta_{t+1}(j) = \max_i \left[\delta_t(i) a_{ij}\right] b_j(o_{t+1})$$

**最適パスの確率：**
$$P^* = \max_i \delta_T(i)$$

> [!TIP]
> アンダーフロー対策として，対数スケール（log-domain）での計算を推奨する．

## 発展課題（余裕がある人向け）

- numpy の行列演算を積極的に活用し，for ループを削減して高速化を試みよ
- スケーリング法（scaling factor）による Forward アルゴリズムのアンダーフロー対策を実装せよ
- 異なる系列長・モデル数でのスケーラビリティを評価せよ

## 発表（次週）

- 取り組んだ内容を周りにわかるように説明
- コードの解説
    - 工夫したところ，苦労したところの解決策はぜひ共有しましょう
- 結果の考察，応用先の調査など
- 発表資料は nas01 の `internal/発表資料/B4輪講/2026/第6回` へアップロードしておくこと

## 注意

- 自分の作業ブランチで課題を行うこと
- プルリクエストをおくる際には**実行結果の画像ファイルも載せること**
- プルリクエストのコメントには，結果を作るために実行したコマンドも書くこと
- 作業前にリポジトリを最新版に更新すること

```bash
$ git checkout main
$ git fetch upstream
$ git merge upstream/main
```
