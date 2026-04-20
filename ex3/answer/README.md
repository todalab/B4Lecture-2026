# 解答例

このディレクトリには第3回課題の解答例を置いている。

## 実行方法

```bash
# データ生成（初回のみ）
python generate_data.py

# 3-1: 散布図の確認
python answer/01_scatter.py

# 3-2: GMM 実装の動作確認
python answer/02_gmm.py

# 3-3: クラスタリング結果・対数尤度の可視化
python answer/03_clustering.py

# 3-4: AIC/BIC によるクラスター数の選択
python answer/04_model_selection.py
```

画像は `fig/` に保存される。

## GMM の実装（`answer/gmm.py`）

`answer/gmm.py` が実装本体で、`03_clustering.py` と `04_model_selection.py` から `from answer.gmm import GMM` としてインポートして使う。

主なクラス・関数:

- `gaussian_pdf(X, mu, Sigma)`: 多変量正規分布の確率密度関数。`einsum` でベクトル化している
- `GMM.fit(X)`: EM アルゴリズムで学習する。収束後に負担率 `r_` とパラメータ（`pi`, `mu`, `Sigma`）を保持する
- `GMM.predict(X)`: 負担率が最大のクラスターを返す（ハード割り当て）
- `GMM.aic(X)`, `GMM.bic(X)`: 情報量基準を計算する

主なパラメータ:

- `K`: クラスター数
- `max_iter`: 最大反復回数（既定値 300）
- `tol`: 収束判定の閾値（対数尤度の変化量、既定値 `1e-6`）
- `reg`: 共分散行列の正則化パラメータ（既定値 `1e-6`、対角成分に加算する）
- `random_state`: 乱数シード

## Eステップ・Mステップ

**Eステップ**では、現在のパラメータ $(\pi_k, \mu_k, \Sigma_k)$ のもとで、各データ点 $x_n$ が各クラスター $k$ に属する事後確率（負担率）を計算する。

$$r_{nk} = \frac{\pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j} \pi_j \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}$$

**Mステップ**では、負担率を重みとして、対数尤度を最大化するパラメータを解析的に更新する。

$$\mu_k = \frac{\sum_n r_{nk} x_n}{\sum_n r_{nk}}, \quad
\Sigma_k = \frac{\sum_n r_{nk}(x_n - \mu_k)(x_n - \mu_k)^\top}{\sum_n r_{nk}}, \quad
\pi_k = \frac{\sum_n r_{nk}}{N}$$

EM アルゴリズムでは対数尤度が反復ごとに単調増加することが保証されている。`fig/*_loglikelihood.png` でこれを確認できる。

## 想定される結果

- `data1.csv`（K=3）: 明瞭に分離した3クラスターをほぼ正しく分割できる
- `data2.csv`（K=4）: 近接する2クラスターの境界付近で負担率が曖昧になり、等高線の重なりが見られる
- `data3.csv`（K=2）: 強い相関を持つ細長い楕円形状のクラスターを、共分散行列が正しく捉える

## AIC・BIC によるクラスター数の選択

GMM のパラメータ数は $K$ クラスター・$d$ 次元のとき

$$p = (K - 1) + Kd + K \cdot \frac{d(d+1)}{2}$$

である（混合係数・平均・共分散の自由度の和）。AIC・BIC はともに対数尤度にパラメータ数のペナルティを加えたもので、値が小さいモデルを選ぶ。BIC はサンプル数 $N$ に応じてペナルティが大きくなるため、AIC より少ないクラスター数を選ぶ傾向がある。

$$\text{AIC} = -2 \log \hat{L} + 2p, \quad \text{BIC} = -2 \log \hat{L} + p \log N$$

## 可視化について

`03_clustering.py` では各ガウス分布の等高線として 1σ・2σ の信頼楕円を描いている。共分散行列を固有値分解し、固有ベクトルを楕円の向き、固有値の平方根を軸長として `matplotlib.patches.Ellipse` で描画している。
