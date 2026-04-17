# 解答例

このディレクトリには第2回課題の解答例を置いている。

## 実行方法

```bash
uv run python pca.py --input ../data/pca_2d.csv
uv run python pca.py --input ../data/pca_3d.csv
uv run python pca.py --input ../data/pca_100d.csv
uv run python lda.py --input ../data/lda_2class.csv
```

画像は `out/` に保存される。

コード中では、講義資料に合わせて、入力データを平均中心化した行列を `z`、共分散行列を `covariance`、LDA のクラス内分散行列を `sw`、クラス間分散行列を `sb` と書いている。

## PCA

`pca.py` は、ヘッダーなしCSVを読み込み、平均中心化、共分散行列の計算、固有値分解、寄与率と累積寄与率の計算を行う。

主な引数:

- `--input`: 入力CSV
- `--normalize`: `center`, `standardize`。既定値は `center`

想定される結果:

- `pca_2d.csv`: 第1主成分がデータの大きな広がりの向きに沿う
- `pca_3d.csv`: 3次元上の主成分軸と、第1・第2主成分へ射影した2次元構造を確認できる
- `pca_100d.csv`: はじめに大きく上がり、その後なだらかに飽和する累積寄与率になる

## LDA

`lda.py` は、`x1,x2,label` のヘッダー付きCSVを読み込み、2クラスLDAを行う。

主な処理:

- クラスごとの平均ベクトルを計算する
- クラス内分散行列 `sw` とクラス間分散行列 `sb` を計算する
- `pinv(sw) @ sb` の固有値問題を解き、射影方向を求める
- 射影後の1次元値をしきい値で分類し、accuracy を計算する

PCA はラベルを使わず、データ全体の分散が最大になる向きを選ぶ。LDA はラベルを使い、クラス内のばらつきを小さく、クラス間の離れを大きくする向きを選ぶ。
