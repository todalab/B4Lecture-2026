# 解答例

このディレクトリには第1回課題の解答例を置いている。

## 実行方法

```bash
uv run python linear_regression.py --input ../data/sample2d_1.csv --degree 1
uv run python linear_regression.py --input ../data/sample2d_2.csv --degree 3
uv run python linear_regression.py --input ../data/sample3d.csv --degree 2
uv run python logistic_regression.py --input ../data/sample_logistic.csv --lr 0.1 --iters 300
```

画像は `out/` に保存される。

コード中では、講義資料に合わせて、計画行列を `X`、正解の値を `y`、重みを `w` と書いている。

## 線形回帰・多項式回帰

`linear_regression.py` は1変数の多項式回帰と、2変数の多項式曲面フィッティングに対応している。

主な引数:

- `--input`: 入力CSV
- `--features`: 説明変数の列名
- `--target`: 目的変数の列名
- `--degree`: 多項式の次数
- `--regularization`: `none`, `ridge`, `lasso`
- `--lambda`: 正則化係数
- `--iters`: lasso 用の座標降下法の反復回数
- `--standardize`: 多項式項を作る前に説明変数を標準化する

想定される結果:

- `sample2d_1.csv`: 1次式でよくフィットする
- `sample2d_2.csv`: 1次・2次より3次のほうがよくフィットする
- `sample3d.csv`: 1次平面より2次曲面のほうがよくフィットする

過学習や正則化の観察例:

```bash
uv run python linear_regression.py --input ../data/sample2d_2.csv --degree 20 --standardize
uv run python linear_regression.py --input ../data/sample2d_2.csv --degree 20 --regularization ridge --lambda 0.1 --standardize
uv run python linear_regression.py --input ../data/sample2d_2.csv --degree 20 --regularization lasso --lambda 0.01 --iters 1000 --standardize
```

## ロジスティック回帰

`logistic_regression.py` は勾配降下法による2クラス分類を行う。

主な引数:

- `--input`: 入力CSV
- `--features`: 説明変数の列名
- `--target`: 目的変数の列名
- `--lr`: 学習率
- `--iters`: 反復回数
- `--regularization`: `none`, `ridge`, `lasso`
- `--lambda`: 正則化係数

出力画像には、損失関数、対数尤度、accuracy の推移を保存する。
