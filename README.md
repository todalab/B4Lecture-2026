# 戸田研究室B4輪講 演習問題

本リポジトリは，戸田研究室のB4輪講の演習問題を取り扱う．

## 演習の始め方

1. 本リポジトリを自分のアカウントにフォークする
  (右上のForkボタンを押す)

2. フォークした自分のリポジトリに自動的に遷移する。

3. フォークした自分のリポジトリを適当な場所へダウンロードする。(リポジトリページ右上の[↓Code]ボタンからURLをコピー)

    ```bash
    $ git clone <自分のgithubのURL>
    ```

4. 本家リポジトリを登録 (upstreamという名前でなくてもいい)

    ```bash
    $ cd B4Lecture-2026
    $ git remote add upstream https://github.com/todalab/B4Lecture-2026
    ```


## 演習の進め方

1. mainブランチに戻る
  ```bash
  $ cd B4Lecture-2026
  $ git checkout main
  ```
2. 本家リポジトリから更新されたソースをマージする
  ```bash
  $ git fetch upstream
  $ git merge upstream/main
  ```
3. ブランチを作成する
  ```bash
  $ git checkout -b ex_XX (ブランチ名。何でもいい)
  ```
3. 自分の名前のディレクトリ（`<firstname-initial>_<lastname>`）を作成する
  ```bash
  $ mkdir -p exXX/m_matsumoto
  ```
6. ディレクトリ内でスクリプトを作成する

7. 適宜gitを使ってコミットする(ローカルのgitに反映される。こまめにやっていいよ)
  ```bash
  # 例
  git add main.py
  git commit -m "新しい関数を追加"
  ```

8. githubにpushする(フォークした自分のgithubに反映される。こまめにやっていいよ)
  ```bash
  $ git push origin exXX (ブランチ名)
  ```

9. 一通り実装したら、githubにアクセスしてプルリクエストを作成し，レビューをお願いする（下参照）

10. レビューを受けてRequest Changesを修正 -> add -> commit -> pushを繰り返す

11. 必須レビュアー（修士学生１人）がApproveしたら、澤田がマージする。マージされたら本家リポジトリに自分のコードが反映される。


## その他

[演習を進める上でのコツ](./TIPS.md)

[スライドの作り方](https://www.slideshare.net/ShinnosukeTakamichi/ss-48987441)

[PowerPointを使いやすくするために](https://drive.google.com/file/d/1DIMUqkFAyphAla_q7Ec5-M-1p3_GlUHx/view?usp=sharing)

[講義ビデオ](https://drive.google.com/drive/folders/1aOAgjTjUutiw3qwPwpRKhxDrnY0n5XEX?usp=sharing)

[過去資料](https://todalab-nu.slack.com/archives/C0AHRB60UBD/p1775626924970089)

<!-- [リンク切れ（GitとGithubの理解）](ｈｔｔｐｓ://docs.google.com/a/g.sp.m.is.nagoya-u.ac.jp/viewer?a=v&pid=sites&srcid=Zy5zcC5tLmlzLm5hZ295YS11LmFjLmpwfHNwbG9jYWwtc2VtaXxneDoxZmI4YWVhZWVlNDBjNDY1) -->

<!-- [過去資料](ｈｔｔｐｓ://sites.google.com/a/g.sp.m.is.nagoya-u.ac.jp/splocal-semi/b4-rinkou) -->