#!/usr/bin/env python3
"""evaluate.py — 翻訳性能評価スクリプト.

異なるサイズの Transformer モデルを評価し、以下の 3 点を出力する:
    1. Perplexity（困惑度）: 低いほど性能が良い
    2. ChrF スコア: 高いほど性能が良い
    3. 任意の入力文に対する翻訳結果

使用例:
    # 引数なし: 全 4 サイズを評価・比較。
    # ただしチェックポイント (checkpoints/translation_{size}_best.pt) が
    # 無いサイズは自動でスキップされるため、全サイズの学習は必須ではない。
    python evaluate.py

    # 学習済みのサイズだけを指定して評価・比較する。
    python evaluate.py --model_sizes tiny small
    python evaluate.py --model_sizes tiny small medium large

    # 翻訳する英文を指定する (未指定時は DEFAULT_SENTENCES を使用)。
    python evaluate.py --model_sizes small --sentences "Good morning." "See you tomorrow."

    # チェックポイントファイルを直接指定する (--model_sizes は 1 つのみ)。
    python evaluate.py --model_sizes small --checkpoint path/to/model.pt

    # 学習時に保存した metrics.json から epoch ごとの train/val loss・perplexity
    # の推移を描画する。図はサイズ別ディレクトリに、epoch 数を名前に含めて
    # 保存される: image/{size}/epoch_metrics_ep{N}.png
    python evaluate.py --model_sizes tiny small --plot_metrics

    # epoch 別チェックポイントから epoch ごとの Perplexity/ChrF を計算し直して
    # 描画する: image/{size}/metrics_over_epochs_ep{N}.png
    # 解像度は学習時のチェックポイント保存間隔に依存。
    python evaluate.py --model_sizes tiny small --chrf_curve

    # 図の保存先ルートを変えたい場合は --image_dir を指定する (default: image)。
    python evaluate.py --model_sizes tiny small --chrf_curve --image_dir fig
"""

import argparse
import glob
import json
import logging
import math
import os
import re

import matplotlib.pyplot as plt
import sacrebleu
import torch
from data_loader import BOS_IDX, EOS_IDX, create_data_loaders
from tqdm import tqdm
from training_utils import get_device
from transformer_skeleton import TranslationModel, get_model_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SENTENCES = [
    "I will check the schedule .",
    "Thank you for your help .",
    "Please send me the report .",
    "The meeting is at three o'clock .",
    "Could you please confirm the details ?",
]


def load_model(
    model_size,
    checkpoint_dir,
    src_vocab_size,
    tgt_vocab_size,
    max_seq_len,
    device,
    checkpoint_path=None,
):
    """チェックポイントからモデルを読み込む.

    Args:
        model_size: モデルサイズ ("tiny" / "small" / "medium" / "large")
        checkpoint_dir: チェックポイントディレクトリ
        src_vocab_size: ソース語彙サイズ
        tgt_vocab_size: ターゲット語彙サイズ
        max_seq_len: 最大系列長
        device: 使用デバイス
        checkpoint_path: チェックポイントファイルへの明示パス。指定時は
            checkpoint_dir / model_size から生成されるパスより優先される。

    Returns:
        (評価モードの TranslationModel, 学習 epoch 数) のタプル。
        チェックポイントが存在しない場合は (None, None)。
    """
    # 明示パスが渡された場合はそれを優先し、無ければサイズ規約からパスを組み立てる
    if checkpoint_path is not None:
        ckpt_path = checkpoint_path
    else:
        ckpt_path = os.path.join(checkpoint_dir, f"translation_{model_size}_best.pt")
    # 学習済みファイルが無いサイズは呼び出し側でスキップできるよう None を返す
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return None, None

    # サイズに応じた層数・次元などの構成を取得し、同じ構成でモデルを再構築する
    config = get_model_config(model_size)
    model = TranslationModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_encoder_layers=config["n_encoder_layers"],
        n_decoder_layers=config["n_decoder_layers"],
        d_ff=config["d_ff"],
        max_seq_len=max_seq_len,
    ).to(device)

    # 学習済みの重みを読み込み、評価モードに切り替える
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch")
    logger.info(
        f"Loaded '{model_size}' from {ckpt_path} (epoch {epoch if epoch is not None else '?'})"
    )
    return model, epoch


def compute_perplexity(model, val_loader, device):
    """バリデーションセット全体の Perplexity を計算する.

    Args:
        model: 評価モードの TranslationModel
        val_loader: バリデーション DataLoader
        device: 使用デバイス

    Returns:
        Perplexity (float)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    # 勾配計算を無効化してメモリと計算量を抑える
    with torch.no_grad():
        for src, tgt_in, tgt_out in tqdm(val_loader, desc="  Perplexity", leave=False):
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            # targets を渡すとモデルが交差エントロピー損失も返す
            _, loss = model(src, tgt_in, targets=tgt_out)
            total_loss += loss.item()
            num_batches += 1
    # Perplexity = exp(平均損失)。min(..., 20) で exp のオーバーフローを防ぐ
    avg_loss = total_loss / num_batches
    return math.exp(min(avg_loss, 20))


def compute_chrf(
    model, val_loader, src_tokenizer, tgt_tokenizer, device, max_len, n_samples
):
    """バリデーションセットの ChrF スコアを計算する.

    Args:
        model: 評価モードの TranslationModel
        val_loader: バリデーション DataLoader
        src_tokenizer: ソース(英語)トークナイザー
        tgt_tokenizer: ターゲット(日本語)トークナイザー
        device: 使用デバイス
        max_len: 最大生成長
        n_samples: 評価に使うサンプル数

    Returns:
        ChrF スコア (float, 0–100)
    """
    model.eval()
    hypotheses = []  # モデルが生成した翻訳文
    references = []  # 正解の参照翻訳文
    count = 0
    with torch.no_grad():
        for src_batch, _, tgt_out_batch in tqdm(val_loader, desc="  ChrF", leave=False):
            src_batch = src_batch.to(device)
            # 自己回帰生成でバッチ分の翻訳を一括生成する
            generated = model.generate(src_batch, BOS_IDX, EOS_IDX, max_len=max_len)
            for gen_ids, ref_ids in zip(generated, tgt_out_batch):
                # トークン ID 列を文字列に戻して仮説・参照を蓄積する
                hyp = tgt_tokenizer.decode(gen_ids.cpu().tolist())
                ref = tgt_tokenizer.decode(ref_ids.tolist())
                hypotheses.append(hyp)
                references.append(ref)
                count += 1
                # 指定サンプル数に達したら内側・外側ループとも打ち切る
                if count >= n_samples:
                    break
            if count >= n_samples:
                break
    # コーパス単位の ChrF を計算 (参照はリストのリストで渡す)
    result = sacrebleu.corpus_chrf(hypotheses, [references])
    return result.score


def show_translations(model, sentences, src_tokenizer, tgt_tokenizer, device, max_len):
    """任意の入力文に対して翻訳を生成して表示する.

    Args:
        model: 評価モードの TranslationModel
        sentences: 翻訳元の英文リスト
        src_tokenizer: ソース(英語)トークナイザー
        tgt_tokenizer: ターゲット(日本語)トークナイザー
        device: 使用デバイス
        max_len: 最大生成長
    """
    model.eval()
    with torch.no_grad():
        for sent in sentences:
            # 英文をトークン ID に変換し、バッチ次元を足して (1, seq_len) にする
            src_ids = src_tokenizer.encode(sent, add_eos=True, max_len=max_len)
            src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
            # 1 文だけ生成し、ID 列を日本語文字列にデコードして表示する
            generated = model.generate(src_tensor, BOS_IDX, EOS_IDX, max_len=max_len)
            translation = tgt_tokenizer.decode(generated[0].cpu().tolist())
            print(f"    EN: {sent}")
            print(f"    JA: {translation}")


def evaluate_epoch_checkpoints(
    model_size,
    checkpoint_dir,
    src_vocab_size,
    tgt_vocab_size,
    max_seq_len,
    val_loader,
    src_tokenizer,
    tgt_tokenizer,
    device,
    max_len,
    chrf_samples,
):
    """epoch ごとの中間チェックポイントを順に評価する.

    `{checkpoint_dir}/translation_{size}_epoch_{N}.pt` を全て読み込み、各 epoch
    時点の Perplexity と ChrF を計算する。ChrF は学習時に記録されないため、
    ここでチェックポイントから計算し直す (解像度は保存間隔に依存)。

    Args:
        model_size: モデルサイズ
        checkpoint_dir: チェックポイントディレクトリ
        src_vocab_size: ソース語彙サイズ
        tgt_vocab_size: ターゲット語彙サイズ
        max_seq_len: 最大系列長
        val_loader: バリデーション DataLoader
        src_tokenizer: ソーストークナイザー
        tgt_tokenizer: ターゲットトークナイザー
        device: 使用デバイス
        max_len: 最大生成長
        chrf_samples: ChrF 計算に使うサンプル数

    Returns:
        epoch 昇順の [{"epoch": int, "perplexity": float, "chrf": float}, ...]。
    """
    # translation_{size}_epoch_{N}.pt を集め、ファイル名から epoch 番号を取り出す
    pattern = os.path.join(checkpoint_dir, f"translation_{model_size}_epoch_*.pt")
    epoch_paths = []
    for path in glob.glob(pattern):
        m = re.search(r"_epoch_(\d+)\.pt$", os.path.basename(path))
        if m:
            epoch_paths.append((int(m.group(1)), path))
    epoch_paths.sort()  # epoch 昇順に並べる

    records = []
    for epoch_num, path in epoch_paths:
        model, _ = load_model(
            model_size=model_size,
            checkpoint_dir=checkpoint_dir,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            max_seq_len=max_seq_len,
            device=device,
            checkpoint_path=path,
        )
        if model is None:
            continue
        ppl = compute_perplexity(model, val_loader, device)
        chrf_score = compute_chrf(
            model=model,
            val_loader=val_loader,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
            max_len=max_len,
            n_samples=chrf_samples,
        )
        print(f"    epoch {epoch_num:>3}: PPL={ppl:.2f}  ChrF={chrf_score:.2f}")
        records.append(
            {"epoch": epoch_num, "perplexity": ppl, "chrf": chrf_score}
        )
    return records


def build_figure_path(image_dir, model_size, name, epoch):
    """モデルサイズ別ディレクトリ内に、epoch を名前に含めた図のパスを作る.

    モデルを変えても上書きしないよう `{image_dir}/{model_size}/` に分け、
    ファイル名に epoch 数を埋め込む。

    Args:
        image_dir: 図の保存先ルートディレクトリ
        model_size: モデルサイズ (サブディレクトリ名になる)
        name: 図の種類を表す基底名 (例: "epoch_metrics")
        epoch: 学習 epoch 数 (不明なら None)

    Returns:
        保存先のフルパス (ディレクトリは作成済み)。
    """
    sub_dir = os.path.join(image_dir, model_size)
    os.makedirs(sub_dir, exist_ok=True)
    epoch_str = f"ep{epoch}" if epoch is not None else "epNA"
    return os.path.join(sub_dir, f"{name}_{epoch_str}.png")


def plot_metrics_over_epochs(model_size, records, image_dir):
    """epoch ごとに計算した Perplexity / ChrF の推移をグラフ化して保存する.

    Args:
        model_size: モデルサイズ
        records: [{"epoch", "perplexity", "chrf"}, ...] (epoch 昇順)
        image_dir: 図の保存先ルートディレクトリ
    """
    if not records:
        print(f"  [INFO] '{model_size}' の epoch 評価結果がありませんでした")
        return

    epochs = [r["epoch"] for r in records]
    # 左: Perplexity (低いほど良い)、右: ChrF (高いほど良い)
    metric_keys = [
        ("perplexity", "Val Perplexity (lower=better)"),
        ("chrf", "ChrF (higher=better)"),
    ]
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(6 * len(metric_keys), 4))
    for ax, (key, title) in zip(axes, metric_keys):
        ax.plot(epochs, [r[key] for r in records], marker="o", markersize=4)
        ax.set_title(f"{model_size}: {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    # ファイル名の epoch は到達した最大 epoch を使う
    out_path = build_figure_path(
        image_dir, model_size, "metrics_over_epochs", max(epochs)
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Epoch ごとの Perplexity/ChrF グラフを保存しました: {out_path}")


def plot_epoch_metrics(model_size, checkpoint_dir, image_dir):
    """学習時に保存した epoch ごとの指標をグラフ化して保存する.

    `{checkpoint_dir}/translation_{size}_metrics.json` を読み込み、
    train_loss / val_loss / val_perplexity の推移を
    `{image_dir}/{model_size}/epoch_metrics_ep{N}.png` に保存する。

    Args:
        model_size: モデルサイズ
        checkpoint_dir: metrics.json があるディレクトリ
        image_dir: 図の保存先ルートディレクトリ
    """
    metrics_path = os.path.join(
        checkpoint_dir, f"translation_{model_size}_metrics.json"
    )
    if not os.path.exists(metrics_path):
        logger.warning(f"Metrics file not found: {metrics_path}")
        return
    with open(metrics_path) as f:
        metrics = json.load(f)

    # 3 つの指標を横並びのサブプロットに描く
    metric_keys = [
        ("train_losses", "Train Loss"),
        ("val_losses", "Val Loss"),
        ("val_perplexities", "Val Perplexity"),
    ]
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(5 * len(metric_keys), 4))
    for ax, (key, title) in zip(axes, metric_keys):
        values = metrics.get(key, [])
        # epoch は 1 始まりで表示する
        ax.plot(range(1, len(values) + 1), values, marker="o", markersize=3)
        ax.set_title(f"{model_size}: {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    # 学習 epoch 数は val_losses の長さから求める
    n_epochs = len(metrics.get("val_losses", []))
    out_path = build_figure_path(
        image_dir, model_size, "epoch_metrics", n_epochs
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Epoch ごとの指標グラフを保存しました: {out_path}")


def main():
    """評価スクリプトのメインエントリポイント."""
    parser = argparse.ArgumentParser(
        description="モデルサイズ別 翻訳性能評価 (Perplexity / ChrF / 翻訳例)"
    )
    parser.add_argument(
        "--model_sizes",
        nargs="+",
        choices=["tiny", "small", "medium", "large"],
        default=["tiny", "small", "medium", "large"],
        help="評価するモデルサイズ (スペース区切りで複数指定可)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="チェックポイントディレクトリ (default: checkpoints)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "チェックポイントファイルへの明示パス。指定時は checkpoint_dir / "
            "model_size から生成されるパスより優先される (model_sizes は 1 つのみ指定)"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="バッチサイズ (default: 64)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=64,
        help="最大系列長 (default: 64)",
    )
    parser.add_argument(
        "--src_vocab_size",
        type=int,
        default=8000,
        help="英語語彙サイズ (default: 8000)",
    )
    parser.add_argument(
        "--tgt_vocab_size",
        type=int,
        default=4000,
        help="日本語語彙サイズ (default: 4000)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100_000,
        help="使用する最大サンプル数 (default: 100000)",
    )
    parser.add_argument(
        "--chrf_samples",
        type=int,
        default=500,
        help="ChrF 計算に使うサンプル数 (default: 500)",
    )
    parser.add_argument(
        "--sentences",
        nargs="+",
        default=None,
        help="翻訳する英文 (スペース区切りで複数指定。未指定時はデフォルト文を使用)",
    )
    parser.add_argument(
        "--plot_metrics",
        action="store_true",
        help="epoch ごとの指標 (train/val loss, perplexity) をグラフ化して保存する",
    )
    parser.add_argument(
        "--chrf_curve",
        action="store_true",
        help=(
            "保存済みの epoch 別チェックポイントから epoch ごとの Perplexity/ChrF を "
            "計算してグラフ化する (計算コスト高。解像度は保存間隔に依存)"
        ),
    )
    parser.add_argument(
        "--image_dir",
        default="image",
        help="グラフの保存先ディレクトリ (default: image)",
    )
    args = parser.parse_args()

    # 明示パス指定は 1 モデル専用 (複数サイズで同じファイルを使うのは曖昧)
    if args.checkpoint is not None and len(args.model_sizes) > 1:
        parser.error("--checkpoint 指定時は --model_sizes を 1 つだけ指定してください")

    device = get_device()
    # 翻訳例に使う文。未指定ならデフォルト文を使う
    sentences = args.sentences if args.sentences else DEFAULT_SENTENCES

    # データセットと語彙は全モデル共通なので、ループ前に一度だけ構築する
    logger.info("Loading dataset and building vocabularies ...")
    train_loader, val_loader, src_tokenizer, tgt_tokenizer = create_data_loaders(
        max_len=args.max_len,
        batch_size=args.batch_size,
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.tgt_vocab_size,
        max_samples=args.max_samples,
    )
    logger.info(
        f"Vocab: en={len(src_tokenizer)} (word), ja={len(tgt_tokenizer)} (char)"
    )

    results = {}  # サイズ名 -> {"perplexity": ..., "chrf": ...}

    # 指定された各サイズを順に評価する
    for model_size in args.model_sizes:
        print(f"\n{'=' * 56}")
        print(f"  Model: {model_size}")
        print(f"{'=' * 56}")

        model, epoch = load_model(
            model_size=model_size,
            checkpoint_dir=args.checkpoint_dir,
            src_vocab_size=len(src_tokenizer),
            tgt_vocab_size=len(tgt_tokenizer),
            max_seq_len=args.max_len * 2,
            device=device,
            checkpoint_path=args.checkpoint,
        )
        # 学習済みチェックポイントが無いサイズはスキップ
        if model is None:
            print(f"  [SKIP] Checkpoint not found for '{model_size}'")
            print(f"  先に: python main.py --model_size {model_size}")
            continue

        # 指標 1: Perplexity (言語モデルとしての予測のしやすさ)
        ppl = compute_perplexity(model, val_loader, device)
        print(f"  Perplexity : {ppl:.2f}  (低いほど良い)")

        # 指標 2: ChrF (生成翻訳と参照訳の文字 n-gram 一致度)
        chrf_score = compute_chrf(
            model=model,
            val_loader=val_loader,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
            max_len=args.max_len,
            n_samples=args.chrf_samples,
        )
        print(f"  ChrF Score : {chrf_score:.2f}  (高いほど良い)")

        # 指標 3: 任意の入力文に対する実際の翻訳例を表示
        print(f"  翻訳例 ({len(sentences)} 文):")
        show_translations(
            model, sentences, src_tokenizer, tgt_tokenizer, device, args.max_len
        )

        results[model_size] = {"perplexity": ppl, "chrf": chrf_score, "epoch": epoch}

    # 2 サイズ以上を評価したときに横並びで比較できるサマリー表を出力
    if results:
        print(f"\n{'=' * 56}")
        print("  比較サマリー")
        print(f"{'=' * 56}")
        header = f"  {'Size':<8} {'Epoch':>6} {'Perplexity':>12} {'ChrF':>8}"
        print(header)
        print(f"  {'-' * 38}")
        # 入力順 (args.model_sizes) を保ちつつ、評価できたサイズだけ表示
        for size in args.model_sizes:
            if size in results:
                r = results[size]
                # epoch が保存されていないチェックポイントでは "?" を表示
                epoch_str = str(r["epoch"]) if r["epoch"] is not None else "?"
                print(
                    f"  {size:<8} {epoch_str:>6} "
                    f"{r['perplexity']:>12.2f} {r['chrf']:>8.2f}"
                )
        print(f"{'=' * 56}")

    # --plot_metrics 指定時は epoch ごとの指標推移をサイズ別に描画して保存する
    if args.plot_metrics:
        print(f"\n{'=' * 56}")
        print("  Epoch ごとの指標グラフ")
        print(f"{'=' * 56}")
        for model_size in args.model_sizes:
            plot_epoch_metrics(model_size, args.checkpoint_dir, args.image_dir)

    # --chrf_curve 指定時は epoch 別チェックポイントから Perplexity/ChrF を計算する
    if args.chrf_curve:
        print(f"\n{'=' * 56}")
        print("  Epoch ごとの Perplexity/ChrF 計算")
        print(f"{'=' * 56}")
        for model_size in args.model_sizes:
            print(f"  Model: {model_size}")
            records = evaluate_epoch_checkpoints(
                model_size=model_size,
                checkpoint_dir=args.checkpoint_dir,
                src_vocab_size=len(src_tokenizer),
                tgt_vocab_size=len(tgt_tokenizer),
                max_seq_len=args.max_len * 2,
                val_loader=val_loader,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                device=device,
                max_len=args.max_len,
                chrf_samples=args.chrf_samples,
            )
            # サイズごとに別ディレクトリ・別ファイル名で保存
            plot_metrics_over_epochs(model_size, records, args.image_dir)


if __name__ == "__main__":
    main()
