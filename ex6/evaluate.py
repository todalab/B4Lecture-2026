#!/usr/bin/env python3
"""evaluate.py — モデルサイズ別 翻訳性能比較スクリプト.

異なるサイズの Transformer モデルを評価し、以下の 3 点を出力する:
    1. Perplexity（困惑度）: 低いほど性能が良い
    2. ChrF スコア: 高いほど性能が良い
    3. 任意の入力文に対する翻訳結果

使用例:
    python evaluate.py --model_sizes tiny small
    python evaluate.py --model_sizes tiny small medium large
    python evaluate.py --model_sizes small --sentences "Good morning." "See you tomorrow."
"""

import argparse
import logging
import math
import os

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
):
    """チェックポイントからモデルを読み込む.

    Args:
        model_size: モデルサイズ ("tiny" / "small" / "medium" / "large")
        checkpoint_dir: チェックポイントディレクトリ
        src_vocab_size: ソース語彙サイズ
        tgt_vocab_size: ターゲット語彙サイズ
        max_seq_len: 最大系列長
        device: 使用デバイス

    Returns:
        評価モードの TranslationModel。チェックポイントが存在しない場合は None。
    """
    ckpt_path = os.path.join(checkpoint_dir, f"translation_{model_size}_best.pt")
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return None

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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(
        f"Loaded '{model_size}' from {ckpt_path} (epoch {ckpt.get('epoch', '?')})"
    )
    return model


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
    with torch.no_grad():
        for src, tgt_in, tgt_out in tqdm(val_loader, desc="  Perplexity", leave=False):
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            _, loss = model(src, tgt_in, targets=tgt_out)
            total_loss += loss.item()
            num_batches += 1
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
    hypotheses = []
    references = []
    count = 0
    with torch.no_grad():
        for src_batch, _, tgt_out_batch in tqdm(val_loader, desc="  ChrF", leave=False):
            src_batch = src_batch.to(device)
            generated = model.generate(src_batch, BOS_IDX, EOS_IDX, max_len=max_len)
            for gen_ids, ref_ids in zip(generated, tgt_out_batch):
                hyp = tgt_tokenizer.decode(gen_ids.cpu().tolist())
                ref = tgt_tokenizer.decode(ref_ids.tolist())
                hypotheses.append(hyp)
                references.append(ref)
                count += 1
                if count >= n_samples:
                    break
            if count >= n_samples:
                break
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
            src_ids = src_tokenizer.encode(sent, add_eos=True, max_len=max_len)
            src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
            generated = model.generate(src_tensor, BOS_IDX, EOS_IDX, max_len=max_len)
            translation = tgt_tokenizer.decode(generated[0].cpu().tolist())
            print(f"    EN: {sent}")
            print(f"    JA: {translation}")


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
    args = parser.parse_args()

    device = get_device()
    sentences = args.sentences if args.sentences else DEFAULT_SENTENCES

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

    results = {}

    for model_size in args.model_sizes:
        print(f"\n{'=' * 56}")
        print(f"  Model: {model_size}")
        print(f"{'=' * 56}")

        model = load_model(
            model_size=model_size,
            checkpoint_dir=args.checkpoint_dir,
            src_vocab_size=len(src_tokenizer),
            tgt_vocab_size=len(tgt_tokenizer),
            max_seq_len=args.max_len * 2,
            device=device,
        )
        if model is None:
            print(f"  [SKIP] Checkpoint not found for '{model_size}'")
            print(f"  先に: python main.py --model_size {model_size}")
            continue

        ppl = compute_perplexity(model, val_loader, device)
        print(f"  Perplexity : {ppl:.2f}  (低いほど良い)")

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

        print(f"  翻訳例 ({len(sentences)} 文):")
        show_translations(
            model, sentences, src_tokenizer, tgt_tokenizer, device, args.max_len
        )

        results[model_size] = {"perplexity": ppl, "chrf": chrf_score}

    if results:
        print(f"\n{'=' * 56}")
        print("  比較サマリー")
        print(f"{'=' * 56}")
        header = f"  {'Size':<8} {'Perplexity':>12} {'ChrF':>8}"
        print(header)
        print(f"  {'-' * 30}")
        for size in args.model_sizes:
            if size in results:
                r = results[size]
                print(f"  {size:<8} {r['perplexity']:>12.2f} {r['chrf']:>8.2f}")
        print(f"{'=' * 56}")


if __name__ == "__main__":
    main()
