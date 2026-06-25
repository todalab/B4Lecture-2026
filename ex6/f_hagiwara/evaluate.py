#!/usr/bin/env python3
"""
evaluate.py - モデルサイズ別 翻訳性能評価スクリプト.
英日翻訳 Transformer (Ex6 B4講義)

評価指標:
  - Perplexity  (困惑度)        : 低いほど良い
  - ChrF スコア (文字 n-gram F値): 高いほど良い
  - 任意入力に対する翻訳結果

使用例:
    # 単一モデルを評価
    python evaluate.py --model_size tiny

    # チェックポイントを明示指定
    python evaluate.py --model_size small
    --checkpoint checkpoints/translation_small_best.pt

    # 全サイズを横断比較
    python evaluate.py --compare

    # 評価後に対話翻訳モードへ移行
    python evaluate.py --model_size tiny --interactive
"""

import argparse
import math
import os
import sys
from typing import Dict, List, Tuple  # [fix1] 未使用の Optional を削除

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# sacrebleu が使えるときは公式実装を優先
try:
    from sacrebleu.metrics import CHRF as SacrebleuCHRF

    _HAS_SACREBLEU = True
except ImportError:
    _HAS_SACREBLEU = False

from data_loader import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    create_data_loaders,
)
from training_utils import get_device, setup_logging
from transformer_coded import TranslationModel, get_model_config

logger = setup_logging("evaluate.log")


# ─────────────────────────────────────────────────────────────
# ChrF 実装 (sacrebleu が無い場合のフォールバック)
# ─────────────────────────────────────────────────────────────
def _ngrams(text: str, n: int) -> Dict[str, int]:
    """文字 n-gram の出現カウントを返す."""
    counts: Dict[str, int] = {}
    for i in range(len(text) - n + 1):
        ng = text[i : i + n]
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _chrf_pair(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """1ペアの ChrF スコア (0-100) を計算."""
    total_prec = total_rec = 0.0
    valid = 0
    for n in range(1, max_n + 1):
        hyp_ng = _ngrams(hyp, n)
        ref_ng = _ngrams(ref, n)
        if not hyp_ng and not ref_ng:
            continue
        match = sum(min(hyp_ng.get(k, 0), v) for k, v in ref_ng.items())
        prec = match / sum(hyp_ng.values()) if hyp_ng else 0.0
        rec = match / sum(ref_ng.values()) if ref_ng else 0.0
        total_prec += prec
        total_rec += rec
        valid += 1
    if valid == 0:
        return 0.0
    p = total_prec / valid
    r = total_rec / valid
    denom = beta**2 * p + r
    if denom == 0:
        return 0.0
    return (1 + beta**2) * p * r / denom * 100.0


def compute_chrf(hypotheses: List[str], references: List[str]) -> float:
    """コーパス全体の平均 ChrF スコアを計算."""
    if _HAS_SACREBLEU:
        return SacrebleuCHRF().corpus_score(hypotheses, [references]).score
    scores = [_chrf_pair(h, r) for h, r in zip(hypotheses, references)]
    return sum(scores) / len(scores) if scores else 0.0


# ─────────────────────────────────────────────────────────────
# モデル読み込み
# ─────────────────────────────────────────────────────────────
def load_model(
    checkpoint_path: str,
    model_size: str,
    src_vocab_size: int,
    tgt_vocab_size: int,
    max_seq_len: int,
    device: torch.device,
) -> TranslationModel:
    """チェックポイントからモデルを復元する."""
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    epoch_info = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", float("nan"))
    logger.info(
        f"Loaded: {checkpoint_path} "
        f"(epoch={epoch_info}, saved_val_loss={val_loss:.4f})"
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """パラメータ数を数える."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# Perplexity 計算
# ─────────────────────────────────────────────────────────────
def compute_perplexity(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """検証セット全体の Perplexity を計算する."""
    model.eval()
    # [fix3] バッチ平均ではなくトークン数で重み付けした正確な平均に修正
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt_in, tgt_out in tqdm(val_loader, desc="  Perplexity", leave=False):
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            _, loss = model(src, tgt_in, targets=tgt_out)
            n_tokens = (tgt_out != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    avg_loss = total_loss / total_tokens
    # overflow 対策 (loss > 20 は exp が inf になる)
    return math.exp(min(avg_loss, 20))


# ─────────────────────────────────────────────────────────────
# ChrF 計算
# ─────────────────────────────────────────────────────────────
def compute_chrf_score(
    model: nn.Module,
    val_loader: DataLoader,
    tgt_tokenizer,
    device: torch.device,
    max_len: int,
    n_samples: int = 500,
) -> float:
    """検証セットから n_samples 文を翻訳して ChrF を計算する."""
    model.eval()
    hypotheses: List[str] = []
    references: List[str] = []

    with torch.no_grad():
        for src, _, tgt_out in tqdm(val_loader, desc="  ChrF     ", leave=False):
            if len(hypotheses) >= n_samples:
                break
            src = src.to(device)
            generated = model.generate(src, BOS_IDX, EOS_IDX, max_len=max_len)
            for i in range(src.size(0)):
                if len(hypotheses) >= n_samples:
                    break
                hyp = tgt_tokenizer.decode(generated[i].cpu().tolist())
                ref = tgt_tokenizer.decode(tgt_out[i].tolist())
                hypotheses.append(hyp)
                references.append(ref)

    return compute_chrf(hypotheses, references)


# ─────────────────────────────────────────────────────────────
# 翻訳 (任意入力)
# ─────────────────────────────────────────────────────────────
def translate(
    model: nn.Module,
    sentences: List[str],
    src_tokenizer,
    tgt_tokenizer,
    device: torch.device,
    max_len: int,
) -> List[Tuple[str, str]]:
    """英文リストを日本語へ翻訳して (原文, 訳文) のリストを返す."""
    model.eval()
    results: List[Tuple[str, str]] = []
    with torch.no_grad():
        for sent in sentences:
            ids = src_tokenizer.encode(sent, add_eos=True, max_len=max_len)
            src_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            gen = model.generate(src_t, BOS_IDX, EOS_IDX, max_len=max_len)
            ja = tgt_tokenizer.decode(gen[0].cpu().tolist())
            results.append((sent, ja))
    return results


# ─────────────────────────────────────────────────────────────
# 対話翻訳モード
# ─────────────────────────────────────────────────────────────
def interactive_mode(
    model: nn.Module,
    src_tokenizer,
    tgt_tokenizer,
    device: torch.device,
    max_len: int,
):
    """対話翻訳モード."""
    print("\n" + "=" * 55)
    print("  対話翻訳モード  (終了: 'q' または Ctrl-C)")
    print("=" * 55)
    while True:
        try:
            text = input("EN> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        if not text:
            continue
        results = translate(
            model, [text], src_tokenizer, tgt_tokenizer, device, max_len
        )
        print(f"JA> {results[0][1]}\n")


# ─────────────────────────────────────────────────────────────
# 結果テーブル表示
# ─────────────────────────────────────────────────────────────
def print_table(rows: List[Dict]):
    """評価結果を整形して表示する."""
    W = [10, 14, 10, 14]
    hdr = (
        f"{'Model':<{W[0]}}  "
        f"{'Perplexity':>{W[1]}}  "
        f"{'ChrF':>{W[2]}}  "
        f"{'Params':>{W[3]}}"
    )
    sep = "-" * (sum(W) + 8)
    print("\n" + sep)
    print(hdr)
    print(sep)
    for r in rows:
        print(
            f"{r['model_size']:<{W[0]}}  "
            f"{r['perplexity']:>{W[1]}.2f}  "
            f"{r['chrf']:>{W[2]}.2f}  "
            f"{r['n_params']:>{W[3]},}"
        )
    print(sep)

    # 最良モデルの強調
    if len(rows) > 1:
        best_ppl = min(rows, key=lambda r: r["perplexity"])
        best_chrf = max(rows, key=lambda r: r["chrf"])
        print(
            f"\n  最低 Perplexity : {best_ppl['model_size']}  "
            f"({best_ppl['perplexity']:.2f})"
        )
        print(
            f"  最高 ChrF       : {best_chrf['model_size']}  "
            f"({best_chrf['chrf']:.2f})"
        )
    print()


# ─────────────────────────────────────────────────────────────
# 単一モデルの評価フロー
# ─────────────────────────────────────────────────────────────
def evaluate_one(
    model_size: str,
    checkpoint_path: str,
    val_loader: DataLoader,
    src_tokenizer,
    tgt_tokenizer,
    device: torch.device,
    max_len: int,
    sample_sentences: List[str],
    chrf_samples: int,
) -> Dict:
    """単一モデルの評価."""
    bar = "=" * 55
    print(f"\n{bar}")
    print(f"  Model : {model_size}   ({checkpoint_path})")
    print(bar)

    model = load_model(
        checkpoint_path,
        model_size,
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        max_seq_len=max_len * 2,
        device=device,
    )
    n_params = count_parameters(model)
    print(f"  パラメータ数 : {n_params:,}")

    # ─── [1] Perplexity ───
    print("\n[1/3] Perplexity を計算中 ...")
    ppl = compute_perplexity(model, val_loader, device)
    print(f"  → Perplexity : {ppl:.2f}")

    # ─── [2] ChrF ───
    print(f"\n[2/3] ChrF を計算中 (サンプル数={chrf_samples}) ...")
    chrf = compute_chrf_score(
        model, val_loader, tgt_tokenizer, device, max_len, chrf_samples
    )
    print(f"  → ChrF       : {chrf:.2f}")

    # ─── [3] サンプル翻訳 ───
    print("\n[3/3] サンプル翻訳 ...")
    translations = translate(
        model, sample_sentences, src_tokenizer, tgt_tokenizer, device, max_len
    )
    for en, ja in translations:
        print(f"  EN: {en}")
        print(f"  JA: {ja}")
        print()

    return {
        "model_size": model_size,
        "perplexity": ppl,
        "chrf": chrf,
        "n_params": n_params,
        "_model": model,  # 対話モード用 (テーブルには出ない)
    }


# ─────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────
def main():
    """main関数."""
    parser = argparse.ArgumentParser(
        description="Evaluate en→ja Transformer model(s)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model_size",
        choices=[
            "tiny",
            "small",
            "medium",
            "large",
            "small_GA",
            "small_GA8",
            "small1e3",
            "small1e2",
        ],
        help="評価するモデルサイズ (--compare 使用時は不要)",
    )
    parser.add_argument(
        "--checkpoint",
        help="チェックポイントファイルのパス (省略時は save_dir から自動検索)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="全サイズを横断比較する",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="評価後に対話翻訳モードへ移行 (単一モデル時のみ)",
    )
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--src_vocab_size", type=int, default=8000)
    parser.add_argument("--tgt_vocab_size", type=int, default=4000)
    parser.add_argument("--max_samples", type=int, default=100_000)
    parser.add_argument(
        "--chrf_samples",
        type=int,
        default=500,
        help="ChrF 計算に使う文数 (多いほど正確、遅い)",
    )
    args = parser.parse_args()

    if not args.compare and not args.model_size:
        parser.error("--model_size または --compare を指定してください")

    device = get_device()

    # ── データ読み込み ──────────────────────────────────────────
    logger.info("検証データを読み込み中 ...")
    try:
        _, val_loader, src_tokenizer, tgt_tokenizer = create_data_loaders(
            max_len=args.max_len,
            batch_size=args.batch_size,
            src_vocab_size=args.src_vocab_size,
            tgt_vocab_size=args.tgt_vocab_size,
            max_samples=args.max_samples,
        )
        logger.info(
            f"語彙サイズ: en={len(src_tokenizer)} (word), "
            f"ja={len(tgt_tokenizer)} (char)"
        )
    except Exception as e:
        logger.error(f"データ読み込み失敗: {e}")
        logger.error("pip install datasets  を実行してください")
        sys.exit(1)

    # 固定サンプル文
    sample_sentences = [
        "I will check the schedule .",
        "Thank you for your help .",
        "Please send me the report .",
        "The meeting is at three o'clock .",
        "I understand your concern .",
        "Could you please repeat that ?",
        "Let me confirm the details .",
        "We need to finalize the budget by Friday .",
        "I will follow up with you later today .",
        "Let us discuss this matter in more detail .",
        "I am a student of Nagoya University .",
        "The train came out of the long tunnel into the snow country .",
    ]

    all_results: List[Dict] = []

    if args.compare:
        # [fix2] --interactive との併用を検知して警告
        if args.interactive:
            logger.warning("--interactive は --compare と併用できません。無視します。")

        # ── 全サイズ横断評価 ─────────────────────────────────────
        sizes = [
            "tiny",
            "small",
            "medium",
            "large",
            "small_GA",
            "small_GA8",
            "small1e3",
            "small1e2",
        ]
        for size in sizes:
            ckpt = os.path.join(args.save_dir, f"translation_{size}_best.pt")
            if not os.path.exists(ckpt):
                logger.warning(f"チェックポイントが見つかりません。スキップ: {ckpt}")
                continue
            result = evaluate_one(
                model_size=size,
                checkpoint_path=ckpt,
                val_loader=val_loader,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                device=device,
                max_len=args.max_len,
                sample_sentences=sample_sentences,
                chrf_samples=args.chrf_samples,
            )
            all_results.append(result)

        if all_results:
            print_table(all_results)
        else:
            print("評価できたモデルがありませんでした。")

    else:
        # ── 単一モデル評価 ───────────────────────────────────────
        ckpt = args.checkpoint or os.path.join(
            args.save_dir, f"translation_{args.model_size}_best.pt"
        )
        if not os.path.exists(ckpt):
            logger.error(f"チェックポイントが見つかりません: {ckpt}")
            sys.exit(1)

        result = evaluate_one(
            model_size=args.model_size,
            checkpoint_path=ckpt,
            val_loader=val_loader,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
            max_len=args.max_len,
            sample_sentences=sample_sentences,
            chrf_samples=args.chrf_samples,
        )
        all_results.append(result)
        print_table(all_results)

        if args.interactive:
            interactive_mode(
                result["_model"],
                src_tokenizer,
                tgt_tokenizer,
                device,
                args.max_len,
            )


if __name__ == "__main__":
    main()
