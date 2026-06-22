"""異なるモデルサイズでの翻訳性能の比較

学習済みチェックポイント (`checkpoints/translation_{size}_best.pt`) を読み込み、
以下の 3 点を出力する評価スクリプト:

  1. Perplexity (検証データの困惑度 — 低いほど良い)
  2. ChrF スコア (文字 n-gram F値 — 高いほど良い)
  3. 任意の入力文に対する翻訳結果

使用例:
    # 単一モデルを評価
    uv run evaluate.py --model_size tiny

    # 学習済み全モデルを横並びで比較
    uv run evaluate.py --compare

    # 任意の英文を翻訳
    uv run evaluate.py --model_size large \
        --translate "I will check the schedule ." "Thank you for your help ."
"""

import argparse
import os
from typing import List, Optional, Tuple

import torch
from data_loader import BOS_IDX, EOS_IDX, create_data_loaders
from tqdm import tqdm
from training_utils import evaluate as evaluate_loss
from training_utils import get_device
from transformer_skeleton import TranslationModel, get_model_config

# モデルサイズ一覧 (table の順)
MODEL_SIZES = ["tiny", "small", "medium", "large"]

# 翻訳サンプル (任意入力が指定されなかった場合のデフォルト)
DEFAULT_SAMPLES = [
    "I will check the schedule .",
    "Thank you for your help .",
    "Please send me the report .",
    "The meeting is at three o'clock .",
]


# ---------------------------------------------------------------------------
# モデル読み込み / 翻訳ヘルパー
# ---------------------------------------------------------------------------
def build_model(
    model_size: str,
    src_vocab_size: int,
    tgt_vocab_size: int,
    max_len: int,
    device: torch.device,
) -> TranslationModel:
    """指定サイズの TranslationModel を構築する"""
    config = get_model_config(model_size)
    model = TranslationModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_encoder_layers=config["n_encoder_layers"],
        n_decoder_layers=config["n_decoder_layers"],
        d_ff=config["d_ff"],
        max_seq_len=max_len * 2,
    ).to(device)
    return model


def load_checkpoint(
    model: TranslationModel,
    model_size: str,
    ckpt_dir: str,
    device: torch.device,
) -> Optional[str]:
    """`translation_{size}_best.pt` を読み込み、見つからなければ None を返す"""
    ckpt_path = os.path.join(ckpt_dir, f"translation_{model_size}_best.pt")
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt_path


def translate_sentence(
    model: TranslationModel,
    sentence: str,
    src_tokenizer,
    tgt_tokenizer,
    device: torch.device,
    max_len: int,
) -> str:
    """1 文を翻訳して日本語文字列を返す"""
    ids = src_tokenizer.encode(sentence, add_eos=True, max_len=max_len)
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    generated = model.generate(src, BOS_IDX, EOS_IDX, max_len=max_len)
    return tgt_tokenizer.decode(generated[0].cpu().tolist())


# ---------------------------------------------------------------------------
# 評価指標
# ---------------------------------------------------------------------------
def compute_perplexity(model, val_loader, device) -> Tuple[float, float]:
    """検証データのロスと Perplexity を返す"""
    val_loss, val_ppl = evaluate_loss(model, val_loader, device)
    return val_loss, val_ppl


def compute_chrf(
    model: TranslationModel,
    val_loader,
    tgt_tokenizer,
    device: torch.device,
    max_len: int,
    max_eval_samples: int = 500,
) -> Optional[float]:
    """検証データの一部から ChrF スコアを計算する

    Args:
        max_eval_samples: 評価に使う最大サンプル数 (生成が遅いので制限)

    Returns:
        ChrF スコア (0〜100)。`sacrebleu` 未インストール時は None
    """
    try:
        from sacrebleu import corpus_chrf
    except ImportError:
        print("[WARN] sacrebleu がインストールされていないため ChrF を計算できません")
        return None

    hypotheses: List[str] = []
    references: List[str] = []

    model.eval()
    with torch.no_grad():
        for src, _tgt_in, tgt_out in tqdm(val_loader, desc="Generating for ChrF"):
            src = src.to(device)
            generated = model.generate(src, BOS_IDX, EOS_IDX, max_len=max_len)
            for i in range(src.size(0)):
                hyp = tgt_tokenizer.decode(generated[i].cpu().tolist())
                ref = tgt_tokenizer.decode(tgt_out[i].cpu().tolist())
                hypotheses.append(hyp)
                references.append(ref)
                if len(hypotheses) >= max_eval_samples:
                    break
            if len(hypotheses) >= max_eval_samples:
                break

    # corpus_chrf は (hypotheses, [list_of_references]) を取る
    chrf = corpus_chrf(hypotheses, [references])
    return chrf.score


# ---------------------------------------------------------------------------
# 評価実行
# ---------------------------------------------------------------------------
def evaluate_single(
    model_size: str,
    src_tokenizer,
    tgt_tokenizer,
    val_loader,
    device: torch.device,
    args: argparse.Namespace,
    sample_sentences: List[str],
) -> Optional[dict]:
    """1 モデルを評価して結果を辞書で返す"""
    print(f"\n{'=' * 60}")
    print(f"  Model size: {model_size}")
    print(f"{'=' * 60}")

    model = build_model(
        model_size=model_size,
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        max_len=args.max_len,
        device=device,
    )
    ckpt_path = load_checkpoint(model, model_size, args.ckpt_dir, device)
    if ckpt_path is None:
        print(
            f"[SKIP] checkpoint not found: "
            f"{os.path.join(args.ckpt_dir, f'translation_{model_size}_best.pt')}"
        )
        return None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"  Parameters  : {n_params:,} ({n_params / 1e6:.2f}M)")

    # 1. Perplexity ---------------------------------------------------------
    val_loss, val_ppl = compute_perplexity(model, val_loader, device)
    print(f"  Val Loss    : {val_loss:.4f}")
    print(f"  Perplexity  : {val_ppl:.2f}    (低いほど良い)")

    # 2. ChrF ---------------------------------------------------------------
    chrf = compute_chrf(
        model,
        val_loader,
        tgt_tokenizer,
        device,
        max_len=args.max_len,
        max_eval_samples=args.chrf_samples,
    )
    if chrf is not None:
        print(
            f"  ChrF        : {chrf:.2f}    "
            f"(高いほど良い, n={args.chrf_samples} sentences)"
        )

    # 3. 任意入力の翻訳結果 ---------------------------------------------------
    print(f"\n  --- Sample translations ({model_size}) ---")
    for sent in sample_sentences:
        translation = translate_sentence(
            model, sent, src_tokenizer, tgt_tokenizer, device, args.max_len
        )
        print(f"  EN: {sent}")
        print(f"  JA: {translation}")

    return {
        "model_size": model_size,
        "params": n_params,
        "val_loss": val_loss,
        "perplexity": val_ppl,
        "chrf": chrf,
    }


def print_comparison_table(results: List[dict]) -> None:
    """全モデルの結果を表形式で表示する"""
    if not results:
        print("\n評価結果がありません。")
        return

    print(f"\n{'=' * 72}")
    print("  Scaling experiment results")
    print(f"{'=' * 72}")
    header = (
        f"  {'Model':<8} {'Params':>12} {'Val Loss':>10} {'Perplexity':>12} {'ChrF':>8}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for r in results:
        chrf_str = f"{r['chrf']:.2f}" if r["chrf"] is not None else "  -"
        print(
            f"  {r['model_size']:<8} "
            f"{r['params'] / 1e6:>10.2f}M "
            f"{r['val_loss']:>10.4f} "
            f"{r['perplexity']:>12.2f} "
            f"{chrf_str:>8}"
        )
    print(f"  {'-' * (len(header) - 2)}")
    print("  (Perplexity は低いほど良い / ChrF は高いほど良い)\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transformer 英日翻訳モデルの評価 (Perplexity / ChrF / 翻訳結果)"
    )
    parser.add_argument(
        "--model_size",
        choices=MODEL_SIZES,
        default="tiny",
        help="評価対象のモデルサイズ (--compare 指定時は無視)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="学習済みの全モデルサイズを一括比較する",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="checkpoints",
        help="チェックポイントの保存ディレクトリ",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=64,
        help="最大系列長 (学習時と同じ値を指定)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="評価時のバッチサイズ",
    )
    parser.add_argument(
        "--src_vocab_size",
        type=int,
        default=8000,
        help="英語語彙サイズ (学習時と同じ値を指定)",
    )
    parser.add_argument(
        "--tgt_vocab_size",
        type=int,
        default=4000,
        help="日本語語彙サイズ (学習時と同じ値を指定)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100_000,
        help="データセットから読み込む最大サンプル数 (学習時と同じ値を指定)",
    )
    parser.add_argument(
        "--chrf_samples",
        type=int,
        default=500,
        help="ChrF 計算に使う検証サンプル数 (生成が遅いので制限)",
    )
    parser.add_argument(
        "--translate",
        nargs="+",
        default=None,
        help="翻訳したい英文 (複数指定可)。未指定ならデフォルトサンプル文を使用",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    # データローダーとトークナイザーを構築 (学習時と同じ手順で再現する)
    print("\nLoading BSD en→ja dataset and building tokenizers ...")
    _, val_loader, src_tokenizer, tgt_tokenizer = create_data_loaders(
        max_len=args.max_len,
        batch_size=args.batch_size,
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.tgt_vocab_size,
        max_samples=args.max_samples,
    )
    print(f"Vocab: en={len(src_tokenizer)} (word), ja={len(tgt_tokenizer)} (char)\n")

    sample_sentences = args.translate if args.translate else DEFAULT_SAMPLES

    # 評価対象のモデルサイズを決定
    if args.compare:
        target_sizes = MODEL_SIZES
    else:
        target_sizes = [args.model_size]

    results: List[dict] = []
    for size in target_sizes:
        res = evaluate_single(
            model_size=size,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            val_loader=val_loader,
            device=device,
            args=args,
            sample_sentences=sample_sentences,
        )
        if res is not None:
            results.append(res)

    # --compare 時は最後に比較表をまとめて表示
    if args.compare:
        print_comparison_table(results)


if __name__ == "__main__":
    main()
