#!/usr/bin/env python3
"""
Main training script for Ex6 B4 Lecture - 英日翻訳タスク
Encoder-Decoder Transformer による英語→日本語翻訳

使用例:
    python main.py --model_size tiny  --epochs 10
    python main.py --model_size small --epochs 30 --batch_size 128
"""

import argparse

from data_loader import create_data_loaders
from training_utils import count_parameters, get_device, setup_logging, train_model
from transformer_answer import TranslationModel, get_model_config


def main():
    parser = argparse.ArgumentParser(description="Train Transformer en→ja translation model")
    parser.add_argument(
        "--model_size",
        choices=["tiny", "small", "medium", "large"],
        default="tiny",
        help="Model size",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_len", type=int, default=64, help="Max sequence length")
    parser.add_argument("--lr", type=float, help="Learning rate (auto if not set)")
    parser.add_argument(
        "--src_vocab_size", type=int, default=8000, help="English vocabulary size (word-level)"
    )
    parser.add_argument(
        "--tgt_vocab_size", type=int, default=4000,
        help="Japanese vocabulary size (char-level)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=100_000, help="Max training samples"
    )
    parser.add_argument(
        "--save_dir", default="checkpoints", help="Checkpoint save directory"
    )
    parser.add_argument("--warmup_steps", type=int, default=4000, help="Warmup steps")
    parser.add_argument(
        "--grad_accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    args = parser.parse_args()

    log_file = f"training_{args.model_size}.log"
    logger = setup_logging(log_file)
    logger.info("=== Transformer 英日翻訳 Ex6 ===")
    logger.info(f"Model size: {args.model_size}")

    device = get_device()

    # データ読み込み (BSD en→ja)
    logger.info("Loading BSD en→ja dataset ...")
    try:
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
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        logger.error("pip install datasets  を実行してください")
        return

    # モデル作成
    config = get_model_config(args.model_size)
    try:
        model = TranslationModel(
            src_vocab_size=len(src_tokenizer),
            tgt_vocab_size=len(tgt_tokenizer),
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_encoder_layers=config["n_encoder_layers"],
            n_decoder_layers=config["n_decoder_layers"],
            d_ff=config["d_ff"],
            max_seq_len=args.max_len * 2,
        ).to(device)
        n_params = count_parameters(model)
        logger.info(f"Model: {n_params:,} parameters ({n_params / 1e6:.2f}M)")
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        return

    # 学習率
    if args.lr is None:
        lr_map = {"tiny": 1e-3, "small": 5e-4, "medium": 3e-4, "large": 1e-4}
        args.lr = lr_map[args.model_size]
    logger.info(f"Learning rate: {args.lr}")

    # 学習
    sample_sentences = [
        "I will check the schedule .",
        "Thank you for your help .",
        "Please send me the report .",
        "The meeting is at three o'clock .",
    ]
    try:
        model_name = f"translation_{args.model_size}"
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
            save_dir=args.save_dir,
            model_name=model_name,
            warmup_steps=args.warmup_steps,
            grad_accumulation_steps=args.grad_accumulation,
            sample_sentences=sample_sentences,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            sample_max_len=args.max_len,
        )
        logger.info("Training completed!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    logger.info("All done!")


if __name__ == "__main__":
    main()
