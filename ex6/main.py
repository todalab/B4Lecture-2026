#!/usr/bin/env python3
"""
Main training script for Ex6 B4 Lecture
Transformer implementation and scaling experiments

使用例:
python main.py --model_size tiny --dataset shakespeare --epochs 10
python main.py --model_size small --dataset wikitext2 --epochs 20
"""

import argparse
import logging
import os
import torch

# ローカルモジュールをインポート
from transformer_skeleton import LanguageModel, get_model_config
from data_loader import create_data_loaders
from training_utils import train_model, count_parameters, get_device, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train Transformer language model")
    parser.add_argument("--model_size", choices=["tiny", "small", "medium", "large"],
                       default="tiny", help="Model size")
    parser.add_argument("--dataset", choices=["shakespeare", "wikitext2"],
                       default="shakespeare", help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--lr", type=float, help="Learning rate (auto if not set)")
    parser.add_argument("--save_dir", default="checkpoints", help="Save directory")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--grad_accumulation", type=int, default=1, help="Gradient accumulation steps")

    args = parser.parse_args()

    # ロギング設定
    log_file = f"training_{args.model_size}_{args.dataset}.log"
    logger = setup_logging(log_file)

    logger.info("=== Transformer Training Ex6 ===")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Epochs: {args.epochs}")

    # デバイス設定
    device = get_device()

    # データローダー作成
    logger.info("Loading data...")
    try:
        train_loader, val_loader, vocab_size, encode_fn, decode_fn = create_data_loaders(
            args.dataset, args.seq_len, args.batch_size, data_dir=args.data_dir
        )
        logger.info(f"Data loaded successfully. Vocab size: {vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.error("Make sure the data directory exists and contains the required files.")
        logger.error(f"Expected: {args.data_dir}/shakespeare.txt or {args.data_dir}/wikitext-2/{{train,valid}}.txt")
        return

    # モデル設定
    config = get_model_config(args.model_size)
    logger.info(f"Model config: {config}")

    # モデル作成
    try:
        model = LanguageModel(
            vocab_size=vocab_size,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_seq_len=args.seq_len * 2  # 生成時により長いシーケンスを許可
        ).to(device)

        param_count = count_parameters(model)
        logger.info(f"Model created: {param_count:,} parameters ({param_count/1e6:.2f}M)")

    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        logger.error("Make sure you have implemented the required parts in transformer_skeleton.py")
        logger.error("Look for ★ markers indicating implementation needed")
        return

    # 学習率設定
    if args.lr is None:
        lr_map = {"tiny": 3e-4, "small": 1e-4, "medium": 5e-5, "large": 1e-5}
        args.lr = lr_map[args.model_size]

    logger.info(f"Learning rate: {args.lr}")

    # 学習実行
    try:
        logger.info("Starting training...")
        model_name = f"{args.model_size}_{args.dataset}"

        metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
            save_dir=args.save_dir,
            model_name=model_name,
            warmup_steps=args.warmup_steps,
            grad_accumulation_steps=args.grad_accumulation
        )

        logger.info("Training completed successfully!")

        # 簡単なテキスト生成例
        logger.info("Generating sample text...")
        try:
            # 最良のチェックポイントを読み込み
            checkpoint_path = os.path.join(args.save_dir, f"{model_name}_best.pt")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # サンプル生成
            if args.dataset == "shakespeare":
                prompt = "To be or not to be"
            else:
                prompt = "artificial intelligence is"

            model.eval()
            context = encode_fn(prompt)
            context = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)

            with torch.no_grad():
                generated = model.generate(context, max_tokens=50, temperature=0.8)

            generated_text = decode_fn(generated[0].cpu().tolist())
            logger.info(f"Generated text: {generated_text[:200]}...")

        except Exception as e:
            logger.warning(f"Text generation failed: {e}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("This might be due to unimplemented parts in transformer_skeleton.py")
        return

    logger.info("All done!")


if __name__ == "__main__":
    main()