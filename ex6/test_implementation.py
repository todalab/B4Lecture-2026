#!/usr/bin/env python3
"""
Implementation Test for Ex6 B4 Lecture
学生が実装した箇所をテストするスクリプト
"""

import torch
import numpy as np
import sys
import traceback

def test_positional_encoding():
    """PositionalEncoding のテスト"""
    print("Testing PositionalEncoding...")
    try:
        from transformer_skeleton import PositionalEncoding

        pos_enc = PositionalEncoding(d_model=128, max_seq_len=100)
        x = torch.randn(1, 50, 128)  # (batch, seq_len, d_model)

        output = pos_enc(x)

        if output is None or torch.equal(output, x):
            print("  ❌ PositionalEncoding: 実装が必要です")
            return False

        if output.shape != x.shape:
            print(f"  ❌ PositionalEncoding: 出力形状が間違っています {output.shape} != {x.shape}")
            return False

        print("  ✅ PositionalEncoding: OK")
        return True

    except Exception as e:
        print(f"  ❌ PositionalEncoding: エラー - {e}")
        return False

def test_multihead_attention():
    """MultiHeadAttention のテスト"""
    print("Testing MultiHeadAttention...")
    try:
        from transformer_skeleton import MultiHeadAttention

        attention = MultiHeadAttention(d_model=128, n_heads=4)
        x = torch.randn(2, 10, 128)  # (batch, seq_len, d_model)

        output, attn_weights = attention(x, x, x)

        if output is None or torch.equal(output, x):
            print("  ❌ MultiHeadAttention: 実装が必要です")
            return False

        if output.shape != x.shape:
            print(f"  ❌ MultiHeadAttention: 出力形状が間違っています {output.shape} != {x.shape}")
            return False

        print("  ✅ MultiHeadAttention: OK")
        return True

    except Exception as e:
        print(f"  ❌ MultiHeadAttention: エラー - {e}")
        return False

def test_transformer_block():
    """TransformerBlock のテスト"""
    print("Testing TransformerBlock...")
    try:
        from transformer_skeleton import TransformerBlock

        block = TransformerBlock(d_model=128, n_heads=4, d_ff=512)
        x = torch.randn(2, 10, 128)  # (batch, seq_len, d_model)

        output = block(x)

        if output is None or torch.equal(output, x):
            print("  ❌ TransformerBlock: 実装が必要です")
            return False

        if output.shape != x.shape:
            print(f"  ❌ TransformerBlock: 出力形状が間違っています {output.shape} != {x.shape}")
            return False

        print("  ✅ TransformerBlock: OK")
        return True

    except Exception as e:
        print(f"  ❌ TransformerBlock: エラー - {e}")
        return False

def test_language_model():
    """LanguageModel のテスト"""
    print("Testing LanguageModel...")
    try:
        from transformer_skeleton import LanguageModel, get_model_config

        config = get_model_config('tiny')
        model = LanguageModel(
            vocab_size=1000,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_seq_len=64
        )

        x = torch.randint(0, 1000, (2, 32))  # (batch, seq_len)
        y = torch.randint(0, 1000, (2, 32))  # targets

        logits, loss = model(x, y)

        if logits is None:
            print("  ❌ LanguageModel: 出力が None です")
            return False

        expected_shape = (2, 32, 1000)
        if logits.shape != expected_shape:
            print(f"  ❌ LanguageModel: 出力形状が間違っています {logits.shape} != {expected_shape}")
            return False

        if loss is None or not torch.is_tensor(loss):
            print("  ❌ LanguageModel: 損失が計算されていません")
            return False

        # 勾配計算テスト
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in model.parameters() if p.requires_grad)

        if not has_grad:
            print("  ❌ LanguageModel: 勾配が計算されていません")
            return False

        print("  ✅ LanguageModel: OK")
        return True

    except Exception as e:
        print(f"  ❌ LanguageModel: エラー - {e}")
        traceback.print_exc()
        return False

def test_data_loader():
    """DataLoader のテスト"""
    print("Testing DataLoader...")
    try:
        from data_loader import create_data_loaders

        # Shakespeare データテスト
        train_loader, val_loader, vocab_size, encode_fn, decode_fn = create_data_loaders(
            "shakespeare", seq_len=32, batch_size=2, data_dir="data"
        )

        # データ取得テスト
        for x, y in train_loader:
            if x.shape[0] != 2 or x.shape[1] != 32:
                print(f"  ❌ DataLoader: バッチ形状が間違っています {x.shape}")
                return False

            # デコードテスト
            text = decode_fn(x[0].tolist())
            if not isinstance(text, str) or len(text) == 0:
                print("  ❌ DataLoader: デコード機能が動作しません")
                return False

            break

        print("  ✅ DataLoader: OK")
        return True

    except Exception as e:
        print(f"  ❌ DataLoader: エラー - {e}")
        print("     データファイルが存在することを確認してください:")
        print("     - data/shakespeare.txt")
        print("     - data/wikitext-2/train.txt")
        print("     - data/wikitext-2/valid.txt")
        return False

def test_training_step():
    """一回の学習ステップをテスト"""
    print("Testing Training Step...")
    try:
        from transformer_skeleton import LanguageModel, get_model_config
        from data_loader import create_data_loaders

        # 小さなモデルでテスト
        config = get_model_config('tiny')
        config['d_model'] = 64  # さらに小さく

        model = LanguageModel(
            vocab_size=100,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=2,  # レイヤー数も減らす
            d_ff=config['d_ff'],
            max_seq_len=32
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # ダミーデータ
        x = torch.randint(0, 100, (4, 16))
        y = torch.randint(0, 100, (4, 16))

        # 学習ステップ
        model.train()
        optimizer.zero_grad()

        logits, loss = model(x, y)
        loss.backward()

        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        print("  ✅ Training Step: OK")
        return True

    except Exception as e:
        print(f"  ❌ Training Step: エラー - {e}")
        traceback.print_exc()
        return False

def main():
    print("=" * 50)
    print("Ex6 Implementation Test")
    print("=" * 50)

    tests = [
        test_positional_encoding,
        test_multihead_attention,
        test_transformer_block,
        test_language_model,
        test_data_loader,
        test_training_step
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"  ❌ テスト実行エラー: {e}")
            results.append(False)
            print()

    print("=" * 50)
    print("Test Results:")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("実装が完了しています。main.py で学習を開始できます。")
        print("\n推奨コマンド:")
        print("python main.py --model_size tiny --dataset shakespeare --epochs 5")
    else:
        print(f"⚠️  {total - passed} tests failed. ({passed}/{total})")
        print("\n実装が必要な箇所:")
        print("- transformer_skeleton.py の ★ マークの箇所を実装してください")
        print("- data/ ディレクトリにデータファイルがあることを確認してください")

    print("\n詳細なヘルプ:")
    print("- README.md の実装仕様を確認")
    print("- transformer_skeleton.py のコメントを読む")
    print("- 質問があれば TA や先輩に相談")

if __name__ == "__main__":
    main()