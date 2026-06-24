#!/usr/bin/env python3
"""Implementation Test.

実装した Transformer の各コンポーネントをテストするスクリプト

使用方法:
    python test_implementation.py
"""

import traceback
import torch

from transformer_skeleton import MultiHeadAttention, PositionalEncoding, TranslationModel, DecoderBlock, EncoderBlock
from transformer_skeleton import get_model_config


def test_positional_encoding():
    """PositionalEncoding のテスト"""
    print("Testing PositionalEncoding ...")
    try:
        pe = PositionalEncoding(d_model=128, dropout=0.0, max_seq_len=100)
        x = torch.randn(2, 50, 128)
        out = pe(x)

        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
        assert not torch.equal(out, x), "Output should differ from input (PE not added)"

        # 偶数チャンネルは sin、奇数は cos → sin(0) = 0, cos(0) = 1
        pe_val = pe.pe[0, 0, :]  # position=0 の PE
        assert abs(pe_val[0].item()) < 1e-5, "pe[pos=0, i=0] should be sin(0) = 0"
        assert abs(pe_val[1].item() - 1.0) < 1e-5, "pe[pos=0, i=1] should be cos(0) = 1"

        print("  ✅ PositionalEncoding: OK")
        return True
    except Exception as e:
        print(f"  ❌ PositionalEncoding: {e}")
        traceback.print_exc()
        return False


def test_multihead_attention_self():
    """MultiHeadAttention の self-attention テスト"""
    print("Testing MultiHeadAttention (self-attention) ...")
    try:
        attn = MultiHeadAttention(d_model=128, n_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 128)
        out, weights = attn(x, x, x)

        assert out.shape == x.shape, f"Output shape: {out.shape}"
        assert weights.shape == (2, 4, 10, 10), f"Weights shape: {weights.shape}"
        assert not torch.equal(out, x), "Output should differ from input"

        # Attention weights の合計は 1 に近いはず
        row_sum = weights.sum(dim=-1)
        assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4), \
            "Attention weights should sum to 1"

        print("  ✅ MultiHeadAttention (self-attention): OK")
        return True
    except Exception as e:
        print(f"  ❌ MultiHeadAttention (self-attention): {e}")
        traceback.print_exc()
        return False


def test_multihead_attention_cross():
    """MultiHeadAttention の cross-attention テスト (encoder出力を参照)"""
    print("Testing MultiHeadAttention (cross-attention) ...")
    try:
        attn = MultiHeadAttention(d_model=128, n_heads=4, dropout=0.0)
        query = torch.randn(2, 8, 128)   # decoder side (tgt_len=8)
        encoder_out = torch.randn(2, 15, 128)  # encoder side (src_len=15)

        out, weights = attn(query, encoder_out, encoder_out)

        assert out.shape == (2, 8, 128), f"Output shape: {out.shape}"
        assert weights.shape == (2, 4, 8, 15), f"Weights shape: {weights.shape}"

        print("  ✅ MultiHeadAttention (cross-attention): OK")
        return True
    except Exception as e:
        print(f"  ❌ MultiHeadAttention (cross-attention): {e}")
        traceback.print_exc()
        return False


def test_encoder_block():
    """EncoderBlock のテスト"""
    print("Testing EncoderBlock ...")
    try:
        block = EncoderBlock(d_model=128, n_heads=4, d_ff=512, dropout=0.0)
        x = torch.randn(2, 20, 128)
        out = block(x)

        assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
        assert not torch.equal(out, x), "Output should differ from input"

        print("  ✅ EncoderBlock: OK")
        return True
    except Exception as e:
        print(f"  ❌ EncoderBlock: {e}")
        traceback.print_exc()
        return False


def test_decoder_block():
    """DecoderBlock のテスト (cross-attention を含む)"""
    print("Testing DecoderBlock ...")
    try:
        block = DecoderBlock(d_model=128, n_heads=4, d_ff=512, dropout=0.0)
        x = torch.randn(2, 10, 128)            # decoder input  (tgt_len=10)
        encoder_out = torch.randn(2, 20, 128)  # encoder output (src_len=20)
        out = block(x, encoder_out)

        assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
        assert not torch.equal(out, x), "Output should differ from input"

        print("  ✅ DecoderBlock: OK")
        return True
    except Exception as e:
        print(f"  ❌ DecoderBlock: {e}")
        traceback.print_exc()
        return False


def test_translation_model():
    """TranslationModel のフルフォワードテスト"""
    print("Testing TranslationModel ...")
    try:
        config = get_model_config("tiny")
        model = TranslationModel(
            src_vocab_size=500,
            tgt_vocab_size=600,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_encoder_layers=config["n_encoder_layers"],
            n_decoder_layers=config["n_decoder_layers"],
            d_ff=config["d_ff"],
            max_seq_len=64,
        )

        src = torch.randint(1, 500, (2, 20))
        tgt = torch.randint(1, 600, (2, 15))
        tgt_out = torch.randint(1, 600, (2, 15))

        logits, loss = model(src, tgt, targets=tgt_out)

        assert logits.shape == (2, 15, 600), f"logits shape: {logits.shape}"
        assert loss is not None and loss.item() > 0, "Loss should be positive"

        # 勾配計算テスト
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "Gradients should flow through the model"

        print("  ✅ TranslationModel: OK")
        return True
    except Exception as e:
        print(f"  ❌ TranslationModel: {e}")
        traceback.print_exc()
        return False

def test_generate():
    """生成 (greedy decoding) のテスト"""
    print("Testing generate (greedy decoding) ...")
    try:
        model = TranslationModel(
            src_vocab_size=100, tgt_vocab_size=100,
            d_model=64, n_heads=4, n_encoder_layers=1, n_decoder_layers=1,
            d_ff=128, max_seq_len=32,
        )
        model.eval()

        src = torch.randint(1, 100, (2, 10))
        generated = model.generate(src, bos_idx=2, eos_idx=3, max_len=20)

        assert generated.shape[0] == 2, f"Batch size: {generated.shape[0]}"
        assert (generated[:, 0] == 2).all(), "Generated should start with BOS"
        assert generated.shape[1] <= 21, f"Max length exceeded: {generated.shape[1]}"

        print("  ✅ Generate: OK")
        return True
    except Exception as e:
        print(f"  ❌ Generate: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 55)
    print("Ex6 Translation Transformer - Implementation Test")
    print("=" * 55)

    tests = [
        test_positional_encoding,
        test_multihead_attention_self,
        test_multihead_attention_cross,
        test_encoder_block,
        test_decoder_block,
        test_translation_model,
        test_generate,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ テスト実行エラー: {e}")
            results.append(False)
        print()

    print("=" * 55)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("実装完了です。main.py で学習を開始できます。")
        print("\n推奨コマンド:")
        print("  python main.py --model_size tiny --epochs 10")
    else:
        print(f"⚠️  {total - passed} tests failed. ({passed}/{total})")
        print("transformer_skeleton.py の ★ 箇所を確認してください。")


if __name__ == "__main__":
    main()
