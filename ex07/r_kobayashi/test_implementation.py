#!/usr/bin/env python3
"""VAE Implementation Test.

実装した VAE の各メソッドをテストするスクリプト。

使用方法:
    python test_implementation.py
"""

import traceback

import torch
from VAE_skeleton import VAE

# テスト共通パラメータ
Z_DIM = 4
H_DIM = 64
BATCH = 8
DEVICE = torch.device("cpu")


def make_model():
    return VAE(z_dim=Z_DIM, h_dim=H_DIM, drop_rate=0.0).eval()


def test_encoder():
    """encoder の shape・決定論性・識別性テスト"""
    print("Testing VAE.encoder ...")
    try:
        model = make_model()
        x = torch.rand(BATCH, 28 * 28)
        x2 = torch.rand(BATCH, 28 * 28)

        mean1, log_var1 = model.encoder(x)
        mean2, log_var2 = model.encoder(x)  # 同じ入力を2回
        mean3, _ = model.encoder(x2)  # 異なる入力

        # shape
        assert mean1.shape == (BATCH, Z_DIM), f"mean shape: {mean1.shape}"
        assert log_var1.shape == (BATCH, Z_DIM), f"log_var shape: {log_var1.shape}"

        # 決定論性: 同じ入力→同じ出力
        assert torch.allclose(
            mean1, mean2
        ), "encoder が非決定論的です（同じ入力で mean が変わる）"
        assert torch.allclose(
            log_var1, log_var2
        ), "encoder が非決定論的です（同じ入力で log_var が変わる）"

        # 識別性: 異なる入力→異なる出力
        assert not torch.allclose(mean1, mean3), "異なる入力に同じ mean を返しています"

        print("  ✅ encoder: OK")
        return True
    except Exception as e:
        print(f"  ❌ encoder: {e}")
        traceback.print_exc()
        return False


def test_reparametrization_trick():
    """reparametrization_trick の shape と確率性のテスト"""
    print("Testing VAE.reparametrization_trick ...")
    try:
        model = make_model()
        mean = torch.zeros(BATCH, Z_DIM)
        log_var = torch.zeros(BATCH, Z_DIM)

        z1 = model.reparametrization_trick(mean, log_var)
        z2 = model.reparametrization_trick(mean, log_var)

        assert z1.shape == (BATCH, Z_DIM), f"z shape: {z1.shape}"
        assert not torch.equal(
            z1, z2
        ), "2回のサンプリング結果が同一です（確率的でない）"

        print("  ✅ reparametrization_trick: OK")
        return True
    except Exception as e:
        print(f"  ❌ reparametrization_trick: {e}")
        traceback.print_exc()
        return False


def test_reparametrization_trick_formula():
    """reparametrization_trick の数値検証: z = mean + ε * exp(0.5 * log_var) を固定シードで確認"""
    print("Testing VAE.reparametrization_trick (numerical formula check) ...")
    try:
        model = make_model()
        mean = torch.tensor([[2.0, -1.0, 0.5, 0.0]])  # (1, 4)
        log_var = torch.tensor([[1.0, 0.0, 2.0, 0.0]])  # (1, 4)

        # 期待値: 同じシードで生成された ε から手計算
        torch.manual_seed(42)
        epsilon_ref = torch.randn_like(mean)
        z_expected = mean + epsilon_ref * torch.exp(0.5 * log_var)

        # 実装を同じシードで呼ぶ → 同一の ε が使われるはず
        torch.manual_seed(42)
        z_actual = model.reparametrization_trick(mean, log_var)

        assert torch.allclose(z_actual, z_expected, atol=1e-5), (
            f"Reparametrization の計算が正しくありません\n"
            f"  期待値: {z_expected.tolist()}\n"
            f"  実際値: {z_actual.tolist()}"
        )

        print("  ✅ reparametrization_trick (formula): OK")
        return True
    except Exception as e:
        print(f"  ❌ reparametrization_trick (formula): {e}")
        traceback.print_exc()
        return False


def test_decoder():
    """decoder の shape・出力範囲・非定数性のテスト"""
    print("Testing VAE.decoder ...")
    try:
        model = make_model()
        z1 = torch.randn(BATCH, Z_DIM)
        z2 = torch.randn(BATCH, Z_DIM)
        y1 = model.decoder(z1)
        y2 = model.decoder(z2)

        x_dim = 28 * 28
        assert y1.shape == (BATCH, x_dim), f"y shape: {y1.shape}"

        # Sigmoid による値域チェック
        assert y1.min().item() >= 0.0, f"出力に負の値: {y1.min().item():.4f}"
        assert y1.max().item() <= 1.0, f"出力が 1 を超える: {y1.max().item():.4f}"

        # 非定数性: 異なる z → 異なる y
        assert not torch.allclose(
            y1, y2
        ), "異なる z に同じ y を返しています（定数出力）"

        print("  ✅ decoder: OK")
        return True
    except Exception as e:
        print(f"  ❌ decoder: {e}")
        traceback.print_exc()
        return False


def test_forward_shapes():
    """forward の出力 shape テスト"""
    print("Testing VAE.forward (shapes) ...")
    try:
        model = make_model()
        x = torch.rand(BATCH, 28 * 28)
        (elbo_kl, elbo_rec), z, y = model(x)

        assert z.shape == (BATCH, Z_DIM), f"z shape: {z.shape}"
        assert y.shape == (BATCH, 28 * 28), f"y shape: {y.shape}"
        assert elbo_kl.dim() == 0, "elbo_kl はスカラーであるべき"
        assert elbo_rec.dim() == 0, "elbo_rec はスカラーであるべき"

        print("  ✅ forward (shapes): OK")
        return True
    except Exception as e:
        print(f"  ❌ forward (shapes): {e}")
        traceback.print_exc()
        return False


def test_kld_numerical():
    """kld の数値検証: 既知の mean・log_var で解析解と一致するか"""
    print("Testing VAE.kld (numerical formula check) ...")
    try:
        import math

        model = make_model()

        # mean=0, log_var=1.0 を直接注入
        mean = torch.zeros(BATCH, Z_DIM)
        log_var = torch.ones(BATCH, Z_DIM)  # log_var = 1.0

        kl = model.kld(mean, log_var)

        # 期待値: -0.5 * sum(1 + 1 - 0² - e¹) = 0.5 * BATCH * Z_DIM * (e - 2)
        expected = 0.5 * BATCH * Z_DIM * (math.e - 2.0)
        assert abs(kl.item() - expected) < 1e-3, (
            f"kld の数値が正しくありません\n"
            f"  期待値: {expected:.6f}\n"
            f"  実際値: {kl.item():.6f}\n"
            f"  ヒント: KL = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))"
        )

        print("  ✅ kld numerical: OK")
        return True
    except Exception as e:
        print(f"  ❌ kld numerical: {e}")
        traceback.print_exc()
        return False


def test_elbo_signs():
    """elbo_kl ≤ 0 かつ elbo_rec ≤ 0 のテスト"""
    print("Testing ELBO signs (elbo_kl ≤ 0, elbo_rec ≤ 0) ...")
    try:
        model = make_model()
        x = torch.rand(BATCH, 28 * 28)
        (elbo_kl, elbo_rec), _, _ = model(x)

        assert elbo_kl.item() <= 1e-5, f"elbo_kl > 0: {elbo_kl.item():.4f}"
        assert elbo_rec.item() <= 1e-5, f"elbo_rec > 0: {elbo_rec.item():.4f}"

        print("  ✅ ELBO signs: OK")
        return True
    except Exception as e:
        print(f"  ❌ ELBO signs: {e}")
        traceback.print_exc()
        return False


def test_kl_zero():
    """mean=0, log_var=0 のとき elbo_kl=0（KL=0 の解析解確認）"""
    print("Testing elbo_kl = 0 when mean=0, log_var=0 ...")
    try:
        # encoder の出力層バイアスをゼロにして mean=0, log_var=0 を強制
        model = make_model()
        torch.nn.init.zeros_(model.enc_fc3_mean.weight)
        torch.nn.init.zeros_(model.enc_fc3_mean.bias)
        torch.nn.init.zeros_(model.enc_fc3_logvar.weight)
        torch.nn.init.zeros_(model.enc_fc3_logvar.bias)

        x = torch.rand(BATCH, 28 * 28)
        (elbo_kl, _), _, _ = model(x)

        assert (
            abs(elbo_kl.item()) < 1e-3
        ), f"mean=0, log_var=0 → elbo_kl=0 であるべき。実際: {elbo_kl.item():.6f}"

        print("  ✅ elbo_kl = 0 (KL=0 special case): OK")
        return True
    except Exception as e:
        print(f"  ❌ elbo_kl = 0 check: {e}")
        traceback.print_exc()
        return False


def test_gradient_flow():
    """loss.backward() で全パラメータに勾配が流れるかのテスト"""
    print("Testing gradient flow ...")
    try:
        model = make_model().train()
        x = torch.rand(BATCH, 28 * 28)
        (elbo_kl, elbo_rec), _, _ = model(x)
        loss = -(elbo_kl + elbo_rec)
        loss.backward()

        has_grad = all(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "一部のパラメータに勾配が流れていません"

        print("  ✅ gradient flow: OK")
        return True
    except Exception as e:
        print(f"  ❌ gradient flow: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 55)
    print("Ex7 VAE - Implementation Test")
    print("=" * 55)

    tests = [
        test_encoder,
        test_reparametrization_trick,
        test_reparametrization_trick_formula,
        test_decoder,
        test_forward_shapes,
        test_kld_numerical,
        test_elbo_signs,
        test_kl_zero,
        test_gradient_flow,
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
        print("  python main.py --epochs 50")
    else:
        print(f"⚠️  {total - passed} tests failed. ({passed}/{total})")
        print("VAEs/VAE_skeleton.py の #TODO 箇所を確認してください。")


if __name__ == "__main__":
    main()
