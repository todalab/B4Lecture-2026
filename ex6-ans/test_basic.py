#!/usr/bin/env python3
"""
Basic test for ex6 answer implementation
Tests transformer functionality without downloading external data
"""

import torch
import numpy as np
from transformer import LanguageModel, get_model_config, count_parameters
from data_utils import CharDataset

def test_transformer():
    """Test basic transformer functionality"""
    print("Testing Transformer implementation...")
    
    # Create small test vocabulary
    vocab_size = 100
    seq_len = 32
    batch_size = 2
    
    # Test different model sizes
    for size in ['tiny', 'small']:
        print(f"\nTesting {size} model...")
        
        config = get_model_config(size)
        model = LanguageModel(
            vocab_size=vocab_size,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_seq_len=seq_len * 2
        )
        
        param_count = count_parameters(model)
        print(f"  Parameters: {param_count:,}")
        
        # Test forward pass
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        logits, loss = model(x, y)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
        
        # Test generation
        context = torch.randint(0, vocab_size, (1, 5))
        generated = model.generate(context, max_tokens=10, temperature=1.0)
        print(f"  Generation input: {context.shape}")
        print(f"  Generation output: {generated.shape}")

def test_dataset():
    """Test dataset functionality with dummy data"""
    print("\nTesting Dataset functionality...")
    
    # Create dummy text data (list of integers)
    dummy_data = list(range(1000))  # Simple sequence 0, 1, 2, ..., 999
    seq_len = 32
    
    dataset = CharDataset(dummy_data, seq_len)
    print(f"  Dataset length: {len(dataset)}")
    
    # Test data loading
    x, y = dataset[0]
    print(f"  First sample - x: {x[:10]}")
    print(f"  First sample - y: {y[:10]}")
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    
    # Check that y is x shifted by 1
    assert torch.equal(x[1:], y[:-1])
    print("  Dataset test passed!")

def test_training_step():
    """Test one training step"""
    print("\nTesting training step...")
    
    # Small model for testing
    config = get_model_config('tiny')
    vocab_size = 100
    model = LanguageModel(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=64
    )
    
    # Dummy data
    x = torch.randint(0, vocab_size, (4, 32))
    y = torch.randint(0, vocab_size, (4, 32))
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    logits, loss = model(x, y)
    loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"  Forward pass successful: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed: {has_grad}")
    
    optimizer.step()
    print("  Training step completed!")

def main():
    print("=== Ex6 Answer Implementation Test ===")
    
    try:
        test_transformer()
        test_dataset()
        test_training_step()
        print("\n✅ All tests passed! Implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()