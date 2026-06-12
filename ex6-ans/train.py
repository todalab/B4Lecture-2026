"""
Training script for Transformer models
Supports different model sizes and datasets
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
import os
import json
from transformer import LanguageModel, get_model_config, count_parameters
from data_utils import create_data_loaders, decode_text


def train_epoch(model, train_loader, optimizer, device, grad_accumulation_steps=1):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        logits, loss = model(x, y)
        loss = loss / grad_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accumulation_steps
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item() * grad_accumulation_steps:.4f}")
    
    return total_loss / num_batches


def evaluate(model, val_loader, device):
    """Evaluate the model on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return avg_loss, perplexity.item()


def get_lr_schedule(optimizer, warmup_steps, total_steps):
    """Learning rate scheduler with warmup and cosine decay"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def generate_sample(model, dataset_name, mapping, device, prompt="", max_tokens=100, temperature=0.8):
    """Generate text sample from the model"""
    model.eval()
    
    if dataset_name == "shakespeare":
        char_to_idx, idx_to_char = mapping
        if not prompt:
            prompt = "To be or not to be"
        context = [char_to_idx.get(c, 0) for c in prompt]
    else:  # wikitext2
        token_to_idx, idx_to_token = mapping
        if not prompt:
            prompt = "The quick brown"
        context = [token_to_idx.get(token, 0) for token in prompt.split()]
    
    context = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = model.generate(context, max_tokens, temperature=temperature, top_k=50)
    
    output_text = decode_text(generated[0].tolist(), 
                             mapping[1] if dataset_name == "shakespeare" else mapping[1],
                             dataset_name)
    
    return output_text


def main():
    parser = argparse.ArgumentParser(description="Train Transformer language model")
    parser.add_argument("--model_size", choices=["tiny", "small", "medium", "large"], default="tiny")
    parser.add_argument("--dataset", choices=["shakespeare", "wikitext2"], default="shakespeare")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_dir", default="./models")
    parser.add_argument("--device", default="auto")
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    train_loader, val_loader, vocab_size, mapping1, mapping2 = create_data_loaders(
        args.dataset, args.seq_len, args.batch_size
    )
    mappings = (mapping1, mapping2)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Model setup
    config = get_model_config(args.model_size)
    model = LanguageModel(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=args.seq_len * 2  # Allow longer generation
    ).to(device)
    
    param_count = count_parameters(model)
    print(f"Model: {args.model_size} ({param_count:,} parameters)")
    
    # Learning rate setup
    if args.lr is None:
        # Default learning rates based on model size
        lr_map = {"tiny": 3e-4, "small": 1e-4, "medium": 5e-5, "large": 1e-5}
        args.lr = lr_map[args.model_size]
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accumulation_steps
    scheduler = get_lr_schedule(optimizer, args.warmup_steps, total_steps)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.grad_accumulation_steps)
        scheduler.step()
        
        # Evaluate
        val_loss, perplexity = evaluate(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, f"{args.model_size}_{args.dataset}_best.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'vocab_size': vocab_size,
                'val_loss': val_loss,
                'perplexity': perplexity,
                'epoch': epoch,
                'args': args
            }, model_path)
            print(f"Saved best model to {model_path}")
        
        # Generate sample text
        if (epoch + 1) % 5 == 0:
            print("\nSample generation:")
            sample = generate_sample(model, args.dataset, mappings, device)
            print(sample[:200] + "..." if len(sample) > 200 else sample)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save final results
    results = {
        'model_size': args.model_size,
        'dataset': args.dataset,
        'param_count': param_count,
        'best_val_loss': best_val_loss,
        'final_perplexity': perplexity,
        'training_time': training_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
        'args': vars(args)
    }
    
    results_path = os.path.join(args.save_dir, f"{args.model_size}_{args.dataset}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()