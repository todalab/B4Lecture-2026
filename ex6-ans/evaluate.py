"""
Evaluation script for scaling experiments and visualization
"""

import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
from transformer import LanguageModel, get_model_config, count_parameters
from data_utils import create_data_loaders, decode_text
import seaborn as sns


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    vocab_size = checkpoint['vocab_size']
    
    model = LanguageModel(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=512
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint


def evaluate_model(model, val_loader, device):
    """Detailed evaluation of a single model"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            
            batch_size, seq_len = y.shape
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity


def run_scaling_experiments(models_dir, dataset_name, device):
    """Run scaling experiments on all trained models"""
    print(f"Running scaling experiments for {dataset_name} models...")
    
    # Load validation data
    _, val_loader, vocab_size, mapping1, mapping2 = create_data_loaders(
        dataset_name, seq_len=128, batch_size=32
    )
    
    # Find all model files for this dataset
    pattern = os.path.join(models_dir, f"*_{dataset_name}_best.pt")
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f"No trained models found for {dataset_name} in {models_dir}")
        return None
    
    results = []
    
    for model_path in sorted(model_files):
        print(f"Evaluating {model_path}")
        
        try:
            model, checkpoint = load_model(model_path, device)
            model_size = checkpoint['args'].model_size if hasattr(checkpoint['args'], 'model_size') else 'unknown'
            
            # Count parameters
            param_count = count_parameters(model)
            
            # Evaluate
            val_loss, perplexity = evaluate_model(model, val_loader, device)
            
            result = {
                'model_size': model_size,
                'param_count': param_count,
                'param_count_M': param_count / 1e6,
                'val_loss': val_loss,
                'perplexity': perplexity,
                'model_path': model_path
            }
            
            results.append(result)
            print(f"  {model_size}: {param_count:,} params, Perplexity: {perplexity:.2f}")
            
        except Exception as e:
            print(f"  Error evaluating {model_path}: {e}")
    
    return results


def plot_scaling_results(results, dataset_name, save_dir):
    """Plot scaling law results"""
    if not results:
        print("No results to plot")
        return
    
    # Sort by parameter count
    results = sorted(results, key=lambda x: x['param_count'])
    
    param_counts = [r['param_count_M'] for r in results]
    perplexities = [r['perplexity'] for r in results]
    val_losses = [r['val_loss'] for r in results]
    model_sizes = [r['model_size'] for r in results]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Perplexity vs Parameters (linear scale)
    ax1.plot(param_counts, perplexities, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Parameters (Millions)')
    ax1.set_ylabel('Perplexity')
    ax1.set_title(f'{dataset_name.title()} - Perplexity vs Model Size')
    ax1.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, size in enumerate(model_sizes):
        ax1.annotate(size, (param_counts[i], perplexities[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # 2. Log-log plot for scaling law
    ax2.loglog(param_counts, perplexities, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Parameters (Millions)')
    ax2.set_ylabel('Perplexity')
    ax2.set_title(f'{dataset_name.title()} - Scaling Law (Log-Log)')
    ax2.grid(True, alpha=0.3)
    
    # Fit power law if we have enough points
    if len(param_counts) >= 3:
        log_params = np.log(param_counts)
        log_perplexity = np.log(perplexities)
        
        # Fit line: log(perplexity) = a + b * log(params)
        coeffs = np.polyfit(log_params, log_perplexity, 1)
        b, a = coeffs
        
        # Plot fit line
        x_fit = np.logspace(np.log10(min(param_counts)), np.log10(max(param_counts)), 100)
        y_fit = np.exp(a) * (x_fit ** b)
        ax2.plot(x_fit, y_fit, '--', alpha=0.7, label=f'Power law: y = {np.exp(a):.2f} * x^{b:.2f}')
        ax2.legend()
    
    # 3. Validation Loss vs Parameters
    ax3.plot(param_counts, val_losses, 'o-', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('Parameters (Millions)')
    ax3.set_ylabel('Validation Loss')
    ax3.set_title(f'{dataset_name.title()} - Loss vs Model Size')
    ax3.grid(True, alpha=0.3)
    
    for i, size in enumerate(model_sizes):
        ax3.annotate(size, (param_counts[i], val_losses[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # 4. Bar chart of model sizes
    ax4.bar(model_sizes, param_counts, color='skyblue', alpha=0.7)
    ax4.set_xlabel('Model Size')
    ax4.set_ylabel('Parameters (Millions)')
    ax4.set_title(f'{dataset_name.title()} - Parameter Count by Model Size')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, (size, count) in enumerate(zip(model_sizes, param_counts)):
        ax4.text(i, count + max(param_counts) * 0.01, f'{count:.1f}M', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f'{dataset_name}_scaling_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Scaling plot saved to {plot_path}")
    
    plt.show()
    
    return plot_path


def compare_datasets(shakespeare_results, wikitext_results, save_dir):
    """Compare scaling results between datasets"""
    if not shakespeare_results or not wikitext_results:
        print("Need results from both datasets for comparison")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data for both datasets
    shakespeare_params = [r['param_count_M'] for r in shakespeare_results]
    shakespeare_perp = [r['perplexity'] for r in shakespeare_results]
    
    wikitext_params = [r['param_count_M'] for r in wikitext_results]
    wikitext_perp = [r['perplexity'] for r in wikitext_results]
    
    # Plot comparison
    ax1.plot(shakespeare_params, shakespeare_perp, 'o-', label='Shakespeare', linewidth=2, markersize=8)
    ax1.plot(wikitext_params, wikitext_perp, 's-', label='WikiText-2', linewidth=2, markersize=8)
    ax1.set_xlabel('Parameters (Millions)')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Dataset Comparison - Perplexity vs Model Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log-log comparison
    ax2.loglog(shakespeare_params, shakespeare_perp, 'o-', label='Shakespeare', linewidth=2, markersize=8)
    ax2.loglog(wikitext_params, wikitext_perp, 's-', label='WikiText-2', linewidth=2, markersize=8)
    ax2.set_xlabel('Parameters (Millions)')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Dataset Comparison - Scaling Law (Log-Log)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'dataset_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_path}")
    
    plt.show()


def generate_text_samples(models_dir, dataset_name, device, num_samples=3):
    """Generate text samples from different model sizes"""
    print(f"\nGenerating text samples for {dataset_name} models...")
    
    # Load mappings
    _, _, vocab_size, mapping1, mapping2 = create_data_loaders(dataset_name, seq_len=128, batch_size=1)
    mappings = (mapping1, mapping2)
    
    # Find model files
    pattern = os.path.join(models_dir, f"*_{dataset_name}_best.pt")
    model_files = glob.glob(pattern)
    
    prompts = {
        'shakespeare': ["To be or not to be", "Romeo and Juliet", "Once upon a time"],
        'wikitext2': ["The quick brown", "Machine learning is", "In recent years"]
    }
    
    for model_path in sorted(model_files):
        try:
            model, checkpoint = load_model(model_path, device)
            model_size = checkpoint['args'].model_size if hasattr(checkpoint['args'], 'model_size') else 'unknown'
            
            print(f"\n=== {model_size.upper()} MODEL ===")
            
            for i, prompt in enumerate(prompts[dataset_name][:num_samples]):
                # Prepare context
                if dataset_name == "shakespeare":
                    char_to_idx, idx_to_char = mappings
                    context = [char_to_idx.get(c, 0) for c in prompt]
                else:
                    token_to_idx, idx_to_token = mappings
                    context = [token_to_idx.get(token, 0) for token in prompt.split()]
                
                context = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
                
                # Generate
                model.eval()
                with torch.no_grad():
                    generated = model.generate(context, max_tokens=100, temperature=0.8, top_k=50)
                
                # Decode
                output_text = decode_text(generated[0].tolist(), mappings[1], dataset_name)
                
                print(f"Prompt: {prompt}")
                print(f"Generated: {output_text[:200]}...")
                print()
                
        except Exception as e:
            print(f"Error with model {model_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer scaling experiments")
    parser.add_argument("--models_dir", default="./models", help="Directory containing trained models")
    parser.add_argument("--results_dir", default="./results", help="Directory to save results")
    parser.add_argument("--figures_dir", default="./figures", help="Directory to save figures")
    parser.add_argument("--dataset", choices=["shakespeare", "wikitext2", "both"], default="both")
    parser.add_argument("--run_scaling_experiments", action="store_true", help="Run scaling experiments")
    parser.add_argument("--plot_results", action="store_true", help="Plot scaling results")
    parser.add_argument("--generate_samples", action="store_true", help="Generate text samples")
    parser.add_argument("--device", default="auto")
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    
    shakespeare_results = None
    wikitext_results = None
    
    # Run scaling experiments
    if args.run_scaling_experiments:
        if args.dataset in ["shakespeare", "both"]:
            shakespeare_results = run_scaling_experiments(args.models_dir, "shakespeare", device)
            if shakespeare_results:
                results_path = os.path.join(args.results_dir, "shakespeare_scaling_results.json")
                with open(results_path, 'w') as f:
                    json.dump(shakespeare_results, f, indent=2)
                print(f"Shakespeare results saved to {results_path}")
        
        if args.dataset in ["wikitext2", "both"]:
            wikitext_results = run_scaling_experiments(args.models_dir, "wikitext2", device)
            if wikitext_results:
                results_path = os.path.join(args.results_dir, "wikitext2_scaling_results.json")
                with open(results_path, 'w') as f:
                    json.dump(wikitext_results, f, indent=2)
                print(f"WikiText-2 results saved to {results_path}")
    
    # Load existing results if not running experiments
    if not args.run_scaling_experiments:
        try:
            if args.dataset in ["shakespeare", "both"]:
                with open(os.path.join(args.results_dir, "shakespeare_scaling_results.json"), 'r') as f:
                    shakespeare_results = json.load(f)
            if args.dataset in ["wikitext2", "both"]:
                with open(os.path.join(args.results_dir, "wikitext2_scaling_results.json"), 'r') as f:
                    wikitext_results = json.load(f)
        except FileNotFoundError as e:
            print(f"Could not load existing results: {e}")
    
    # Plot results
    if args.plot_results:
        if shakespeare_results:
            plot_scaling_results(shakespeare_results, "shakespeare", args.figures_dir)
        if wikitext_results:
            plot_scaling_results(wikitext_results, "wikitext2", args.figures_dir)
        if shakespeare_results and wikitext_results:
            compare_datasets(shakespeare_results, wikitext_results, args.figures_dir)
    
    # Generate text samples
    if args.generate_samples:
        if args.dataset in ["shakespeare", "both"]:
            generate_text_samples(args.models_dir, "shakespeare", device)
        if args.dataset in ["wikitext2", "both"]:
            generate_text_samples(args.models_dir, "wikitext2", device)


if __name__ == "__main__":
    main()