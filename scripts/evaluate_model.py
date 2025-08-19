#!/usr/bin/env python3
"""
Model Evaluation Script
Оценка производительности Question-Conditioned LLaVA
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config.base_config import ExperimentConfig
from selector.architecture import QuestionConditionedSelector
from data.coco_pretraining import create_coco_dataloaders
from data.vqa_dataset import VQADataset


def evaluate_efficiency(model, dataloader, device, num_samples=100):
    """Оценка эффективности модели"""
    
    model.eval()
    
    times = []
    memory_usage = []
    sparsity_ratios = []
    
    print(f"🔬 Evaluating efficiency on {num_samples} samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= num_samples:
                break
            
            visual_features = batch['visual_features'].to(device)
            question_embeds = batch['question_embeds'].to(device)
            
            # Measure inference time
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            outputs = model(visual_features, question_embeds)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # ms
            times.append(inference_time)
            
            # Measure memory usage
            if device.type == 'cuda':
                memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                memory_usage.append(memory)
                torch.cuda.reset_peak_memory_stats()
            
            # Calculate sparsity
            sparsity = outputs['selection_mask'].float().mean().item()
            sparsity_ratios.append(sparsity)
    
    results = {
        'avg_inference_time_ms': np.mean(times),
        'std_inference_time_ms': np.std(times),
        'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
        'avg_sparsity_ratio': np.mean(sparsity_ratios),
        'visual_tokens_original': visual_features.shape[1],
        'visual_tokens_selected': int(visual_features.shape[1] * np.mean(sparsity_ratios)),
        'speedup_estimate': 1 / np.mean(sparsity_ratios)
    }
    
    return results


def evaluate_reconstruction_quality(model, dataloader, device, num_samples=100):
    """Оценка качества реконструкции"""
    
    model.eval()
    
    mse_losses = []
    cosine_similarities = []
    
    print(f"🎯 Evaluating reconstruction quality on {num_samples} samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= num_samples:
                break
            
            visual_features = batch['visual_features'].to(device)
            question_embeds = batch['question_embeds'].to(device)
            
            outputs = model(visual_features, question_embeds)
            
            # MSE Loss
            mse = F.mse_loss(visual_features, outputs['reconstructed'])
            mse_losses.append(mse.item())
            
            # Cosine Similarity
            orig_mean = visual_features.mean(dim=1)
            recon_mean = outputs['reconstructed'].mean(dim=1)
            cos_sim = F.cosine_similarity(orig_mean, recon_mean, dim=-1).mean()
            cosine_similarities.append(cos_sim.item())
    
    results = {
        'avg_mse_loss': np.mean(mse_losses),
        'std_mse_loss': np.std(mse_losses),
        'avg_cosine_similarity': np.mean(cosine_similarities),
        'std_cosine_similarity': np.std(cosine_similarities)
    }
    
    return results


def evaluate_attention_patterns(model, dataloader, device, num_samples=50):
    """Анализ attention patterns"""
    
    model.eval()
    
    attention_entropies = []
    max_attentions = []
    attention_distributions = []
    
    print(f"👁️  Evaluating attention patterns on {num_samples} samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= num_samples:
                break
            
            visual_features = batch['visual_features'].to(device)
            question_embeds = batch['question_embeds'].to(device)
            
            outputs = model(visual_features, question_embeds, return_attention=True)
            
            importance_scores = outputs['importance_scores'].squeeze(-1)  # [batch, num_patches]
            
            for j in range(importance_scores.shape[0]):
                scores = importance_scores[j]
                
                # Entropy (measure of attention spread)
                probs = F.softmax(scores, dim=0)
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                attention_entropies.append(entropy.item())
                
                # Max attention
                max_attention = scores.max().item()
                max_attentions.append(max_attention)
                
                # Distribution statistics
                attention_distributions.append(scores.cpu().numpy())
    
    results = {
        'avg_attention_entropy': np.mean(attention_entropies),
        'avg_max_attention': np.mean(max_attentions),
        'attention_concentration': 1 / np.mean(attention_entropies),  # Higher = more focused
        'attention_distributions': attention_distributions
    }
    
    return results


def compare_question_types(model, dataloader, device):
    """Сравнение поведения для разных типов вопросов"""
    
    model.eval()
    
    question_analysis = {
        'color': {'samples': [], 'sparsity': [], 'patterns': []},
        'count': {'samples': [], 'sparsity': [], 'patterns': []},
        'location': {'samples': [], 'sparsity': [], 'patterns': []},
        'general': {'samples': [], 'sparsity': [], 'patterns': []}
    }
    
    print("🔍 Analyzing question type behaviors...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= 50:  # Limit samples
                break
            
            visual_features = batch['visual_features'].to(device)
            question_embeds = batch['question_embeds'].to(device)
            
            # Get questions if available
            questions = batch.get('question', [''] * visual_features.shape[0])
            
            outputs = model(visual_features, question_embeds)
            importance_scores = outputs['importance_scores'].squeeze(-1)
            sparsity_ratios = outputs['selection_mask'].float().mean(dim=1)
            
            for j in range(len(questions)):
                question = questions[j].lower()
                sparsity = sparsity_ratios[j].item()
                pattern = importance_scores[j].cpu().numpy()
                
                # Categorize question
                if 'color' in question or 'what color' in question:
                    category = 'color'
                elif 'how many' in question or 'count' in question:
                    category = 'count'
                elif 'where' in question or 'location' in question:
                    category = 'location'
                else:
                    category = 'general'
                
                question_analysis[category]['samples'].append(question)
                question_analysis[category]['sparsity'].append(sparsity)
                question_analysis[category]['patterns'].append(pattern)
    
    # Calculate statistics for each category
    results = {}
    for category, data in question_analysis.items():
        if data['samples']:
            results[category] = {
                'count': len(data['samples']),
                'avg_sparsity': np.mean(data['sparsity']),
                'std_sparsity': np.std(data['sparsity']),
                'sample_questions': data['samples'][:3]  # First 3 examples
            }
    
    return results


def create_evaluation_report(model_path, results, output_dir):
    """Создает детальный отчет об оценке"""
    
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("📊 Question-Conditioned Selector Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Model: {model_path}\n")
        f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Efficiency Results
        if 'efficiency' in results:
            eff = results['efficiency']
            f.write("🚀 EFFICIENCY METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average Inference Time: {eff['avg_inference_time_ms']:.2f} ± {eff['std_inference_time_ms']:.2f} ms\n")
            f.write(f"Average Memory Usage: {eff['avg_memory_mb']:.1f} MB\n")
            f.write(f"Visual Tokens: {eff['visual_tokens_selected']}/{eff['visual_tokens_original']} ({eff['avg_sparsity_ratio']:.1%})\n")
            f.write(f"Estimated Speedup: {eff['speedup_estimate']:.2f}x\n\n")
        
        # Quality Results
        if 'quality' in results:
            qual = results['quality']
            f.write("🎯 RECONSTRUCTION QUALITY\n")
            f.write("-" * 30 + "\n")
            f.write(f"MSE Loss: {qual['avg_mse_loss']:.6f} ± {qual['std_mse_loss']:.6f}\n")
            f.write(f"Cosine Similarity: {qual['avg_cosine_similarity']:.4f} ± {qual['std_cosine_similarity']:.4f}\n\n")
        
        # Attention Results
        if 'attention' in results:
            att = results['attention']
            f.write("👁️  ATTENTION ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average Attention Entropy: {att['avg_attention_entropy']:.4f}\n")
            f.write(f"Average Max Attention: {att['avg_max_attention']:.4f}\n")
            f.write(f"Attention Concentration: {att['attention_concentration']:.4f}\n\n")
        
        # Question Type Analysis
        if 'question_types' in results:
            qt = results['question_types']
            f.write("🔍 QUESTION TYPE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for category, data in qt.items():
                f.write(f"{category.upper()}:\n")
                f.write(f"  Samples: {data['count']}\n")
                f.write(f"  Avg Sparsity: {data['avg_sparsity']:.3f} ± {data['std_sparsity']:.3f}\n")
                f.write(f"  Examples: {', '.join(data['sample_questions'])}\n\n")
    
    print(f"📄 Evaluation report saved to: {report_path}")


def create_visualization_plots(results, output_dir):
    """Создает визуализации результатов"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Attention distribution plot
    if 'attention' in results and 'attention_distributions' in results['attention']:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        distributions = results['attention']['attention_distributions']
        
        # Sample attention maps
        for i in range(min(4, len(distributions))):
            row, col = i // 2, i % 2
            attention = distributions[i].reshape(24, 24)  # Assuming 576 = 24x24
            
            im = axes[row, col].imshow(attention, cmap='hot', interpolation='nearest')
            axes[row, col].set_title(f'Attention Map {i+1}')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_maps.png'), dpi=150)
        plt.close()
    
    # Question type comparison
    if 'question_types' in results:
        qt = results['question_types']
        categories = list(qt.keys())
        sparsities = [qt[cat]['avg_sparsity'] for cat in categories]
        errors = [qt[cat]['std_sparsity'] for cat in categories]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, sparsities, yerr=errors, capsize=5, alpha=0.7)
        plt.title('Sparsity by Question Type')
        plt.ylabel('Average Sparsity Ratio')
        plt.xlabel('Question Category')
        
        # Add value labels on bars
        for bar, val in zip(bars, sparsities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'question_type_analysis.png'), dpi=150)
        plt.close()
    
    print(f"📊 Visualization plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Question-Conditioned Selector")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="coco", 
                       choices=['coco', 'vqav2'],
                       help="Dataset to evaluate on")
    parser.add_argument("--split", type=str, default="val",
                       choices=['train', 'val'],
                       help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    print("🔬 Question-Conditioned Selector Evaluation")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Samples: {args.num_samples}")
    
    # Setup device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Load model
    config = ExperimentConfig()
    model = QuestionConditionedSelector(
        visual_dim=config.model.visual_dim,
        text_dim=config.model.text_dim,
        sparsity_factor=config.model.sparsity_factor,
        num_heads=config.model.num_heads,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout
    ).to(device)
    
    # Load checkpoint
    print(f"📥 Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Model weights loaded")
    
    # Create dataset
    print(f"📊 Creating {args.dataset} dataset...")
    
    if args.dataset == "coco":
        _, dataloader = create_coco_dataloaders(config, distributed=False, num_workers=2)
        if args.split == "train":
            dataloader, _ = create_coco_dataloaders(config, distributed=False, num_workers=2)
    else:
        try:
            dataset = VQADataset(config, split=args.split, max_samples=args.num_samples)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        except:
            print("⚠️  VQA dataset not available, using COCO as fallback")
            _, dataloader = create_coco_dataloaders(config, distributed=False, num_workers=2)
    
    print(f"✅ Dataset created")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluations
    results = {}
    
    print("\n🔄 Running evaluations...")
    
    # 1. Efficiency evaluation
    results['efficiency'] = evaluate_efficiency(model, dataloader, device, args.num_samples)
    
    # 2. Quality evaluation
    results['quality'] = evaluate_reconstruction_quality(model, dataloader, device, args.num_samples)
    
    # 3. Attention analysis
    results['attention'] = evaluate_attention_patterns(model, dataloader, device, min(100, args.num_samples))
    
    # 4. Question type analysis
    results['question_types'] = compare_question_types(model, dataloader, device)
    
    # Print summary
    print("\n📋 EVALUATION SUMMARY")
    print("=" * 30)
    
    if 'efficiency' in results:
        eff = results['efficiency']
        print(f"⚡ Inference Time: {eff['avg_inference_time_ms']:.1f}ms")
        print(f"💾 Memory Usage: {eff['avg_memory_mb']:.1f}MB")
        print(f"🎯 Sparsity: {eff['avg_sparsity_ratio']:.1%}")
        print(f"🚀 Speedup: {eff['speedup_estimate']:.2f}x")
    
    if 'quality' in results:
        qual = results['quality']
        print(f"📊 Reconstruction MSE: {qual['avg_mse_loss']:.4f}")
        print(f"📈 Cosine Similarity: {qual['avg_cosine_similarity']:.4f}")
    
    # Save detailed results
    create_evaluation_report(args.model_path, results, args.output_dir)
    create_visualization_plots(results, args.output_dir)
    
    # Save raw results
    results_file = os.path.join(args.output_dir, "evaluation_results.pt")
    torch.save(results, results_file)
    print(f"💾 Raw results saved to: {results_file}")
    
    print(f"\n✅ Evaluation completed! Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
