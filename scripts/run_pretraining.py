#!/usr/bin/env python3
"""
COCO Pretraining Script - Test Version
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from config.base_config import ExperimentConfig, create_config_variants
from selector.architecture import QuestionConditionedSelector
from selector.losses import SelectorLossFunction


def test_model_creation():
    """Test model creation and basic functionality"""
    
    print("🚀 Question-Conditioned Selector - Test Run")
    print("=" * 50)
    
    # Load configuration
    config = ExperimentConfig()
    print(f"Experiment: {config.experiment_name}")
    print(f"Sparsity factor: {config.model.sparsity_factor}")
    print(f"Device: {config.training.device}")
    
    # Create model
    model = QuestionConditionedSelector(
        visual_dim=config.model.visual_dim,
        text_dim=config.model.text_dim,
        sparsity_factor=config.model.sparsity_factor,
        num_heads=config.model.num_heads,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params:,} parameters")
    
    # Test forward pass
    import torch
    device = torch.device(config.training.device)
    model = model.to(device)
    
    print(f"Testing on device: {device}")
    
    # Mock inputs (typical LLaVA dimensions)
    batch_size = 2
    num_patches = 576  # 24x24 patches for 336x336 image
    seq_len = 20      # question length
    
    visual_features = torch.randn(batch_size, num_patches, config.model.visual_dim).to(device)
    question_embeds = torch.randn(batch_size, seq_len, config.model.text_dim).to(device)
    
    print(f"Input shapes:")
    print(f"  Visual features: {visual_features.shape}")
    print(f"  Question embeds: {question_embeds.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(visual_features, question_embeds)
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test sparsity
    sparsity_stats = model.get_sparsity_stats(outputs['selection_mask'])
    print(f"\nSparsity statistics:")
    for key, value in sparsity_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test loss computation
    loss_fn = SelectorLossFunction(
        mse_weight=config.model.mse_weight,
        cosine_weight=config.model.cosine_weight,
        sparsity_weight=config.model.sparsity_weight,
        target_sparsity=config.model.sparsity_factor
    )
    
    losses = loss_fn(
        original_features=visual_features,
        reconstructed=outputs['reconstructed'],
        selection_mask=outputs['selection_mask']
    )
    
    print(f"\nLoss computation:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")
    
    print(f"\n✅ All tests passed!")
    print(f"🎯 Model is ready for training")
    
    # Test different config variants
    print(f"\n📋 Available config variants:")
    variants = create_config_variants()
    for name, variant_config in variants.items():
        print(f"  {name}: sparsity={variant_config.model.sparsity_factor}, "
              f"epochs={variant_config.training.epochs}")
    
    print(f"\n🚀 Next steps:")
    print(f"1. Add data loading (COCO dataset)")
    print(f"2. Implement full training loop")
    print(f"3. Add VQA fine-tuning")
    print(f"4. Run experiments!")
    
    return model, config


def main():
    """Main function"""
    try:
        test_model_creation()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
