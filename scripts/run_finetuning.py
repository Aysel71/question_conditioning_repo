#!/usr/bin/env python3
"""
VQA Fine-tuning Script
Stage 2: Fine-tune selector на VQA dataset с ground truth attention
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

import torch
from torch.utils.data import DataLoader

from config.base_config import ExperimentConfig, create_config_variants
from selector.architecture import QuestionConditionedSelector
from training.coco_trainer import COCOPretrainer
from llava_integration.attention_extraction import VQAGroundTruthGenerator
from data.vqa_dataset import VQADataset


def main():
    parser = argparse.ArgumentParser(description="VQA Fine-tuning")
    parser.add_argument("--pretrained_selector", type=str, required=True,
                       help="Path to pretrained selector checkpoint")
    parser.add_argument("--config_variant", type=str, default="base",
                       choices=['base', 'high_sparsity', 'low_sparsity', 'debug'])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--generate_ground_truth", action="store_true",
                       help="Generate ground truth attention data")
    parser.add_argument("--max_samples", type=int, help="Limit dataset size")
    
    args = parser.parse_args()
    
    print("🚀 VQA Fine-tuning")
    print("=" * 50)
    
    # Load configuration
    configs = create_config_variants()
    config = configs[args.config_variant]
    
    # Override config with args
    if args.epochs:
        config.training.finetune_epochs = args.epochs
    if args.batch_size:
        config.training.finetune_batch_size = args.batch_size
    if args.learning_rate:
        config.training.finetune_learning_rate = args.learning_rate
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Generate ground truth if needed
    if args.generate_ground_truth:
        print("🔄 Generating VQA ground truth data...")
        generator = VQAGroundTruthGenerator(device=device)
        
        gt_output_file = os.path.join(args.output_dir, "vqa_ground_truth.pt")
        generator.process_vqa_dataset(
            vqa_questions_file=config.data.vqa_questions_val,
            vqa_annotations_file=config.data.vqa_annotations_val,
            coco_images_dir=config.data.coco_val_images_path,
            output_file=gt_output_file,
            max_samples=args.max_samples
        )
        print(f"✅ Ground truth saved to: {gt_output_file}")
    
    # Create VQA dataset and dataloaders
    try:
        print("📊 Creating VQA dataloaders...")
        train_dataset = VQADataset(config, split='train', max_samples=args.max_samples)
        val_dataset = VQADataset(config, split='val', max_samples=100)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.training.finetune_batch_size,
            shuffle=True, 
            num_workers=2
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.training.finetune_batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        print("🔧 Using mock datasets for testing...")
        
        # Mock datasets
        from data.coco_pretraining import create_coco_dataloaders
        train_dataloader, val_dataloader = create_coco_dataloaders(config)
    
    # Load pretrained selector
    print(f"📥 Loading pretrained selector from: {args.pretrained_selector}")
    
    model = QuestionConditionedSelector(
        visual_dim=config.model.visual_dim,
        text_dim=config.model.text_dim,
        sparsity_factor=config.model.sparsity_factor,
        num_heads=config.model.num_heads,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout
    ).to(device)
    
    # Load pretrained weights
    if os.path.exists(args.pretrained_selector):
        checkpoint = torch.load(args.pretrained_selector, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Pretrained weights loaded")
    else:
        print("⚠️  Pretrained checkpoint not found, using random initialization")
    
    # Setup output directory
    output_dir = os.path.join(args.output_dir, f"{config.experiment_name}_vqa_finetune")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create trainer (reuse COCO trainer but with different phase)
    trainer = COCOPretrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        device=device,
        output_dir=output_dir,
        experiment_name=f"{config.experiment_name}_vqa_finetune"
    )
    
    # Change to finetune phase
    trainer.training_phase = 'finetune'
    
    # Adjust training parameters for fine-tuning
    trainer.config.training.pretrain_epochs = config.training.finetune_epochs
    trainer.config.training.pretrain_learning_rate = config.training.finetune_learning_rate
    trainer.config.training.pretrain_batch_size = config.training.finetune_batch_size
    
    # Update loss weights for fine-tuning
    trainer.loss_function.attention_weight = 0.8  # Higher weight for attention alignment
    trainer.loss_function.mse_weight = 0.5        # Lower weight for reconstruction
    
    print("🚀 Starting VQA fine-tuning...")
    print(f"Epochs: {config.training.finetune_epochs}")
    print(f"Learning rate: {config.training.finetune_learning_rate}")
    print(f"Batch size: {config.training.finetune_batch_size}")
    
    # Start training
    try:
        training_summary = trainer.train()
        
        print("\n🎉 VQA FINE-TUNING COMPLETED!")
        print(f"Best validation loss: {trainer.best_val_loss:.6f}")
        print(f"Final model: {os.path.join(output_dir, 'checkpoints', 'best_checkpoint.pt')}")
        
        # Create symlink for easy access
        final_model_path = os.path.join(args.output_dir, "final_model.pt")
        best_checkpoint = os.path.join(output_dir, "checkpoints", "best_checkpoint.pt")
        
        if os.path.exists(best_checkpoint):
            if os.path.exists(final_model_path):
                os.remove(final_model_path)
            os.symlink(os.path.abspath(best_checkpoint), final_model_path)
            print(f"🔗 Final model linked to: {final_model_path}")
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
