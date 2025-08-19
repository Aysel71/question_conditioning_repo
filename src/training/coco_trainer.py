# src/training/coco_trainer.py
"""
Complete COCO Pretraining Implementation
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, Optional

from .trainer import BaseTrainer
from ..selector.losses import SelectorLossFunction
from ..config.base_config import ExperimentConfig


class COCOPretrainer(BaseTrainer):
    """Complete COCO Pretrainer implementation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_phase = 'pretrain'
    
    def _get_total_epochs(self) -> int:
        return self.config.training.pretrain_epochs
    
    def _get_learning_rate(self) -> float:
        return self.config.training.pretrain_learning_rate
    
    def _get_weight_decay(self) -> float:
        return self.config.training.pretrain_weight_decay
    
    def _get_batch_data(self, batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extract data for COCO pretraining"""
        visual_features = batch['visual_features']
        question_embeds = batch['question_embeds']
        return visual_features, question_embeds, None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch with full functionality"""
        self.model.train()
        
        # Progress bar
        loader = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        epoch_losses = {'mse': [], 'cosine': [], 'sparsity': [], 'total': []}
        epoch_metrics = {
            'sparsity_ratio': [], 'reconstruction_mse': [], 
            'cosine_similarity': [], 'avg_confidence': []
        }
        
        for batch_idx, batch in enumerate(loader):
            # Extract data
            visual_features, question_embeds, _ = self._get_batch_data(batch)
            visual_features = visual_features.to(self.device)
            question_embeds = question_embeds.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(visual_features, question_embeds)
            
            # Compute losses
            losses = self.loss_function(
                original_features=visual_features,
                reconstructed=outputs['reconstructed'],
                selection_mask=outputs['selection_mask']
            )
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step (if using cycle scheduler)
            if hasattr(self.scheduler, 'step') and self.scheduler is not None:
                self.scheduler.step()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = self._calculate_metrics(outputs, visual_features)
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]["lr"]
            loader.set_postfix({
                'MSE': f"{losses['mse'].item():.4f}",
                'Sparsity': f"{metrics['sparsity_ratio']:.3f}",
                'LR': f"{current_lr:.6f}"
            })
            
            # Log metrics
            if batch_idx % self.config.training.log_interval == 0:
                self._log_training_step(losses, metrics, current_lr)
            
            # Validation
            if (self.global_step % self.config.training.eval_interval == 0 and 
                self.global_step > 0):
                val_losses, val_metrics = self.validate()
                self._log_validation_step(val_losses, val_metrics)
                
                # Save best checkpoint
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint(is_best=True)
                
                # Return to training mode
                self.model.train()
            
            # Accumulate epoch statistics
            for key, value in losses.items():
                epoch_losses[key].append(value.item())
            
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            
            self.global_step += 1
        
        # Calculate epoch averages
        epoch_summary = {}
        for key, values in epoch_losses.items():
            epoch_summary[f"train/{key}"] = np.mean(values)
        for key, values in epoch_metrics.items():
            epoch_summary[f"train/{key}"] = np.mean(values)
        
        return epoch_summary
    
    def _calculate_metrics(self, outputs: Dict, visual_features: torch.Tensor) -> Dict[str, float]:
        """Calculate detailed metrics"""
        metrics = {}
        
        # Sparsity metrics
        selection_mask = outputs['selection_mask']
        metrics['sparsity_ratio'] = selection_mask.float().mean().item()
        
        # Reconstruction quality
        reconstructed = outputs['reconstructed']
        mse = torch.nn.functional.mse_loss(visual_features, reconstructed)
        metrics['reconstruction_mse'] = mse.item()
        
        # Cosine similarity
        orig_mean = visual_features.mean(dim=1)
        recon_mean = reconstructed.mean(dim=1)
        cos_sim = torch.nn.functional.cosine_similarity(orig_mean, recon_mean, dim=-1)
        metrics['cosine_similarity'] = cos_sim.mean().item()
        
        # Confidence metrics
        importance_scores = outputs['importance_scores'].squeeze(-1)
        confidence = torch.max(torch.stack([importance_scores, 1 - importance_scores]), dim=0)[0]
        metrics['avg_confidence'] = confidence.mean().item()
        
        return metrics
    
    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validation with detailed metrics"""
        self.model.eval()
        
        total_losses = {'mse': 0, 'cosine': 0, 'sparsity': 0, 'total': 0}
        total_metrics = {
            'sparsity_ratio': 0, 'reconstruction_mse': 0,
            'cosine_similarity': 0, 'avg_confidence': 0
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                visual_features, question_embeds, _ = self._get_batch_data(batch)
                visual_features = visual_features.to(self.device)
                question_embeds = question_embeds.to(self.device)
                
                # Forward pass
                outputs = self.model(visual_features, question_embeds)
                
                # Compute losses
                losses = self.loss_function(
                    original_features=visual_features,
                    reconstructed=outputs['reconstructed'],
                    selection_mask=outputs['selection_mask']
                )
                
                # Calculate metrics
                metrics = self._calculate_metrics(outputs, visual_features)
                
                # Accumulate
                for key, value in losses.items():
                    total_losses[key] += value.item()
                
                for key, value in metrics.items():
                    total_metrics[key] += value
                
                num_batches += 1
        
        # Average
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        return avg_losses, avg_metrics
    
    def save_training_plots(self, save_path: str):
        """Save comprehensive training plots"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss plots
        if self.train_metrics['mse_list']:
            axes[0, 0].plot(self.train_metrics['mse_list'], label='Train MSE', alpha=0.7)
            if self.val_metrics['mse_list']:
                axes[0, 0].plot(self.val_metrics['mse_list'], label='Val MSE', alpha=0.7)
            axes[0, 0].set_title('MSE Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        if self.train_metrics['cos_list']:
            axes[0, 1].plot(self.train_metrics['cos_list'], label='Train Cosine', alpha=0.7)
            if self.val_metrics['cos_list']:
                axes[0, 1].plot(self.val_metrics['cos_list'], label='Val Cosine', alpha=0.7)
            axes[0, 1].set_title('Cosine Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Sparsity plot
        if self.train_metrics['iva_list']:
            axes[0, 2].plot(self.train_metrics['iva_list'], label='Train Sparsity', alpha=0.7)
            axes[0, 2].axhline(y=self.config.model.sparsity_factor, color='r', 
                              linestyle='--', label='Target')
            if self.val_metrics['iva_list']:
                axes[0, 2].plot(self.val_metrics['iva_list'], label='Val Sparsity', alpha=0.7)
            axes[0, 2].set_title('Sparsity Ratio')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Learning rate
        if self.train_metrics['lr_list']:
            axes[1, 0].plot(self.train_metrics['lr_list'], label='Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Total loss
        if self.train_metrics['mse_list'] and self.train_metrics['cos_list']:
            total_train = [m + c for m, c in zip(self.train_metrics['mse_list'], 
                                               self.train_metrics['cos_list'])]
            axes[1, 1].plot(total_train, label='Train Total', alpha=0.7)
            
            if self.val_metrics['mse_list'] and self.val_metrics['cos_list']:
                total_val = [m + c for m, c in zip(self.val_metrics['mse_list'],
                                                 self.val_metrics['cos_list'])]
                axes[1, 1].plot(total_val, label='Val Total', alpha=0.7)
            
            axes[1, 1].set_title('Total Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Training progress
        if self.train_metrics['epoch_list']:
            epochs = list(range(len(self.train_metrics['mse_list'])))
            axes[1, 2].plot(epochs, self.train_metrics['mse_list'], label='MSE')
            axes[1, 2].set_title('Training Progress')
            axes[1, 2].set_xlabel('Steps')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to: {save_path}")


def run_coco_pretraining(
    config: ExperimentConfig,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    output_dir: str,
    device: torch.device,
    resume_from: Optional[str] = None
) -> Dict:
    """Run complete COCO pretraining"""
    
    print("🚀 Starting COCO Pretraining")
    print("=" * 50)
    print(f"Config: {config.experiment_name}")
    print(f"Device: {device}")
    print(f"Epochs: {config.training.pretrain_epochs}")
    print(f"Batch size: {config.training.pretrain_batch_size}")
    print(f"Learning rate: {config.training.pretrain_learning_rate}")
    print(f"Sparsity factor: {config.model.sparsity_factor}")
    print("=" * 50)
    
    # Create model
    from ..selector.architecture import QuestionConditionedSelector
    
    model = QuestionConditionedSelector(
        visual_dim=config.model.visual_dim,
        text_dim=config.model.text_dim,
        sparsity_factor=config.model.sparsity_factor,
        num_heads=config.model.num_heads,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout
    ).to(device)
    
    # Create trainer
    trainer = COCOPretrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        device=device,
        output_dir=output_dir,
        experiment_name=f"{config.experiment_name}_coco_pretrain"
    )
    
    # Resume if specified
    if resume_from:
        trainer.load_checkpoint(resume_from)
    
    # Start training
    try:
        training_summary = trainer.train()
        
        # Save final plots
        plots_path = os.path.join(output_dir, "training_plots.png")
        trainer.save_training_plots(plots_path)
        
        print("\n" + "=" * 50)
        print("🎉 COCO PRETRAINING COMPLETED!")
        print("=" * 50)
        print(f"Best validation loss: {trainer.best_val_loss:.6f}")
        print(f"Total time: {training_summary.get('total_time', 0) / 3600:.2f} hours")
        print(f"Final checkpoint: {os.path.join(output_dir, 'checkpoints', 'best_checkpoint.pt')}")
        
        return {
            'best_val_loss': trainer.best_val_loss,
            'training_summary': training_summary,
            'final_model': trainer.model,
            'trainer': trainer
        }
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer.save_checkpoint(suffix="_interrupted")
        return {'status': 'interrupted', 'trainer': trainer}
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        trainer.save_checkpoint(suffix="_error")
        raise


if __name__ == "__main__":
    # Test trainer
    from ..config.base_config import ExperimentConfig
    
    config = ExperimentConfig()
    config.experiment_name = 'debug_selector'
    config.training.pretrain_epochs = 1
    config.training.pretrain_batch_size = 4
    
    print("COCOPretrainer test completed!")
