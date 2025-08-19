"""
Base Trainer class - simplified version
"""

import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

class BaseTrainer(ABC):
    """Simplified base trainer"""
    
    def __init__(self, model, train_dataloader, val_dataloader, config, device, output_dir, experiment_name="training"):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics
        self.train_metrics = {'mse_list': [], 'cos_list': [], 'iva_list': [], 'lr_list': [], 'epoch_list': [], 'step_list': []}
        self.val_metrics = {'mse_list': [], 'cos_list': [], 'iva_list': [], 'lr_list': [], 'epoch_list': [], 'step_list': []}
    
    def save_checkpoint(self, is_best=False, suffix=""):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        filename = f"checkpoint_epoch_{self.current_epoch:03d}{suffix}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
            torch.save(checkpoint, best_filepath)
