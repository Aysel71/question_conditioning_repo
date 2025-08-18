"""
Loss Functions для Question-Conditioned Selector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SelectorLossFunction(nn.Module):
    """Комбинированная loss function для Selector"""
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        cosine_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        target_sparsity: float = 0.4
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.sparsity_weight = sparsity_weight
        self.target_sparsity = target_sparsity
        
    def forward(
        self,
        original_features: torch.Tensor,
        reconstructed: torch.Tensor,
        selection_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss"""
        
        losses = {}
        
        # MSE reconstruction loss
        losses['mse'] = F.mse_loss(original_features, reconstructed)
        
        # Cosine similarity loss
        orig_mean = original_features.mean(dim=1)
        recon_mean = reconstructed.mean(dim=1)
        cosine_sim = F.cosine_similarity(orig_mean, recon_mean, dim=-1)
        losses['cosine'] = 1.0 - cosine_sim.mean()
        
        # Sparsity loss
        current_sparsity = selection_mask.float().mean()
        target_tensor = torch.tensor(self.target_sparsity, device=selection_mask.device)
        losses['sparsity'] = F.mse_loss(current_sparsity, target_tensor)
        
        # Combined loss
        total_loss = (
            self.mse_weight * losses['mse'] +
            self.cosine_weight * losses['cosine'] +
            self.sparsity_weight * losses['sparsity']
        )
        
        losses['total'] = total_loss
        return losses
