"""
Question-Conditioned Visual Selector
Адаптация VQVAE архитектуры для question-aware visual token selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class QuestionProjector(nn.Module):
    """Проекция question embeddings в visual space"""
    
    def __init__(self, text_dim: int = 4096, visual_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(text_dim, visual_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(visual_dim * 2, visual_dim),
            nn.LayerNorm(visual_dim)
        )
    
    def forward(self, question_embeds: torch.Tensor) -> torch.Tensor:
        return self.projection(question_embeds)


class CrossAttentionLayer(nn.Module):
    """Cross-attention между visual features и question"""
    
    def __init__(self, visual_dim: int = 1024, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(visual_dim)
        
    def forward(self, visual_features: torch.Tensor, question_features: torch.Tensor):
        conditioned, attention_weights = self.cross_attention(
            query=visual_features,
            key=question_features,
            value=question_features
        )
        conditioned_visual = self.norm(visual_features + conditioned)
        return conditioned_visual, attention_weights


class ImportancePredictor(nn.Module):
    """Предсказание важности каждого visual patch"""
    
    def __init__(self, visual_dim: int = 1024, hidden_dims: list = [512, 256, 128], dropout: float = 0.1):
        super().__init__()
        
        layers = []
        input_dim = visual_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        layers.extend([nn.Linear(input_dim, 1), nn.Sigmoid()])
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, conditioned_visual: torch.Tensor) -> torch.Tensor:
        return self.predictor(conditioned_visual)


class SparseSelector(nn.Module):
    """Sparse selection based on importance scores"""
    
    def __init__(self, sparsity_factor: float = 0.4):
        super().__init__()
        self.sparsity_factor = sparsity_factor
    
    def forward(self, visual_features: torch.Tensor, importance_scores: torch.Tensor):
        batch_size, num_patches, visual_dim = visual_features.shape
        k = int(num_patches * self.sparsity_factor)
        
        scores = importance_scores.squeeze(-1)
        _, indices = torch.topk(scores, k, dim=1)
        
        # Create mask
        mask = torch.zeros_like(scores)
        mask.scatter_(1, indices, 1.0)
        
        # Apply mask and get compact representation
        mask_expanded = mask.unsqueeze(-1)
        selected_compact = torch.gather(
            visual_features, 1, 
            indices.unsqueeze(-1).expand(-1, -1, visual_dim)
        )
        
        return selected_compact, mask_expanded, indices


class QuestionConditionedSelector(nn.Module):
    """
    Полная архитектура Question-Conditioned Selector
    """
    
    def __init__(
        self,
        visual_dim: int = 1024,
        text_dim: int = 4096,
        sparsity_factor: float = 0.4,
        num_heads: int = 16,
        hidden_dims: list = [512, 256, 128],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.sparsity_factor = sparsity_factor
        
        # Core components
        self.question_projector = QuestionProjector(text_dim, visual_dim, dropout)
        self.cross_attention = CrossAttentionLayer(visual_dim, num_heads, dropout)
        self.importance_predictor = ImportancePredictor(visual_dim, hidden_dims, dropout)
        self.sparse_selector = SparseSelector(sparsity_factor)
        
        # Simple reconstruction decoder for training
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(visual_dim, visual_dim * 2),
            nn.ReLU(),
            nn.Linear(visual_dim * 2, visual_dim)
        )
        
    def forward(self, visual_features: torch.Tensor, question_embeds: torch.Tensor):
        """
        Args:
            visual_features: [batch_size, num_patches, visual_dim] 
            question_embeds: [batch_size, seq_len, text_dim]
            
        Returns:
            Dictionary with selected features, masks, and reconstruction
        """
        # 1. Project question to visual space
        question_projected = self.question_projector(question_embeds)
        
        # 2. Cross-attention conditioning
        conditioned_visual, attention_weights = self.cross_attention(
            visual_features, question_projected
        )
        
        # 3. Predict importance scores
        importance_scores = self.importance_predictor(conditioned_visual)
        
        # 4. Sparse selection
        selected_features, selection_mask, selected_indices = self.sparse_selector(
            visual_features, importance_scores
        )
        
        # 5. Reconstruction for training
        reconstructed_patches = self.reconstruction_decoder(selected_features)
        
        # Expand reconstruction to full size for loss computation
        batch_size, num_patches, visual_dim = visual_features.shape
        reconstructed = torch.zeros_like(visual_features)
        reconstructed.scatter_(
            1, 
            selected_indices.unsqueeze(-1).expand(-1, -1, visual_dim),
            reconstructed_patches
        )
        
        return {
            'selected_features': selected_features,
            'importance_scores': importance_scores,
            'selection_mask': selection_mask,
            'reconstructed': reconstructed,
            'selected_indices': selected_indices,
            'attention_weights': attention_weights
        }
    
    def get_sparsity_stats(self, selection_mask: torch.Tensor) -> Dict[str, float]:
        """Get statistics about sparsity"""
        selected_ratio = selection_mask.float().mean().item()
        return {
            'selected_ratio': selected_ratio,
            'target_ratio': self.sparsity_factor,
            'sparsity_error': abs(selected_ratio - self.sparsity_factor)
        }
