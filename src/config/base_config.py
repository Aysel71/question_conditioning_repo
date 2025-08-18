"""
Base configuration для Question-Conditioned LLaVA
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """Configuration для Selector модели"""
    
    # Architecture parameters
    visual_dim: int = 1024
    text_dim: int = 4096
    sparsity_factor: float = 0.4
    num_heads: int = 16
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.1
    
    # Loss weights
    mse_weight: float = 1.0
    cosine_weight: float = 0.1
    sparsity_weight: float = 0.01


@dataclass
class TrainingConfig:
    """Configuration для training"""
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 10
    log_interval: int = 100


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "question_conditioned_selector"
    output_dir: str = "outputs"


def create_config_variants():
    """Create different configuration variants"""
    
    configs = {}
    
    # Base configuration
    configs['base'] = ExperimentConfig()
    
    # High sparsity (more aggressive)
    configs['high_sparsity'] = ExperimentConfig()
    configs['high_sparsity'].model.sparsity_factor = 0.3
    configs['high_sparsity'].experiment_name = "high_sparsity_selector"
    
    # Debug (fast training)
    configs['debug'] = ExperimentConfig()
    configs['debug'].training.epochs = 2
    configs['debug'].training.batch_size = 8
    configs['debug'].experiment_name = "debug_selector"
    
    return configs
