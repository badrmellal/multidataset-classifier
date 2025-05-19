"""
Configuration module for managing experiment configurations.
"""

from .base_config import Config, ModelConfig, DatasetConfig, TrainingConfig
from .dataset_configs import get_available_configs

__all__ = [
    'Config',
    'ModelConfig',
    'DatasetConfig',
    'TrainingConfig',
    'get_available_configs'
]