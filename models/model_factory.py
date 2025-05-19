from typing import Dict, List, Optional
import os
import torch
from transformers import AutoTokenizer

from configs.base_config import Config
from .multi_head import MultiHeadClassifier


def create_model(config: Config) -> MultiHeadClassifier:
    """
    Create a multi-head model based on the configuration

    Args:
        config: Project configuration

    Returns:
        Initialized MultiHeadClassifier model
    """
    # Get dataset heads configuration
    dataset_heads = {}
    for dataset_id, dataset_config in config.datasets.items():
        dataset_heads[dataset_id] = dataset_config.num_labels

    # Create model
    model = MultiHeadClassifier(
        model_name=config.model.model_name,
        dataset_heads=dataset_heads,
        dropout_rate=config.model.dropout_rate,
        hidden_dim=config.model.hidden_dim
    )

    return model


def save_model(
        model: MultiHeadClassifier,
        tokenizer: AutoTokenizer,
        save_directory: str,
        config: Optional[Config] = None
):
    """
    Save model, tokenizer and config to directory

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        save_directory: Directory to save to
        config: Optional configuration to save
    """
    # Create directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Save model
    model.save_pretrained(save_directory)

    # Save tokenizer
    tokenizer.save_pretrained(save_directory)

    # Save config if provided
    if config:
        config.save(os.path.join(save_directory, "config.yaml"))


def _get_best_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_model(
    load_directory: str,
    device: str = None
) -> tuple:
    if device is None:
        device = _get_best_device()
    """
    Load model, tokenizer and config from directory

    Args:
        load_directory: Directory to load from
        device: Device to load model to

    Returns:
        Tuple of (model, tokenizer, config)
    """
    # Check if directory exists
    if not os.path.exists(load_directory):
        raise ValueError(f"Directory {load_directory} does not exist")

    # Load model
    model = MultiHeadClassifier.from_pretrained(load_directory)
    model = model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(load_directory)

    # Load config if exists
    config = None
    config_path = os.path.join(load_directory, "config.yaml")
    if os.path.exists(config_path):
        from configs.base_config import Config
        config = Config.from_file(config_path)

    return model, tokenizer, config