#!/usr/bin/env python
"""
Main script for training multi-dataset text classification models
"""

import os
import logging
import argparse
import torch
import random
import numpy as np
from datetime import datetime

from configs.base_config import Config
from configs.dataset_configs import get_available_configs
from data.dataset_manager import DatasetManager
from models.model_factory import create_model
from training.batch_trainer import BatchTrainer


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str):
    """Setup logging"""
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "training.log"))
        ]
    )


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train multi-dataset text classifier")
    parser.add_argument(
        "--config", type=str,
        help="Path to config YAML file or name of predefined config"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (overrides config setting)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config setting)"
    )
    parser.add_argument(
        "--list_configs", action="store_true",
        help="List available predefined configs and exit"
    )

    args = parser.parse_args()

    # List available configs and exit
    if args.list_configs:
        print("Available predefined configs:")
        for config_name in get_available_configs().keys():
            print(f"  - {config_name}")
        return

    # Load config
    if not args.config:
        print("Error: --config argument is required")
        parser.print_help()
        return

    # Check if predefined config or file path
    available_configs = get_available_configs()
    if args.config in available_configs:
        config = available_configs[args.config]
        print(f"Loaded predefined config: {args.config}")
    else:
        try:
            config = Config.from_file(args.config)
            print(f"Loaded config from file: {args.config}")
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return

    # Override config settings if provided
    if args.output_dir:
        config.training.output_dir = args.output_dir

    if args.seed is not None:
        config.seed = args.seed

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        config.training.output_dir,
        f"{config.experiment_name}_{timestamp}"
    )
    config.training.output_dir = output_dir

    # Setup logging
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    # Log config
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of datasets: {len(config.datasets)}")
    for dataset_id, dataset_config in config.datasets.items():
        logger.info(f"Dataset: {dataset_id} ({dataset_config.name})")

    # Set random seed
    set_seed(config.seed)
    logger.info(f"Random seed set to {config.seed}")

    # Check for MPS, CUDA, then fallback to CPU
    def get_device(preferred_device="auto"):
        if preferred_device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        elif preferred_device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif preferred_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    device = torch.device(get_device(config.device))
    logger.info(f"Using device: {device}")

    # Create dataset manager
    logger.info("Loading datasets...")
    dataset_manager = DatasetManager(config)

    # Log dataset stats
    for dataset_id, dataset_config in config.datasets.items():
        train_dataset = dataset_manager.get_dataset(dataset_id, "train")
        val_dataset = dataset_manager.get_dataset(dataset_id, "validation")
        test_dataset = dataset_manager.get_dataset(dataset_id, "test")

        logger.info(f"Dataset {dataset_id}:")
        logger.info(f"  - Train: {len(train_dataset) if train_dataset else 0} examples")
        logger.info(f"  - Validation: {len(val_dataset) if val_dataset else 0} examples")
        logger.info(f"  - Test: {len(test_dataset) if test_dataset else 0} examples")
        logger.info(f"  - Labels: {dataset_config.num_labels}")

    # Create model
    logger.info(f"Creating model with base: {config.model.model_name}")
    model = create_model(config)

    # Log model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = BatchTrainer(
        config=config,
        model=model,
        dataset_manager=dataset_manager,
        output_dir=output_dir,
        device=device
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()