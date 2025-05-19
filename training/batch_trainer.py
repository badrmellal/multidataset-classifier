import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from configs.base_config import Config
from data.dataset_manager import DatasetManager
from models.multi_head import MultiHeadClassifier
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


class BatchTrainer:
    """
    Trainer for batch training multiple datasets
    """

    def __init__(
            self,
            config: Config,
            model: MultiHeadClassifier,
            dataset_manager: DatasetManager,
            output_dir: Optional[str] = None,
            device: Optional[str] = None
    ):
        self.config = config
        self.model = model
        self.dataset_manager = dataset_manager

        def _get_best_device():
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"

        self.device = device or _get_best_device()
        self.output_dir = output_dir or config.training.output_dir

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize tensorboard writer
        self.tb_writer = SummaryWriter(
            log_dir=os.path.join(self.output_dir, "logs")
        )

        # Set model to device
        self.model = self.model.to(self.device)

    def _create_dataloaders(self):
        """Create dataloaders for all datasets"""
        train_loaders = {}
        val_loaders = {}
        test_loaders = {}

        for dataset_id, dataset_config in self.config.datasets.items():
            # Get dataset splits
            train_dataset = self.dataset_manager.get_dataset(dataset_id, "train")
            val_dataset = self.dataset_manager.get_dataset(dataset_id, "validation")
            test_dataset = self.dataset_manager.get_dataset(dataset_id, "test")

            # Create dataloaders if datasets exist
            if train_dataset:
                train_loaders[dataset_id] = DataLoader(
                    train_dataset,
                    sampler=RandomSampler(train_dataset),
                    batch_size=dataset_config.batch_size
                )

            if val_dataset:
                val_loaders[dataset_id] = DataLoader(
                    val_dataset,
                    sampler=SequentialSampler(val_dataset),
                    batch_size=dataset_config.batch_size
                )

            if test_dataset:
                test_loaders[dataset_id] = DataLoader(
                    test_dataset,
                    sampler=SequentialSampler(test_dataset),
                    batch_size=dataset_config.batch_size
                )

        return train_loaders, val_loaders, test_loaders

    def train(self):
        """Train model on all datasets"""
        # Create dataloaders
        train_loaders, val_loaders, test_loaders = self._create_dataloaders()

        # No training datasets found
        if not train_loaders:
            logger.error("No training datasets found")
            return

        # Calculate total number of training steps
        total_steps = sum(
            len(dataloader) for dataloader in train_loaders.values()) * self.config.training.num_train_epochs

        # Prepare optimizer and schedule
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.training.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.training.warmup_ratio),
            num_training_steps=total_steps
        )

        # Train
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {self.config.training.num_train_epochs}")
        logger.info(f"  Total optimization steps = {total_steps}")

        global_step = 0
        best_val_metric = float("-inf")

        # Training loop
        for epoch in range(int(self.config.training.num_train_epochs)):
            epoch_iterator = tqdm(range(total_steps // int(self.config.training.num_train_epochs)),
                                  desc=f"Epoch {epoch + 1}/{self.config.training.num_train_epochs}")

            # Training phase
            self.model.train()
            epoch_loss = 0.0

            # Iterate over datasets
            for dataset_id, dataloader in train_loaders.items():
                dataset_config = self.config.datasets[dataset_id]

                # Set active dataset
                self.model.set_active_dataset(dataset_id)

                # Iterate over batches
                for batch in dataloader:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # Forward pass
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        dataset_id=dataset_id,
                        labels=batch["labels"]
                    )

                    # Apply dataset weight
                    loss = outputs["loss"] * dataset_config.weight

                    # Backward pass
                    loss.backward()

                    # Update metrics
                    epoch_loss += loss.item()

                    # Update parameters
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    epoch_iterator.update(1)

                    # Log metrics
                    if global_step % 100 == 0:
                        self.tb_writer.add_scalar("train/loss", loss.item(), global_step)
                        self.tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            # Log epoch metrics
            avg_epoch_loss = epoch_loss / sum(len(dl) for dl in train_loaders.values())
            logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
            self.tb_writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)

            # Validation phase
            if val_loaders:
                val_metrics = self.evaluate(val_loaders)

                # Log validation metrics
                for dataset_id, metrics in val_metrics.items():
                    for metric_name, metric_value in metrics.items():
                        self.tb_writer.add_scalar(f"val/{dataset_id}/{metric_name}",
                                                  metric_value,
                                                  epoch)

                # Overall validation metric (weighted average of all datasets)
                overall_metric = self._calculate_overall_metric(val_metrics)
                logger.info(
                    f"Epoch {epoch + 1} - Overall Validation {self.config.training.metric_for_best_model}: {overall_metric:.4f}")

                # Save best model
                if self.config.training.load_best_model_at_end and overall_metric > best_val_metric:
                    best_val_metric = overall_metric
                    logger.info(
                        f"New best model with {self.config.training.metric_for_best_model}: {best_val_metric:.4f}")

                    # Save model
                    self._save_model(os.path.join(self.output_dir, "best_model"))

            # Save checkpoint
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.config.training.num_train_epochs:
                self._save_model(os.path.join(self.output_dir, f"checkpoint-{epoch + 1}"))

        # Final evaluation on test set
        if test_loaders:
            logger.info("***** Running evaluation on test set *****")
            test_metrics = self.evaluate(test_loaders)

            # Log test metrics
            for dataset_id, metrics in test_metrics.items():
                logger.info(f"Test results for {dataset_id}:")
                for metric_name, metric_value in metrics.items():
                    logger.info(f"  {metric_name}: {metric_value:.4f}")

            # Overall test metric
            overall_metric = self._calculate_overall_metric(test_metrics)
            logger.info(f"Overall Test {self.config.training.metric_for_best_model}: {overall_metric:.4f}")

        # Close tensorboard writer
        self.tb_writer.close()

        return global_step, best_val_metric

    def evaluate(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on multiple datasets

        Args:
            dataloaders: Dictionary of dataset_id -> dataloader

        Returns:
            Dictionary of dataset_id -> metrics
        """
        # Set model to evaluation mode
        self.model.eval()

        # Results dictionary
        results = {}

        # Evaluate on each dataset
        for dataset_id, dataloader in dataloaders.items():
            # Set active dataset
            self.model.set_active_dataset(dataset_id)

            # Collect predictions and labels
            all_logits = []
            all_labels = []

            # Disable gradient computation
            with torch.no_grad():
                for batch in dataloader:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # Forward pass
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        dataset_id=dataset_id
                    )

                    # Collect outputs
                    all_logits.append(outputs["logits"].detach().cpu().numpy())
                    all_labels.append(batch["labels"].detach().cpu().numpy())

            # Concatenate all batches
            all_logits = np.concatenate(all_logits, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            # Compute metrics
            metrics = compute_metrics((all_logits, all_labels))

            # Store results
            results[dataset_id] = metrics

        return results

    def _calculate_overall_metric(self, metrics: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate overall metric as weighted average across datasets

        Args:
            metrics: Dictionary of dataset_id -> metrics

        Returns:
            Weighted average of specified metric across datasets
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for dataset_id, dataset_metrics in metrics.items():
            if self.config.training.metric_for_best_model in dataset_metrics:
                dataset_config = self.config.datasets[dataset_id]
                weighted_sum += dataset_metrics[self.config.training.metric_for_best_model] * dataset_config.weight
                total_weight += dataset_config.weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _save_model(self, output_dir: str):
        """
        Save model, tokenizer and config to output directory

        Args:
            output_dir: Directory to save to
        """
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving model to {output_dir}")

        # Save model
        self.model.save_pretrained(output_dir)

        # Save tokenizer
        self.dataset_manager.tokenizer.save_pretrained(output_dir)

        # Save config
        self.config.save(os.path.join(output_dir, "config.yaml"))

        logger.info(f"Model saved to {output_dir}")