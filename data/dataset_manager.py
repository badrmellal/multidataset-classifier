from typing import Dict, List, Tuple, Optional, Union
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from configs.base_config import DatasetConfig, Config


class DatasetManager:
    """Manager for handling multiple datasets"""

    def __init__(self, config: Config, tokenizer_name: Optional[str] = None):
        self.config = config
        self.datasets: Dict[str, Dict[str, Dataset]] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or config.model.model_name
        )
        self._load_all_datasets()

    def _load_all_datasets(self):
        """Load all datasets specified in the configuration"""
        for dataset_id, dataset_config in self.config.datasets.items():
            self._load_single_dataset(dataset_id, dataset_config)

    def _load_single_dataset(self, dataset_id: str, dataset_config: DatasetConfig):
        """Load a single dataset and preprocess it"""
        # Load dataset from Hugging Face
        try:
            raw_dataset = load_dataset(dataset_config.hf_dataset_name)
        except Exception as e:
            print(f"Error loading dataset {dataset_config.hf_dataset_name}: {e}")
            return

        # Extract splits
        splits = {}
        if dataset_config.train_split in raw_dataset:
            splits['train'] = raw_dataset[dataset_config.train_split]
        if dataset_config.test_split in raw_dataset:
            splits['test'] = raw_dataset[dataset_config.test_split]
        if dataset_config.val_split in raw_dataset:
            splits['validation'] = raw_dataset[dataset_config.val_split]

        # Create validation set from train if not provided
        if 'validation' not in splits and 'train' in splits and dataset_config.val_size > 0:
            train_val = splits['train'].train_test_split(
                test_size=dataset_config.val_size,
                seed=self.config.seed
            )
            splits['train'] = train_val['train']
            splits['validation'] = train_val['test']

        # Get label names if not provided
        if not dataset_config.label_names and dataset_config.num_labels == -1:
            # Try to get label names from the dataset
            if hasattr(raw_dataset[dataset_config.train_split].features[dataset_config.label_field], 'names'):
                dataset_config.label_names = raw_dataset[dataset_config.train_split].features[
                    dataset_config.label_field].names
                dataset_config.num_labels = len(dataset_config.label_names)
            else:
                # If label names not available, get unique values
                unique_labels = set()
                for split in splits.values():
                    unique_labels.update(split[dataset_config.label_field])
                dataset_config.num_labels = len(unique_labels)

        # Tokenize datasets
        for split_name, split_data in splits.items():
            splits[split_name] = self._tokenize_dataset(split_data, dataset_config)

        self.datasets[dataset_id] = splits

    def _tokenize_dataset(self, dataset: Dataset, dataset_config: DatasetConfig) -> Dataset:
        """Tokenize a dataset using the configured tokenizer"""

        def preprocess_function(examples):
            # Extract text from the configured field
            texts = examples[dataset_config.text_field]

            # Validate and clean texts to ensure proper format
            valid_texts = []
            for text in texts:
                if text is None:
                    text = ""  # Replace None with empty string
                elif isinstance(text, (list, dict)):
                    text = str(text)  # Convert complex types to string
                elif not isinstance(text, str):
                    text = str(text)  # Convert any other non-string to string
                valid_texts.append(text)

            # Tokenize the valid texts
            tokenized = self.tokenizer(
                valid_texts,
                padding="max_length",
                truncation=True,
                max_length=self.config.model.max_length
            )

            # Handle labels - we ensure they are single integers (simple fix for multi-label datasets)
            labels = examples[dataset_config.label_field]
            processed_labels = []

            for label in labels:
                if isinstance(label, list):
                    # For multi-label datasets like Reuters, we take the first label only
                    processed_labels.append(int(label[0]) if label else 0)
                else:
                    # For single-label datasets, we ensure it's an integer
                    processed_labels.append(int(label))

            tokenized["labels"] = processed_labels

            return tokenized

        # Apply preprocessing
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Set format for PyTorch
        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        return tokenized_dataset

    def get_dataset(self, dataset_id: str, split: str) -> Optional[Dataset]:
        """Get a specific dataset split"""
        if dataset_id not in self.datasets:
            return None
        return self.datasets[dataset_id].get(split)

    def get_all_datasets(self, split: str) -> Dict[str, Dataset]:
        """Get all datasets for a specific split"""
        result = {}
        for dataset_id, dataset_splits in self.datasets.items():
            if split in dataset_splits:
                result[dataset_id] = dataset_splits[split]
        return result

    def get_dataset_config(self, dataset_id: str) -> Optional[DatasetConfig]:
        """Get configuration for a specific dataset"""
        return self.config.datasets.get(dataset_id)

    def get_num_labels(self, dataset_id: str) -> int:
        """Get number of labels for a specific dataset"""
        dataset_config = self.get_dataset_config(dataset_id)
        if dataset_config:
            return dataset_config.num_labels
        return 0

    def get_label_names(self, dataset_id: str) -> List[str]:
        """Get label names for a specific dataset"""
        dataset_config = self.get_dataset_config(dataset_id)
        if dataset_config:
            return dataset_config.label_names
        return []