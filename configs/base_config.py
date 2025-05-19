from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import os
import yaml
import json


@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    dropout_rate: float = 0.1
    hidden_dim: Optional[int] = None


@dataclass
class DatasetConfig:
    name: str
    hf_dataset_name: str  # Hugging Face dataset name
    text_field: str = "text"  # Field containing the text to classify
    label_field: str = "label"  # Field containing the label
    num_labels: int = -1  # Will be determined automatically if -1
    label_names: List[str] = field(default_factory=list)  # Will be filled automatically
    train_split: str = "train"
    test_split: str = "test"
    val_split: Optional[str] = None  # If None, will create from train
    val_size: float = 0.1
    batch_size: int = 16
    weight: float = 1.0  # Weight in multi-dataset training


@dataclass
class TrainingConfig:
    output_dir: str = "./output"
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    gradient_accumulation_steps: int = 1
    fp16: bool = True


@dataclass
class Config:
    experiment_name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = "auto"  # "auto", "mps", "cuda", or "cpu"

    def save(self, file_path: str):
        """Save config to file"""
        # Convert to dict first
        config_dict = {
            "experiment_name": self.experiment_name,
            "model": self.model.__dict__,
            "datasets": {k: v.__dict__ for k, v in self.datasets.items()},
            "training": self.training.__dict__,
            "seed": self.seed,
            "device": self.device
        }

        # Determine file type from extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif ext.lower() in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @classmethod
    def from_file(cls, file_path: str) -> 'Config':
        """Load config from file"""
        _, ext = os.path.splitext(file_path)

        if ext.lower() == '.json':
            with open(file_path) as f:
                config_dict = json.load(f)
        elif ext.lower() in ['.yaml', '.yml']:
            with open(file_path) as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Create config
        config = cls(
            experiment_name=config_dict["experiment_name"],
            seed=config_dict.get("seed", 42),
            device=config_dict.get("device", "mps")
        )

        # Set model config
        model_dict = config_dict.get("model", {})
        config.model = ModelConfig(**model_dict)

        # Set dataset configs
        for dataset_name, dataset_dict in config_dict.get("datasets", {}).items():
            config.datasets[dataset_name] = DatasetConfig(**dataset_dict)

        # Set training config
        training_dict = config_dict.get("training", {})
        config.training = TrainingConfig(**training_dict)

        return config