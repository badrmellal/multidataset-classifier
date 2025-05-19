import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional


class MultiHeadClassifier(nn.Module):
    """
    Transformer model with multiple classification heads for different datasets
    """

    def __init__(
            self,
            model_name: str,
            dataset_heads: Dict[str, int],  # dataset_id -> num_labels
            dropout_rate: float = 0.1,
            hidden_dim: Optional[int] = None
    ):
        super().__init__()

        # Load base transformer model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        # Determine hidden dimension
        self.hidden_dim = hidden_dim or self.config.hidden_size

        # Create classification heads for each dataset
        self.dataset_heads = nn.ModuleDict()
        for dataset_id, num_labels in dataset_heads.items():
            self.dataset_heads[dataset_id] = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.config.hidden_size, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.hidden_dim, num_labels)
            )

        # Active dataset tracker
        self.active_dataset_id = None

    def set_active_dataset(self, dataset_id: str):
        """Set the active dataset for forward pass"""
        if dataset_id not in self.dataset_heads:
            raise ValueError(f"Dataset {dataset_id} not found in model heads")
        self.active_dataset_id = dataset_id

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_id: Optional[str] = None,
            labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through the model

        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            dataset_id: ID of the dataset to use for classification (overrides active_dataset_id)
            labels: Optional labels for loss calculation [batch_size]

        Returns:
            Dictionary containing logits and loss (if labels provided)
        """
        # Use provided dataset_id or fall back to active_dataset_id
        current_dataset_id = dataset_id or self.active_dataset_id

        if not current_dataset_id:
            raise ValueError("No active dataset set. Call set_active_dataset first or provide dataset_id.")

        if current_dataset_id not in self.dataset_heads:
            raise ValueError(f"Dataset {current_dataset_id} not found in model heads")

        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0]

        # Forward through the classification head for this dataset
        logits = self.dataset_heads[current_dataset_id](pooled_output)

        # Prepare result
        result = {"logits": logits}

        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            result["loss"] = loss_fct(logits.view(-1, self.dataset_heads[current_dataset_id][-1].out_features),
                                      labels.view(-1))

        return result

    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        # Save transformer
        self.transformer.save_pretrained(save_directory)

        # Save classification heads
        heads_dict = {k: v.state_dict() for k, v in self.dataset_heads.items()}
        torch.save(heads_dict, f"{save_directory}/classification_heads.pt")

        # Save config info with dataset heads
        config_dict = self.config.to_dict()
        config_dict["dataset_heads"] = {k: v[-1].out_features for k, v in self.dataset_heads.items()}
        config_dict["hidden_dim"] = self.hidden_dim

        with open(f"{save_directory}/multi_head_config.json", "w") as f:
            import json
            json.dump(config_dict, f)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "MultiHeadClassifier":
        """Load model from directory"""
        import json
        import os

        # Load config
        with open(f"{load_directory}/multi_head_config.json") as f:
            config_dict = json.load(f)

        # Extract dataset heads and hidden_dim
        dataset_heads = config_dict.pop("dataset_heads", {})
        hidden_dim = config_dict.pop("hidden_dim", None)

        # Create model
        model = cls(
            model_name=load_directory,
            dataset_heads=dataset_heads,
            hidden_dim=hidden_dim
        )

        # Load classification heads
        heads_dict = torch.load(f"{load_directory}/classification_heads.pt")
        for dataset_id, state_dict in heads_dict.items():
            model.dataset_heads[dataset_id].load_state_dict(state_dict)

        return model