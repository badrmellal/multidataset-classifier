from typing import Dict
from .base_config import Config, DatasetConfig, ModelConfig, TrainingConfig

def get_news_classification_config() -> Config:
    """Configuration for news classification datasets"""
    config = Config(experiment_name="news_classification")

    # Model configuration
    config.model = ModelConfig(
        model_name="distilbert-base-uncased",
        max_length=256,
        dropout_rate=0.1
    )

    # Dataset configurations
    config.datasets = {
        "ag_news": DatasetConfig(
            name="AG News",
            hf_dataset_name="fancyzhx/ag_news",
            num_labels=4,
            label_names=["World", "Sports", "Business", "Sci/Tech"],  # Explicit labels
            batch_size=16,
            weight=1.0
        ),
        "bbc_news": DatasetConfig(
            name="BBC News",
            hf_dataset_name="SetFit/bbc-news",
            text_field="text",
            label_field="label",
            num_labels=5,
            label_names=["business", "entertainment", "politics", "sport", "tech"],  # Explicit labels
            batch_size=16,
            weight=1.0
        )
    }

    # Training configuration
    config.training = TrainingConfig(
        output_dir="./output/news_classification",
        learning_rate=2e-5,
        num_train_epochs=3,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        fp16=True
    )

    return config


def get_sentiment_classification_config() -> Config:
    """Configuration for sentiment analysis datasets"""
    config = Config(experiment_name="sentiment_classification")

    # Model configuration
    config.model = ModelConfig(
        model_name="distilbert-base-uncased",
        max_length=128,
        dropout_rate=0.1
    )

    # Dataset configurations
    config.datasets = {
        "sst2": DatasetConfig(
            name="SST-2",
            hf_dataset_name="stanfordnlp/sst2",
            text_field="sentence",
            batch_size=32,
            weight=1.0
        ),
        "imdb": DatasetConfig(
            name="IMDB",
            hf_dataset_name="stanfordnlp/imdb",
            batch_size=16,
            weight=1.0
        )
    }

    # Training configuration
    config.training = TrainingConfig(
        output_dir="./output/sentiment_classification",
        learning_rate=3e-5,
        num_train_epochs=4,
        warmup_ratio=0.1,
        weight_decay=0.01
    )

    return config


def get_available_configs() -> Dict[str, Config]:
    """Return all available configurations"""
    return {
        "news_classification": get_news_classification_config(),
        "sentiment_classification": get_sentiment_classification_config()
    }