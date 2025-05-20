# BERT Multi-Dataset Text Classifier

## Overview
This project is an end-to-end deep learning system for training and deploying transformer-based text classification models across multiple datasets. The architecture supports batch training on various text classification tasks while sharing knowledge across datasets.


<img width="1512" alt="Screenshot 2025-05-19 at 5 25 15 PM" src="https://github.com/user-attachments/assets/35004e92-fa10-46a6-84d4-313aac25f369" />
<img width="1512" alt="Screenshot 2025-05-19 at 5 25 29 PM" src="https://github.com/user-attachments/assets/7a8c3f9b-a03d-4d57-a21c-5d3daf2fa3f3" />
<img width="1512" alt="Screenshot 2025-05-19 at 5 25 02 PM" src="https://github.com/user-attachments/assets/383083cd-ea34-40b1-9f98-3e04d8d617b0" />

## Features

- **Multi-dataset Training**: Train a single model on multiple datasets simultaneously with weighted batch training
- **Modular Architecture**: Clean separation of concerns with independent modules for configuration, data loading, model management, and training
- **Flexible Configuration**: YAML/JSON-based configuration system for easy experiment management
- **Multi-head Classification**: Specialized classification heads for each dataset that share a common transformer backbone
- **Comprehensive Metrics**: Performance tracking with accuracy, precision, recall, F1-score, and ROC AUC
- **Interactive UI**: Streamlit-based web interface for making predictions on trained models
- **Batch Processing**: Support for batch prediction with CSV upload and download
- **Visualizations**: Training metrics visualization and prediction confidence display


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/badrmellal/multi-dataset-classifier.git
   cd multi-dataset-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a Model

1. Using a predefined configuration:
   ```bash
   python run_training.py --config news_classification
   ```
      ```bash
   python run_training.py --config sentiment_classification
   ```

2. Using a custom configuration file:
   ```bash
   python run_training.py --config path/to/config.yaml
   ```

3. List available predefined configurations:
   ```bash
   python run_training.py --list_configs
   ```

### Deploying the Web Interface

```bash
streamlit run app/app.py
```

## Configuration Format

The project uses a YAML/JSON configuration format:

```yaml
experiment_name: "news_classification"
model:
  model_name: "distilbert-base-uncased"
  max_length: 256
  dropout_rate: 0.1
datasets:
  ag_news:
    name: "AG News"
    hf_dataset_name: "ag_news"
    num_labels: 4
    batch_size: 16
    weight: 1.0
  bbc_news:
    name: "BBC News"
    hf_dataset_name: "SetFit/bbc-news"
    num_labels: 5
    batch_size: 16
    weight: 1.0
training:
  output_dir: "./output/news_classification"
  learning_rate: 2e-5
  num_train_epochs: 3
  warmup_ratio: 0.1
  weight_decay: 0.01
seed: 42
device: "cuda"
```

## Adding a New Dataset

To add a new dataset:

1. Add the dataset configuration to `configs/dataset_configs.py`
2. Make sure your dataset is available on the Hugging Face Hub or provide a custom dataset loader

## Performance Metrics

The system tracks the following metrics during training and evaluation:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- ROC AUC (multi-class)

Metrics are logged to TensorBoard and can be viewed with:

```bash
tensorboard --logdir output/your_experiment/logs
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.15.0+
- Datasets 2.0.0+
- Streamlit 1.8.0+
- Scikit-learn 1.0.0+
- Pandas
- Matplotlib
- Seaborn


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Hugging Face for their Transformers and Datasets libraries
- Streamlit for the interactive UI framework
- This project was created by Badr Mellal & Iliass Benayed for the Machine and Deep Learning course at MIT
