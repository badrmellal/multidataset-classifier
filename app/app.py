
import os
import sys


# Disable Streamlit's file watcher
os.environ["STREAMLIT_SERVER_WATCH_PATTERNS"] = ""

# Tell PyTorch to ignore MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


# Parent directory to path so we can import from configs, models, etc.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


# Lazy import for torch and related modules
def load_torch_modules():
    import torch
    return torch

from configs.base_config import Config
from models.model_factory import load_model
from inference.predictor import Predictor

# Fixed label maps for the datasets (fallback if not found in config)
FALLBACK_LABEL_MAPS = {
    "ag_news": ["World", "Sports", "Business", "Sci/Tech"],
    "bbc_news": ["business", "entertainment", "politics", "sport", "tech"],
    "sst2": ["negative", "positive"],
    "imdb": ["negative", "positive"]
}


def find_all_models():
    """Find all trained models in the output directory"""
    available_models = {}

    # Look in parent directory for output folder
    output_dir = os.path.join(parent_dir, "output")

    if not os.path.exists(output_dir):
        return available_models

    # Walk through the output directory to find models
    for root, dirs, files in os.walk(output_dir):
        # Look for directories containing model files
        if any(f in files for f in ["config.yaml", "config.yml"]) and \
                any(f in files for f in ["pytorch_model.bin", "model.safetensors"]):

            # Try to determine the model type from the path
            if "news" in root.lower():
                model_name = f"News Classification ({os.path.basename(root)})"
            elif "sentiment" in root.lower():
                model_name = f"Sentiment Analysis ({os.path.basename(root)})"
            else:
                model_name = f"Model ({os.path.basename(root)})"

            available_models[model_name] = root

    return available_models


# Cache loading of model resources to avoid reloading on each interaction
@st.cache_resource
def load_resources(model_dir: str):
    """Load model, tokenizer and config"""
    # Import torch here to avoid file watcher issues
    torch = load_torch_modules()

    model, tokenizer, config = load_model(model_dir)

    # Get label maps for all datasets
    label_maps = {}
    for dataset_id, dataset_config in config.datasets.items():
        if dataset_config.label_names:
            label_maps[dataset_id] = dataset_config.label_names
        else:
            # Use fallback label names if not found in config
            if dataset_id in FALLBACK_LABEL_MAPS:
                label_maps[dataset_id] = FALLBACK_LABEL_MAPS[dataset_id]
                st.warning(f"Using fallback label names for {dataset_id}")
            else:
                # Generate default labels if we have num_labels
                num_labels = dataset_config.num_labels
                label_maps[dataset_id] = [f"Label_{i}" for i in range(num_labels)]
                st.warning(f"Generated default label names for {dataset_id}")

    # Create predictor
    predictor = Predictor(
        model=model,
        tokenizer=tokenizer,
        dataset_label_maps=label_maps
    )

    return predictor, config


# App title and description
st.title("🤖 Multi-Dataset Text Classifier")
st.markdown(
    """
    This application classifies text using a transformer-based model trained on multiple datasets.
    You can select different datasets for classification and see confidence scores for the predictions.
    """
)

# Get available models
available_models = find_all_models()

if not available_models:
    st.error("No trained models found!")
    st.markdown("### Debugging Information")

    # Show what's in the output directory
    output_dir = os.path.join(parent_dir, "output")
    if os.path.exists(output_dir):
        st.markdown("**Contents of ./output directory:**")
        output_contents = []
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(output_dir, "").count(os.sep)
            indent = " " * 2 * level
            output_contents.append(f"{indent}📁 {os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                output_contents.append(f"{subindent}📄 {file}")

        for item in output_contents[:50]:  # Limit output
            st.text(item)

        if len(output_contents) > 50:
            st.text(f"... and {len(output_contents) - 50} more items")
    else:
        st.error(f"./output directory doesn't exist at {output_dir}")

    st.markdown("### Expected structure:")
    st.code("""
    ./output/
    ├── news_classification/
    │   └── news_classification_YYYYMMDD_HHMMSS/
    │       └── best_model/
    │           ├── config.yaml
    │           ├── pytorch_model.bin
    │           └── ...
    └── sentiment_classification/
        └── sentiment_classification_YYYYMMDD_HHMMSS/
            └── best_model/
                ├── config.yaml
                ├── pytorch_model.bin
                └── ...
    """)

    st.info("Please train your models first using the training script.")
    st.stop()

# Model selection
model_name = st.sidebar.selectbox(
    "Select Model",
    options=list(available_models.keys())
)

model_dir = available_models[model_name]

# Load model resources
try:
    predictor, config = load_resources(model_dir)
    st.sidebar.success(f"Model loaded successfully: {config.experiment_name}")

    # Show debug information about label maps
    with st.sidebar.expander("Debug: Label Maps"):
        for dataset_id, labels in predictor.dataset_label_maps.items():
            st.write(f"**{dataset_id}**: {labels}")

    # Dataset selection
    dataset_options = list(config.datasets.keys())
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        options=dataset_options,
        format_func=lambda x: config.datasets[x].name
    )

    # Get labels for selected dataset
    label_names = predictor.dataset_label_maps[selected_dataset]

    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Info"])

    # Tab 1: Single Prediction
    with tab1:
        st.subheader(f"Classify Text - {config.datasets[selected_dataset].name}")

        # Sample texts for testing
        sample_texts = {
            "news": [
                "Apple announces record quarterly earnings exceeding analyst expectations",
                "Local football team wins championship match 3-1",
                "Scientists discover new planet in distant solar system",
                "Stock market reaches all-time high amid economic recovery"
            ],
            "sentiment": [
                "This movie was absolutely fantastic! Best film I've ever seen.",
                "Terrible service, worst experience ever. Never going back.",
                "The product is okay, nothing special but does the job.",
                "I love this restaurant, amazing food and great atmosphere!"
            ]
        }

        # Determine sample type based on model name or datasets
        sample_type = "sentiment" if "sentiment" in model_name.lower() else "news"

        # Add sample text buttons
        st.markdown("**Try these sample texts:**")
        cols = st.columns(2)
        for i, sample in enumerate(sample_texts[sample_type][:4]):
            with cols[i % 2]:
                if st.button(f"Sample {i + 1}", key=f"sample_{i}"):
                    st.session_state.sample_text = sample

        # Text input
        text_input = st.text_area(
            "Enter text to classify",
            height=150,
            value=st.session_state.get('sample_text', ''),
            placeholder="Type or paste your text here..."
        )

        # Predict button
        if st.button("Classify", key="classify_single"):
            if text_input:
                with st.spinner("Classifying..."):
                    # Make prediction
                    prediction = predictor.predict_with_labels(
                        text_input,
                        selected_dataset,
                        return_probabilities=True
                    )[0]  # Get first result

                    # Display prediction
                    predicted_label, probabilities = prediction

                    # Show prediction
                    st.success(f"Predicted label: **{predicted_label}**")

                    # Show probabilities as bar chart
                    st.subheader("Confidence Scores")

                    # Sort probabilities
                    sorted_probs = sorted(
                        probabilities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    # Create dataframe for visualization
                    df = pd.DataFrame({
                        "Label": [label for label, _ in sorted_probs],
                        "Confidence": [prob for _, prob in sorted_probs]
                    })

                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(x="Confidence", y="Label", data=df, ax=ax)
                    ax.set_xlim(0, 1)
                    ax.grid(axis="x", alpha=0.3)
                    ax.set_xlabel("Confidence Score")
                    st.pyplot(fig)
            else:
                st.warning("Please enter some text to classify.")

    # Tab 2: Batch Prediction
    with tab2:
        st.subheader(f"Batch Classification - {config.datasets[selected_dataset].name}")

        # Create sample CSV for download
        if st.button("Download Sample CSV Template"):
            if "news" in model_name.lower():
                sample_df = pd.DataFrame({
                    'text': [
                        "Tech company reports strong quarterly results",
                        "Basketball team wins playoff game",
                        "New study reveals climate change impacts",
                        "Government announces new policy changes"
                    ]
                })
            else:
                sample_df = pd.DataFrame({
                    'text': [
                        "I absolutely love this product!",
                        "Not satisfied with the service",
                        "Pretty good overall experience",
                        "Waste of money, very disappointed"
                    ]
                })

            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="Download sample.csv",
                data=csv,
                file_name="sample_text_classification.csv",
                mime="text/csv"
            )

        # File upload
        uploaded_file = st.file_uploader("Upload a CSV file with texts to classify", type="csv")

        if uploaded_file is not None:
            # Read file
            df = pd.read_csv(uploaded_file)

            # Text column selection
            text_col = st.selectbox(
                "Select text column",
                options=df.columns
            )

            # Preview data
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Classify button
            if st.button("Classify Batch", key="classify_batch"):
                # Get texts
                texts = df[text_col].tolist()

                # Add progress bar
                progress_bar = st.progress(0)

                # Make predictions
                with st.spinner("Classifying texts..."):
                    batch_size = 32
                    predictions = []

                    # Process in smaller batches to show progress
                    for i in range(0, len(texts), batch_size):
                        end_idx = min(i + batch_size, len(texts))
                        batch_texts = texts[i:end_idx]

                        # Make batch prediction
                        batch_preds = predictor.predict_with_labels(
                            batch_texts,
                            selected_dataset
                        )

                        predictions.extend(batch_preds)

                        # Update progress
                        progress = min(end_idx / len(texts), 1.0)
                        progress_bar.progress(progress)

                # Add predictions to dataframe
                df["predicted_label"] = predictions

                # Display results
                st.subheader("Classification Results")
                st.dataframe(df)

                # Download link
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name="classification_results.csv",
                    mime="text/csv"
                )

                # Show label distribution
                st.subheader("Label Distribution")

                # Count predictions
                label_counts = df["predicted_label"].value_counts()

                # Create pie chart
                fig, ax = plt.subplots(figsize=(10, 6))
                label_counts.plot.pie(
                    autopct="%1.1f%%",
                    ax=ax,
                    startangle=90
                )
                plt.ylabel("")
                st.pyplot(fig)

    # Tab 3: Model Info
    with tab3:
        st.subheader("Model Information")

        # Display model details
        st.markdown(f"**Experiment Name:** {config.experiment_name}")
        st.markdown(f"**Base Model:** {config.model.model_name}")

        # Display dataset information
        st.subheader("Datasets")
        for dataset_id, dataset_config in config.datasets.items():
            with st.expander(f"{dataset_config.name} ({dataset_id})"):
                st.markdown(f"**HF Dataset:** {dataset_config.hf_dataset_name}")
                st.markdown(f"**Number of Labels:** {dataset_config.num_labels}")

                # Display labels
                if dataset_id in predictor.dataset_label_maps:
                    st.markdown("**Labels:**")
                    for i, label in enumerate(predictor.dataset_label_maps[dataset_id]):
                        st.markdown(f"- {i}: {label}")

                st.markdown(f"**Training Weight:** {dataset_config.weight}")

        # Training parameters
        with st.expander("Training Parameters"):
            training_params = {
                "Learning Rate": config.training.learning_rate,
                "Epochs": config.training.num_train_epochs,
                "Batch Size": "Variable per dataset",
                "Weight Decay": config.training.weight_decay,
                "Warmup Ratio": config.training.warmup_ratio,
                "FP16": "Yes" if config.training.fp16 else "No",
                "Gradient Accumulation Steps": config.training.gradient_accumulation_steps
            }

            # Create parameter table
            param_df = pd.DataFrame(list(training_params.items()),
                                    columns=["Parameter", "Value"])
            st.table(param_df)

except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.info("Please make sure you have trained a model first and specified the correct path.")
    st.code(f"Expected path: {model_dir}")
    st.exception(e)

# Sidebar - About section
with st.sidebar.expander("About"):
    st.markdown(
        """
        This application is an end-to-end deep learning system for text classification
        using transformer models. It supports multiple datasets and provides both 
        single and batch prediction capabilities.

        The model is trained using a multi-head architecture that allows for efficient
        training and prediction across various text classification tasks.

        **Created by Badr Mellal & Iliass Benayed for MIT Machine & Deep Learning Module**
        """
    )