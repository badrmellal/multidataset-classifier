import os

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from transformers import AutoTokenizer

from models.multi_head import MultiHeadClassifier


class Predictor:
    """
    Predictor for making predictions with the trained model
    """

    def __init__(
            self,
            model: MultiHeadClassifier,
            tokenizer: AutoTokenizer,
            dataset_label_maps: Optional[Dict[str, List[str]]] = None,
            device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_label_maps = dataset_label_maps or {}

        def _get_best_device():
            # On Streamlit's servers, we never use MPS
            if os.getenv('STREAMLIT_SERVER_SENT_EVENTS'):  # Check if on Streamlit
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"

            # For local development, we can use MPS if available
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"

        self.device = device or _get_best_device()

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(
            self,
            text: Union[str, List[str]],
            dataset_id: str,
            return_probabilities: bool = False
    ) -> Union[List[int], List[Tuple[int, List[float]]]]:
        """
        Make predictions for input text

        Args:
            text: Input text or list of texts
            dataset_id: ID of the dataset to use for prediction
            return_probabilities: Whether to return probabilities along with predictions

        Returns:
            If return_probabilities is False:
                List of predicted class indices
            If return_probabilities is True:
                List of tuples (predicted_class, probabilities)
        """
        # Set active dataset
        self.model.set_active_dataset(dataset_id)

        # Convert single text to list
        if isinstance(text, str):
            text = [text]

        # Tokenize text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Make predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                dataset_id=dataset_id
            )

        # Get predictions
        logits = outputs["logits"].detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1).tolist()

        # Return predictions with probabilities if requested
        if return_probabilities:
            # Convert to probabilities using softmax
            probs = torch.nn.functional.softmax(
                torch.tensor(logits), dim=1
            ).numpy().tolist()
            return list(zip(predictions, probs))

        return predictions

    def predict_with_labels(
            self,
            text: Union[str, List[str]],
            dataset_id: str,
            return_probabilities: bool = False
    ) -> Union[List[str], List[Tuple[str, Dict[str, float]]]]:
        """
        Make predictions and return the predicted labels

        Args:
            text: Input text or list of texts
            dataset_id: ID of the dataset to use for prediction
            return_probabilities: Whether to return probabilities along with predictions

        Returns:
            If return_probabilities is False:
                List of predicted class labels
            If return_probabilities is True:
                List of tuples (predicted_label, {label: probability})
        """
        # Get label names for this dataset
        label_names = self.dataset_label_maps.get(dataset_id, [])
        if not label_names:
            raise ValueError(f"No label map found for dataset {dataset_id}")

        # Make predictions
        predictions = self.predict(text, dataset_id, return_probabilities)

        # Convert predictions to labels
        if return_probabilities:
            result = []
            for pred_idx, probs in predictions:
                pred_label = label_names[pred_idx]
                prob_map = {label_names[i]: prob for i, prob in enumerate(probs)}
                result.append((pred_label, prob_map))
            return result
        else:
            return [label_names[pred_idx] for pred_idx in predictions]

    def predict_batch(
            self,
            texts: List[str],
            dataset_id: str,
            batch_size: int = 32,
            return_probabilities: bool = False
    ) -> Union[List[int], List[Tuple[int, List[float]]]]:
        """
        Make predictions for a large batch of texts

        Args:
            texts: List of input texts
            dataset_id: ID of the dataset to use for prediction
            batch_size: Size of batches to process
            return_probabilities: Whether to return probabilities along with predictions

        Returns:
            If return_probabilities is False:
                List of predicted class indices
            If return_probabilities is True:
                List of tuples (predicted_class, probabilities)
        """
        # Set active dataset
        self.model.set_active_dataset(dataset_id)

        all_predictions = []
        all_probs = [] if return_probabilities else None

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # Make predictions
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    dataset_id=dataset_id
                )

            # Get predictions
            logits = outputs["logits"].detach().cpu().numpy()
            batch_predictions = np.argmax(logits, axis=1).tolist()
            all_predictions.extend(batch_predictions)

            # Calculate probabilities if requested
            if return_probabilities:
                probs = torch.nn.functional.softmax(
                    torch.tensor(logits), dim=1
                ).numpy().tolist()
                all_probs.extend(probs)

        # Return predictions with probabilities if requested
        if return_probabilities:
            return list(zip(all_predictions, all_probs))

        return all_predictions