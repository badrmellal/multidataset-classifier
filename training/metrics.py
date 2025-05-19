import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute metrics for model evaluation

    Args:
        eval_pred: Tuple of (logits, labels)
            - logits: numpy array of shape (n_samples, n_classes)
            - labels: numpy array of shape (n_samples,)

    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    # Basic metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    # ROC AUC if binary or multiclass
    try:
        if logits.shape[1] == 2:  # Binary classification
            # Convert probabilities using softmax
            prob = softmax(logits, axis=1)[:, 1]
            auc = roc_auc_score(labels, prob)
        else:  # Multiclass
            # One-hot encode labels
            y_true = np.zeros((labels.size, logits.shape[1]))
            y_true[np.arange(labels.size), labels] = 1

            # Convert logits to probabilities using softmax
            prob = softmax(logits, axis=1)

            # Compute AUC
            auc = roc_auc_score(y_true, prob, multi_class="ovr", average="weighted")
    except:
        # If AUC calculation fails, set to -1
        auc = -1.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc if auc != -1.0 else None
    }


def softmax(x: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Compute softmax values for each row of matrix x

    Args:
        x: Input array
        axis: Axis along which to compute softmax

    Returns:
        Softmax of x along specified axis
    """
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def compute_metrics_by_class(
        eval_pred: Tuple[np.ndarray, np.ndarray],
        label_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each class

    Args:
        eval_pred: Tuple of (logits, labels)
        label_names: List of label names

    Returns:
        Dictionary of class -> metrics
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    # Compute precision, recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None
    )

    # Organize metrics by class
    class_metrics = {}
    for i, label_name in enumerate(label_names):
        class_metrics[label_name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i]
        }

    return class_metrics