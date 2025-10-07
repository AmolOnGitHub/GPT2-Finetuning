import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
from pathlib import Path


def compute_metrics(eval_pred):
    """
    Compute accuracy, precision, recall, and F1 score from model predictions.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def save_confusion_matrix(y_true, y_pred, label_names, out_path: str, title: str):
    """
    Save a confusion matrix plot to the specified path.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap="Blues", values_format="d", xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def classification_text_report(y_true, y_pred, label_names, path):
    """
    Save a text classification report to the specified path.
    """
    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(report)