from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import numpy as np


def evaluate_model(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    return metrics


def threshold_tuning(y_true, y_prob, metric_func=f1_score):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_t = 0.5
    best_score = 0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = metric_func(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score


def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, digits=4))
