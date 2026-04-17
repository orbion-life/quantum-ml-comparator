"""
Unified metrics computation for quantum vs classical ML comparison.
Supports binary and multiclass classification with comprehensive metric reporting.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
)


def compute_metrics(y_true, y_pred, y_prob=None, task="binary", class_names=None):
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like or None
        Predicted probabilities (for AUC-ROC).
        For binary: shape (n_samples,) or (n_samples, 2).
        For multiclass: shape (n_samples, n_classes).
    task : str
        'binary' or 'multiclass'.
    class_names : list of str or None
        Class names for per-class metrics.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1_macro, f1_weighted,
        auc_roc, per_class_f1, per_class_precision, per_class_recall,
        confusion_matrix, classification_report, mcc, cohen_kappa,
        n_samples, n_classes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_classes = len(np.unique(y_true))
    average = "binary" if task == "binary" and n_classes == 2 else "macro"

    results = {
        "n_samples": len(y_true),
        "n_classes": n_classes,
        "task": task,
    }

    # Core scalar metrics
    results["accuracy"] = float(accuracy_score(y_true, y_pred))
    results["precision"] = float(
        precision_score(y_true, y_pred, average=average, zero_division=0)
    )
    results["recall"] = float(
        recall_score(y_true, y_pred, average=average, zero_division=0)
    )
    results["f1_macro"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    results["f1_weighted"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    results["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    results["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)

    if class_names is not None:
        results["per_class_f1"] = {
            name: float(v) for name, v in zip(class_names, per_class_f1)
        }
        results["per_class_precision"] = {
            name: float(v) for name, v in zip(class_names, per_class_prec)
        }
        results["per_class_recall"] = {
            name: float(v) for name, v in zip(class_names, per_class_rec)
        }
    else:
        results["per_class_f1"] = {
            str(i): float(v) for i, v in enumerate(per_class_f1)
        }
        results["per_class_precision"] = {
            str(i): float(v) for i, v in enumerate(per_class_prec)
        }
        results["per_class_recall"] = {
            str(i): float(v) for i, v in enumerate(per_class_rec)
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results["confusion_matrix"] = cm.tolist()

    # Classification report (text and dict)
    target_names = class_names if class_names else [str(i) for i in range(n_classes)]
    results["classification_report"] = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    results["classification_report_dict"] = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True
    )

    # AUC-ROC (requires probabilities)
    results["auc_roc"] = _compute_auc(y_true, y_prob, task, n_classes)

    return results


def _compute_auc(y_true, y_prob, task, n_classes):
    """Safely compute AUC-ROC, returning None on failure."""
    if y_prob is None:
        return None

    y_prob = np.asarray(y_prob)

    try:
        if task == "binary" and n_classes == 2:
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            return float(roc_auc_score(y_true, y_prob))
        else:
            return float(
                roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
            )
    except ValueError:
        return None


def compare_models(results_dict):
    """
    Compare multiple models side by side.

    Parameters
    ----------
    results_dict : dict
        Mapping of model_name -> metrics dict from compute_metrics.

    Returns
    -------
    dict
        'ranking': list of model names sorted by f1_macro descending.
        'summary': dict of model_name -> key scalar metrics.
    """
    summary = {}
    for name, metrics in results_dict.items():
        summary[name] = {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "auc_roc": metrics.get("auc_roc"),
            "mcc": metrics["mcc"],
        }

    ranking = sorted(
        summary.keys(), key=lambda k: summary[k]["f1_macro"], reverse=True
    )

    return {
        "ranking": ranking,
        "summary": summary,
    }
