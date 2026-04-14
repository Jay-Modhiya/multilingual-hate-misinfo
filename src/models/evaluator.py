"""
src/models/evaluator.py
────────────────────────
Model evaluation: accuracy, precision, recall, F1, confusion matrix.

Mirrors the CA-3 evaluation approach (5-metric comparison) but extended for:
  - Multilingual per-language breakdown
  - Saving charts to disk for Streamlit display
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Core metrics
# ─────────────────────────────────────────────

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_map: Dict[int, str],
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    Matches Table I from CA-3 report.
    """
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, average=average, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred, average=average, zero_division=0) * 100, 2),
        "f1":        round(f1_score(y_true, y_pred, average=average, zero_division=0) * 100, 2),
    }


def full_report(
    y_true: List[int],
    y_pred: List[int],
    label_map: Dict[int, str],
) -> str:
    """Per-class classification report string."""
    target_names = [label_map[i] for i in sorted(label_map.keys())]
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


# ─────────────────────────────────────────────
#  Confusion matrix plot
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    y_true:    List[int],
    y_pred:    List[int],
    label_map: Dict[int, str],
    title:     str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Generates a normalised confusion matrix.
    Mirrors Fig. 3 from CA-3 report.
    """
    labels      = sorted(label_map.keys())
    tick_labels = [label_map[l] for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.round(cm, 2)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=".2f" if normalize else "d",
        cmap="Blues", ax=ax,
        xticklabels=tick_labels, yticklabels=tick_labels,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label",      fontsize=10)
    ax.set_title(title,              fontsize=11)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved → {save_path}")

    return fig


# ─────────────────────────────────────────────
#  Per-language breakdown
# ─────────────────────────────────────────────

def per_language_metrics(
    df: pd.DataFrame,
    pred_col:  str = "pred",
    label_col: str = "label",
    lang_col:  str = "language",
    label_map: Dict[int, str] = None,
) -> pd.DataFrame:
    """
    Break down F1 / accuracy per language group.
    Used in bias analysis section.
    """
    results = []
    for lang in df[lang_col].unique():
        subset = df[df[lang_col] == lang]
        m = compute_metrics(
            subset[label_col].tolist(),
            subset[pred_col].tolist(),
            label_map or {},
        )
        m["language"] = lang
        m["n_samples"] = len(subset)
        results.append(m)
    return pd.DataFrame(results).set_index("language")


# ─────────────────────────────────────────────
#  Summary table (Streamlit-ready)
# ─────────────────────────────────────────────

def build_summary_table(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Given { model_name: metrics_dict }, return a formatted DataFrame.
    Matches Table I from CA-3.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            "Model":     model_name,
            "Accuracy":  f"{metrics['accuracy']:.2f}%",
            "Precision": f"{metrics['precision']:.2f}%",
            "Recall":    f"{metrics['recall']:.2f}%",
            "F1-Score":  f"{metrics['f1']:.2f}%",
        })
    return pd.DataFrame(rows).set_index("Model")


# ─────────────────────────────────────────────
#  Model inference helper
# ─────────────────────────────────────────────

def predict_batch(model, dataloader, device) -> tuple:
    """Run inference on a DataLoader; return (all_preds, all_labels)."""
    import torch
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            batch  = {k: v.to(device) for k, v in batch.items()}
            _, logits = model(**batch)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────
#  Smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1]
    label_map = {0: "not_hate", 1: "hate"}

    m = compute_metrics(y_true, y_pred, label_map)
    print("Metrics:", m)
    print(full_report(y_true, y_pred, label_map))

    fig = plot_confusion_matrix(y_true, y_pred, label_map, title="Test Matrix")
    plt.show()
    print("Evaluator smoke test passed ✓")
