"""
src/bias/bias_analyzer.py
──────────────────────────
Bias and fairness analysis across language groups.

Measures:
  - Per-language accuracy, F1, FPR, FNR
  - Equalized Odds gap between language groups
  - Generates bias report for Streamlit display
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Core bias metrics
# ─────────────────────────────────────────────

def compute_fpr_fnr(y_true: List, y_pred: List) -> Dict:
    """
    Compute False Positive Rate and False Negative Rate.
    FPR = FP / (FP + TN)  → model wrongly flags clean content
    FNR = FN / (FN + TP)  → model misses actual hate
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        fpr, fnr = 0.0, 0.0
    return {"fpr": round(fpr, 4), "fnr": round(fnr, 4)}


def compute_group_metrics(
    df:        pd.DataFrame,
    pred_col:  str = "pred",
    label_col: str = "label",
    group_col: str = "language",
) -> pd.DataFrame:
    """
    Compute per-group fairness metrics.

    Returns DataFrame with columns:
        group, n_samples, accuracy, f1, fpr, fnr
    """
    rows = []
    for group in df[group_col].unique():
        subset = df[df[group_col] == group]
        y_true = subset[label_col].tolist()
        y_pred = subset[pred_col].tolist()

        if len(set(y_true)) < 2:
            logger.warning(f"Group '{group}' has only one class — skipping FPR/FNR.")

        fpr_fnr = compute_fpr_fnr(y_true, y_pred)

        rows.append({
            "Language Group": group,
            "Samples":        len(subset),
            "Accuracy":       round(accuracy_score(y_true, y_pred) * 100, 2),
            "F1-Score":       round(f1_score(y_true, y_pred, average="weighted",
                                             zero_division=0) * 100, 2),
            "FPR (%)":        round(fpr_fnr["fpr"] * 100, 2),
            "FNR (%)":        round(fpr_fnr["fnr"] * 100, 2),
        })

    return pd.DataFrame(rows).set_index("Language Group")


# ─────────────────────────────────────────────
#  Equalized Odds Gap
# ─────────────────────────────────────────────

def equalized_odds_gap(metrics_df: pd.DataFrame) -> Dict:
    """
    Equalized Odds: model should have similar FPR and FNR
    across all language groups.

    Gap = max - min across groups.
    Smaller gap = fairer model.
    """
    fpr_gap = round(metrics_df["FPR (%)"].max() - metrics_df["FPR (%)"].min(), 2)
    fnr_gap = round(metrics_df["FNR (%)"].max() - metrics_df["FNR (%)"].min(), 2)
    f1_gap  = round(metrics_df["F1-Score"].max() - metrics_df["F1-Score"].min(), 2)

    return {
        "fpr_gap": fpr_gap,
        "fnr_gap": fnr_gap,
        "f1_gap":  f1_gap,
        "verdict": _fairness_verdict(fpr_gap, fnr_gap, f1_gap),
    }


def _fairness_verdict(fpr_gap: float, fnr_gap: float, f1_gap: float) -> str:
    max_gap = max(fpr_gap, fnr_gap, f1_gap)
    if max_gap < 5.0:
        return "✅ Fair — gaps are within acceptable range (<5%)"
    elif max_gap < 10.0:
        return "⚠️ Moderate bias — consider rebalancing training data"
    else:
        return "❌ High bias — significant performance gap across language groups"


# ─────────────────────────────────────────────
#  Bias plots
# ─────────────────────────────────────────────

def plot_bias_comparison(
    metrics_df: pd.DataFrame,
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing F1, FPR, FNR across language groups.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = ["F1-Score", "FPR (%)", "FNR (%)"]
    colors  = ["#3498db", "#e74c3c", "#f39c12"]
    titles  = ["F1-Score by Language", "False Positive Rate", "False Negative Rate"]

    for ax, metric, color, title in zip(axes, metrics, colors, titles):
        groups = metrics_df.index.tolist()
        values = metrics_df[metric].tolist()

        bars = ax.bar(groups, values, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(metric, fontsize=9)
        ax.set_ylim(0, max(values) * 1.25 if values else 100)
        ax.tick_params(axis="x", rotation=15)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9
            )

    plt.suptitle("Bias Analysis — Performance Across Language Groups",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Bias plot saved → {save_path}")

    return fig


# ─────────────────────────────────────────────
#  Full bias report
# ─────────────────────────────────────────────

def generate_bias_report(
    df:        pd.DataFrame,
    pred_col:  str = "pred",
    label_col: str = "label",
    group_col: str = "language",
    save_path: Optional[str] = None,
) -> Dict:
    """
    Full bias analysis pipeline.

    Args:
        df: DataFrame with columns: text, label, pred, language

    Returns dict with:
        metrics_df, gaps, verdict, figure
    """
    logger.info("Running bias analysis …")

    metrics_df = compute_group_metrics(df, pred_col, label_col, group_col)
    gaps       = equalized_odds_gap(metrics_df)
    fig        = plot_bias_comparison(metrics_df, save_path=save_path)

    logger.info(f"Bias Analysis Complete:")
    logger.info(f"  F1 Gap  : {gaps['f1_gap']}%")
    logger.info(f"  FPR Gap : {gaps['fpr_gap']}%")
    logger.info(f"  FNR Gap : {gaps['fnr_gap']}%")
    logger.info(f"  Verdict : {gaps['verdict']}")

    return {
        "metrics_df": metrics_df,
        "gaps":       gaps,
        "verdict":    gaps["verdict"],
        "figure":     fig,
    }


# ─────────────────────────────────────────────
#  Smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic test
    np.random.seed(42)
    n = 300

    df_test = pd.DataFrame({
        "text":     [f"sample text {i}" for i in range(n)],
        "label":    np.random.randint(0, 2, n),
        "pred":     np.random.randint(0, 2, n),
        "language": np.random.choice(["en", "hi", "hinglish"], n),
    })

    report = generate_bias_report(df_test)
    print("\nMetrics per language group:")
    print(report["metrics_df"])
    print(f"\nVerdict: {report['verdict']}")
    print("Bias analyzer smoke test passed ✓")
