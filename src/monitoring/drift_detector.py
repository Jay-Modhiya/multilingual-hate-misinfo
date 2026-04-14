"""
src/monitoring/drift_detector.py
──────────────────────────────────
Data drift detection using Kolmogorov-Smirnov test.

Detects when incoming text data shifts from training
distribution — triggers alert for model retraining.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Text feature extractor (simple, no GPU)
# ─────────────────────────────────────────────

def extract_text_features(texts: List[str]) -> np.ndarray:
    """
    Extract simple statistical features from texts.
    Used to compare distributions without needing embeddings.

    Features per text:
        0: text length (chars)
        1: word count
        2: avg word length
        3: punctuation ratio
        4: uppercase ratio
    """
    features = []
    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            features.append([0, 0, 0, 0, 0])
            continue

        words      = text.split()
        n_words    = len(words)
        avg_wlen   = np.mean([len(w) for w in words]) if words else 0
        punct_r    = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        upper_r    = sum(1 for c in text if c.isupper()) / len(text)

        features.append([len(text), n_words, avg_wlen, punct_r, upper_r])

    return np.array(features)


# ─────────────────────────────────────────────
#  KS Test drift detector
# ─────────────────────────────────────────────

class DriftDetector:
    """
    Detects data drift using Kolmogorov-Smirnov test.

    Compares incoming text feature distributions
    against the training reference distribution.
    """

    def __init__(self, cfg: dict):
        self.threshold   = cfg["monitoring"]["drift_threshold"]
        self.method      = cfg["monitoring"]["drift_method"]
        self.ref_features: Optional[np.ndarray] = None
        self.drift_log:    List[dict] = []
        logger.info(f"DriftDetector ready | method={self.method} | threshold={self.threshold}")

    def fit(self, reference_texts: List[str]):
        """Store reference (training) distribution."""
        self.ref_features = extract_text_features(reference_texts)
        logger.info(f"Reference distribution set | n={len(reference_texts)}")

    def detect(self, incoming_texts: List[str]) -> Dict:
        """
        Run KS test between reference and incoming texts.

        Returns:
            drift_detected: bool
            feature_results: per-feature KS stats and p-values
            overall_verdict: string summary
        """
        if self.ref_features is None:
            raise ValueError("Call fit() with training texts first.")

        incoming_features = extract_text_features(incoming_texts)
        feature_names     = ["text_length", "word_count", "avg_word_len",
                             "punct_ratio", "upper_ratio"]

        results   = {}
        drifted   = []

        for i, fname in enumerate(feature_names):
            ref_col = self.ref_features[:, i]
            inc_col = incoming_features[:, i]

            ks_stat, p_value = stats.ks_2samp(ref_col, inc_col)
            is_drift = p_value < self.threshold

            results[fname] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value":      round(float(p_value), 4),
                "drift":        is_drift,
            }
            if is_drift:
                drifted.append(fname)

        overall_drift = len(drifted) > 0
        verdict = (
            f"⚠️ Drift detected in: {', '.join(drifted)}"
            if overall_drift
            else "✅ No significant drift detected"
        )

        log_entry = {
            "timestamp":     datetime.now().isoformat(),
            "n_incoming":    len(incoming_texts),
            "drift_detected": overall_drift,
            "drifted_features": drifted,
            "verdict":       verdict,
            "feature_results": results,
        }
        self.drift_log.append(log_entry)

        logger.info(f"Drift check: {verdict}")
        return log_entry

    def plot_distribution(
        self,
        incoming_texts: List[str],
        feature:        str = "text_length",
        save_path:      Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot reference vs incoming distribution for one feature.
        """
        feature_names = ["text_length", "word_count", "avg_word_len",
                         "punct_ratio", "upper_ratio"]
        idx = feature_names.index(feature)

        ref_vals = self.ref_features[:, idx]
        inc_vals = extract_text_features(incoming_texts)[:, idx]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(ref_vals, bins=40, alpha=0.6, label="Training (reference)",
                color="#3498db", density=True)
        ax.hist(inc_vals, bins=40, alpha=0.6, label="Incoming (production)",
                color="#e74c3c", density=True)

        ax.set_xlabel(feature.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"Distribution Comparison — {feature}", fontsize=11,
                     fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Drift plot saved → {save_path}")

        return fig

    def get_drift_summary(self) -> pd.DataFrame:
        """Return drift log as DataFrame for Streamlit display."""
        if not self.drift_log:
            return pd.DataFrame()
        rows = [{
            "Timestamp":  e["timestamp"],
            "Incoming":   e["n_incoming"],
            "Drift":      "Yes" if e["drift_detected"] else "No",
            "Verdict":    e["verdict"],
        } for e in self.drift_log]
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  Smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    ref_texts = [
        "This is a normal tweet about politics.",
        "I love cricket and football.",
        "The weather is nice today.",
    ] * 100

    new_texts = [
        "HATE HATE HATE!!!!! Die you!!!!!",
        "FAKE NEWS FAKE NEWS FAKE NEWS!!!!",
        "LIES EVERYWHERE!!!",
    ] * 100

    detector = DriftDetector(cfg)
    detector.fit(ref_texts)

    result = detector.detect(new_texts)
    print("Verdict:", result["verdict"])
    print("Drifted features:", result["drifted_features"])
    print("Drift detector smoke test passed ✓")
