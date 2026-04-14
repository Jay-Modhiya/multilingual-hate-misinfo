"""
src/explainability/lime_explainer.py
─────────────────────────────────────
LIME-based explainability for MuRIL classifier.

For each prediction, LIME:
  1. Perturbs the input text (masks random words)
  2. Runs each perturbed version through MuRIL
  3. Fits a local linear model to find which words
     pushed the prediction most strongly
  4. Returns word → importance score mapping

Supports both English and Hindi/Hinglish text.
"""

import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
from lime.lime_text import LimeTextExplainer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Predictor wrapper (LIME needs a function)
# ─────────────────────────────────────────────

class MuRILPredictor:
    """
    Wraps MuRIL model into a predict_proba function
    that LIME can call repeatedly.
    """

    def __init__(self, model, tokenizer_wrapper, device, label_map: Dict):
        self.model     = model
        self.tok       = tokenizer_wrapper
        self.device    = device
        self.label_map = label_map
        self.model.eval()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Takes a list of (perturbed) texts.
        Returns probability array of shape (n_texts, n_classes).
        """
        all_probs = []

        # Process in small batches to avoid OOM
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = self.tok.tokenize_batch(batch_texts)
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                _, logits = self.model(**enc)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)


# ─────────────────────────────────────────────
#  LIME Explainer
# ─────────────────────────────────────────────

class HateSpeechLIMEExplainer:
    """
    LIME explainer for hate speech and misinfo detection.

    Usage:
        explainer = HateSpeechLIMEExplainer(model, tokenizer, device, label_map, cfg)
        result    = explainer.explain("This is a hateful tweet")
        fig       = explainer.plot(result)
    """

    def __init__(
        self,
        model,
        tokenizer_wrapper,
        device,
        label_map: Dict[int, str],
        cfg: dict,
    ):
        self.predictor = MuRILPredictor(model, tokenizer_wrapper, device, label_map)
        self.label_map = label_map
        self.cfg       = cfg

        # Class names for LIME
        class_names = [label_map[i] for i in sorted(label_map.keys())]

        self.lime = LimeTextExplainer(
            class_names=class_names,
            split_expression=r"\s+",   # works for both English and Hindi
            bow=False,                  # preserve word order context
            random_state=cfg["project"]["seed"],
        )

        self.num_features = cfg["lime"]["num_features"]
        self.num_samples  = cfg["lime"]["num_samples"]

        logger.info(
            f"LIME explainer ready | "
            f"features={self.num_features} | samples={self.num_samples}"
        )

    def explain(self, text: str, label_idx: int = 1) -> dict:
        """
        Generate LIME explanation for a single text.

        Args:
            text      : input text (English or Hindi)
            label_idx : class index to explain (default=1 → hate/fake)

        Returns dict with:
            text, predicted_label, predicted_prob,
            word_scores, explanation object
        """
        # Get model prediction first
        probs = self.predictor.predict_proba([text])[0]
        pred_idx  = int(np.argmax(probs))
        pred_prob = float(probs[pred_idx])
        pred_label = self.label_map[pred_idx]

        # Run LIME
        exp = self.lime.explain_instance(
            text,
            self.predictor.predict_proba,
            num_features=self.num_features,
            num_samples=self.num_samples,
            labels=[label_idx],
        )

        # Extract word scores
        word_scores = dict(exp.as_list(label=label_idx))

        return {
            "text":          text,
            "predicted_label": pred_label,
            "predicted_prob":  pred_prob,
            "pred_idx":       pred_idx,
            "class_probs":    {self.label_map[i]: float(probs[i])
                               for i in range(len(probs))},
            "word_scores":    word_scores,
            "explanation":    exp,
            "label_idx":      label_idx,
        }

    def plot(
        self,
        result: dict,
        title:     Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot LIME word importance as a horizontal bar chart.
        Green = supports predicted class, Red = opposes it.
        """
        word_scores = result["word_scores"]
        if not word_scores:
            logger.warning("No word scores to plot.")
            return None

        # Sort by absolute importance
        sorted_items = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        sorted_items = sorted_items[:self.num_features]

        words  = [w for w, _ in sorted_items]
        scores = [s for _, s in sorted_items]
        colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in scores]

        fig, ax = plt.subplots(figsize=(8, max(4, len(words) * 0.45)))
        bars = ax.barh(range(len(words)), scores, color=colors, edgecolor="white", height=0.7)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("LIME Importance Score", fontsize=10)

        chart_title = title or (
            f"LIME — Prediction: {result['predicted_label']} "
            f"({result['predicted_prob']*100:.1f}%)"
        )
        ax.set_title(chart_title, fontsize=11, fontweight="bold", pad=12)

        # Legend
        pos_patch = mpatches.Patch(color="#2ecc71", label="Supports prediction")
        neg_patch = mpatches.Patch(color="#e74c3c", label="Opposes prediction")
        ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)

        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"LIME plot saved → {save_path}")

        return fig

    def explain_batch(self, texts: List[str], label_idx: int = 1) -> List[dict]:
        """Explain multiple texts."""
        return [self.explain(t, label_idx) for t in texts]

    def get_top_words(self, result: dict, n: int = 5) -> Tuple[List, List]:
        """
        Returns (top_positive_words, top_negative_words).
        Positive = push toward predicted class.
        Negative = push away from predicted class.
        """
        scores = result["word_scores"]
        pos = sorted([(w, s) for w, s in scores.items() if s > 0],
                     key=lambda x: x[1], reverse=True)[:n]
        neg = sorted([(w, s) for w, s in scores.items() if s < 0],
                     key=lambda x: x[1])[:n]
        return pos, neg


# ─────────────────────────────────────────────
#  Streamlit-ready summary
# ─────────────────────────────────────────────

def format_explanation_for_ui(result: dict) -> dict:
    """
    Format LIME result into a clean dict for Streamlit display.
    """
    pos_words, neg_words = [], []
    for word, score in result["word_scores"].items():
        if score > 0:
            pos_words.append({"word": word, "score": round(score, 4)})
        else:
            neg_words.append({"word": word, "score": round(score, 4)})

    pos_words = sorted(pos_words, key=lambda x: x["score"], reverse=True)
    neg_words = sorted(neg_words, key=lambda x: x["score"])

    return {
        "prediction":   result["predicted_label"],
        "confidence":   f"{result['predicted_prob']*100:.1f}%",
        "class_probs":  {k: f"{v*100:.1f}%" for k, v in result["class_probs"].items()},
        "top_hate_words":   pos_words[:5],
        "top_safe_words":   neg_words[:5],
        "all_word_scores":  result["word_scores"],
    }


# ─────────────────────────────────────────────
#  Smoke-test (CPU, no real model needed)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("LIME explainer module loaded successfully ✓")
    print("To use: instantiate HateSpeechLIMEExplainer with a trained model.")
    print("Example:")
    print("  explainer = HateSpeechLIMEExplainer(model, tokenizer, device, label_map, cfg)")
    print("  result    = explainer.explain('This tweet contains hate speech')")
    print("  fig       = explainer.plot(result)")
