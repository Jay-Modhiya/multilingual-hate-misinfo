"""
src/models/batch_predictor.py
──────────────────────────────
Batch Prediction Strategy for MuRIL models.

Deployment Strategy: BATCH (chosen over real-time/streaming because)
  - Content moderation works best on batches of posts
  - More efficient GPU utilization
  - Can process thousands of posts per minute
  - Suitable for scheduled moderation pipelines

Usage:
    predictor = BatchPredictor(cfg)
    results = predictor.predict_batch(texts, task='hate')
    predictor.save_results(results, 'outputs/batch_results.csv')
"""

import os
import logging
import time
import pandas as pd
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


class BatchPredictor:
    """
    Batch prediction engine for hate speech and misinformation detection.

    Deployment Strategy: BATCH
    ─────────────────────────
    Why Batch over Real-time?
      1. Content moderation platforms process thousands of posts at once
      2. Batch processing is 10x more efficient than one-by-one inference
      3. Allows scheduling (e.g., run every hour on new posts)
      4. Better memory management for large transformer models

    Why Batch over Streaming?
      1. No need for complex message queues (Kafka/Kinesis)
      2. Simpler deployment on t2.micro EC2
      3. Sufficient latency for content moderation use case
    """

    LABEL_MAPS = {
        "hate":   {0: "not_hate",  1: "hate"},
        "misinfo": {0: "real",     1: "fake"},
    }

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizer = None
        logger.info(f"BatchPredictor initialized | device={self.device}")

    def _load_tokenizer(self):
        if self.tokenizer is None:
            from src.data.preprocessor import MuRILTokenizerWrapper
            self.tokenizer = MuRILTokenizerWrapper(self.cfg)
            logger.info("Tokenizer loaded ✓")

    def _load_model(self, task: str):
        if task not in self.models:
            from src.models.muril_classifier import load_model
            model_path = os.path.join(
                self.cfg["paths"]["model_dir"],
                f"muril_{task}_best.pt"
            )
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    f"Train the model first using: python src/training/trainer.py --task {task}"
                )
            self.models[task] = load_model(self.cfg, model_path, task=task)
            self.models[task] = self.models[task].to(self.device)
            logger.info(f"{task.upper()} model loaded ✓")

    def predict_batch(
        self,
        texts:      List[str],
        task:       str = "hate",
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Run batch inference on a list of texts.

        Args:
            texts      : list of input texts (English/Hindi/Hinglish)
            task       : "hate" or "misinfo"
            batch_size : number of texts per inference batch

        Returns:
            DataFrame with columns:
                text, predicted_label, confidence, processing_time_ms
        """
        if task not in self.LABEL_MAPS:
            raise ValueError(f"task must be 'hate' or 'misinfo', got '{task}'")

        self._load_tokenizer()
        self._load_model(task)

        label_map = self.LABEL_MAPS[task]
        model     = self.models[task]
        model.eval()

        results      = []
        total_start  = time.time()
        n_batches    = (len(texts) + batch_size - 1) // batch_size

        logger.info(
            f"Starting batch prediction | task={task} | "
            f"texts={len(texts)} | batch_size={batch_size} | "
            f"n_batches={n_batches}"
        )

        for batch_idx in range(n_batches):
            start  = batch_idx * batch_size
            end    = min(start + batch_size, len(texts))
            batch  = texts[start:end]

            t0  = time.time()
            enc = self.tokenizer.tokenize_batch(batch)
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                _, logits = model(**enc)
                probs     = torch.softmax(logits, dim=-1).cpu().numpy()
                preds     = np.argmax(probs, axis=1)

            elapsed_ms = (time.time() - t0) * 1000

            for i, (text, pred, prob_row) in enumerate(zip(batch, preds, probs)):
                results.append({
                    "text":              text[:100] + "..." if len(text) > 100 else text,
                    "predicted_label":   label_map[int(pred)],
                    "confidence":        round(float(prob_row[pred]) * 100, 2),
                    "prob_class_0":      round(float(prob_row[0]) * 100, 2),
                    "prob_class_1":      round(float(prob_row[1]) * 100, 2),
                    "processing_time_ms": round(elapsed_ms / len(batch), 2),
                })

            logger.info(
                f"Batch {batch_idx+1}/{n_batches} done | "
                f"texts={len(batch)} | time={elapsed_ms:.1f}ms"
            )

        total_time = time.time() - total_start
        df = pd.DataFrame(results)

        logger.info(
            f"\nBatch prediction complete!"
            f"\n  Total texts     : {len(texts)}"
            f"\n  Total time      : {total_time:.2f}s"
            f"\n  Avg per text    : {total_time/len(texts)*1000:.1f}ms"
            f"\n  Throughput      : {len(texts)/total_time:.0f} texts/sec"
        )

        return df

    def save_results(
        self,
        df:        pd.DataFrame,
        save_path: str,
        task:      str = "hate",
    ) -> str:
        """Save batch prediction results to CSV."""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        # Add metadata
        df["task"]      = task
        df["model"]     = self.cfg["model"]["hf_checkpoint"]
        df["timestamp"] = datetime.now().isoformat()

        df.to_csv(save_path, index=False)
        logger.info(f"Results saved → {save_path}")
        return save_path

    def get_summary(self, df: pd.DataFrame, task: str = "hate") -> dict:
        """Generate summary statistics for a batch prediction run."""
        label_map  = self.LABEL_MAPS[task]
        total      = len(df)
        flagged    = (df["predicted_label"] != label_map[0]).sum()
        avg_conf   = df["confidence"].mean()
        high_conf  = (df["confidence"] >= 90).sum()

        return {
            "total_texts":        total,
            "flagged":            int(flagged),
            "flagged_pct":        round(flagged / total * 100, 1),
            "clean":              int(total - flagged),
            "avg_confidence":     round(avg_conf, 1),
            "high_confidence":    int(high_conf),
            "avg_time_ms":        round(df["processing_time_ms"].mean(), 1),
        }


# ─────────────────────────────────────────────
#  Demo batch prediction script
# ─────────────────────────────────────────────

def run_demo(cfg: dict):
    """
    Demonstrate batch prediction on sample texts.
    This is what Sir wants to see — feed a small batch
    and show predictions.
    """
    predictor = BatchPredictor(cfg)

    # ── Sample batch for HATE detection ──
    hate_texts = [
        # English hate
        "You people don't belong here, go back to your country.",
        "I hate all people from that community.",
        # English not hate
        "I love watching cricket matches with my friends.",
        "The weather is beautiful today in Mumbai.",
        # Hindi hate
        "तुम जैसे लोगों को यहाँ नहीं रहना चाहिए।",
        "इन लोगों से नफरत है मुझे।",
        # Hindi not hate
        "आज मौसम बहुत अच्छा है।",
        "मुझे क्रिकेट देखना पसंद है।",
        # Hinglish
        "Yeh log bahut bure hain, inhe bahar nikalo.",
        "Aaj match dekhne ka plan hai, bahut maza aayega.",
    ]

    # ── Sample batch for MISINFO detection ──
    misinfo_texts = [
        # Fake news
        "Scientists confirm 5G towers are spreading the virus to control population.",
        "Secret cure for cancer suppressed by pharmaceutical companies.",
        "Microchips found in COVID vaccines, insider reveals.",
        # Real news
        "Government announces new infrastructure budget for rural development.",
        "Scientists discover new species of deep-sea fish in Pacific Ocean.",
        "Olympic committee announces host city for 2032 games.",
    ]

    print("\n" + "="*60)
    print("BATCH PREDICTION DEMO")
    print("Deployment Strategy: BATCH")
    print("Model: google/muril-base-cased")
    print("="*60)

    # ── Hate detection batch ──
    print("\n── HATE SPEECH DETECTION BATCH ──")
    hate_results = predictor.predict_batch(hate_texts, task="hate", batch_size=4)
    print(hate_results[["text", "predicted_label", "confidence"]].to_string(index=False))

    hate_summary = predictor.get_summary(hate_results, task="hate")
    print(f"\nSummary: {hate_summary['flagged']}/{hate_summary['total_texts']} flagged as hate ({hate_summary['flagged_pct']}%)")
    print(f"Avg confidence: {hate_summary['avg_confidence']}%")
    print(f"Avg processing time: {hate_summary['avg_time_ms']}ms per text")

    predictor.save_results(hate_results, "outputs/batch_hate_results.csv", task="hate")

    # ── Misinfo detection batch ──
    print("\n── MISINFORMATION DETECTION BATCH ──")
    misinfo_results = predictor.predict_batch(misinfo_texts, task="misinfo", batch_size=4)
    print(misinfo_results[["text", "predicted_label", "confidence"]].to_string(index=False))

    misinfo_summary = predictor.get_summary(misinfo_results, task="misinfo")
    print(f"\nSummary: {misinfo_summary['flagged']}/{misinfo_summary['total_texts']} flagged as fake ({misinfo_summary['flagged_pct']}%)")

    predictor.save_results(misinfo_results, "outputs/batch_misinfo_results.csv", task="misinfo")

    print("\n" + "="*60)
    print("Batch prediction demo complete!")
    print("Results saved to outputs/")
    print("="*60)


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_demo(cfg)
