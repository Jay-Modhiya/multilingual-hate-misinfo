"""
tests/test_model.py
Unit tests for evaluator, bias analyzer, drift detector.
Run: pytest tests/test_model.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import pandas as pd

from src.models.evaluator import compute_metrics, full_report, build_summary_table
from src.bias.bias_analyzer import compute_group_metrics, equalized_odds_gap
from src.monitoring.drift_detector import DriftDetector, extract_text_features


@pytest.fixture
def cfg():
    import yaml
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_preds():
    return {
        "y_true":    [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        "y_pred":    [0, 1, 0, 0, 0, 1, 1, 0, 1, 1],
        "label_map": {0: "not_hate", 1: "hate"},
    }


def test_compute_metrics(sample_preds):
    m = compute_metrics(
        sample_preds["y_true"],
        sample_preds["y_pred"],
        sample_preds["label_map"]
    )
    assert "accuracy"  in m
    assert "precision" in m
    assert "recall"    in m
    assert "f1"        in m
    assert 0 <= m["accuracy"]  <= 100
    assert 0 <= m["f1"]        <= 100


def test_perfect_predictions():
    y = [0, 1, 0, 1, 1]
    m = compute_metrics(y, y, {0: "not_hate", 1: "hate"})
    assert m["accuracy"]  == 100.0
    assert m["f1"]        == 100.0


def test_build_summary_table():
    results = {
        "MuRIL": {"accuracy": 92.0, "precision": 91.5, "recall": 92.0, "f1": 91.8}
    }
    df = build_summary_table(results)
    assert "MuRIL" in df.index
    assert "F1-Score" in df.columns


def test_group_metrics(cfg):
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "label":    np.random.randint(0, 2, n),
        "pred":     np.random.randint(0, 2, n),
        "language": np.random.choice(["en", "hi"], n),
    })
    result = compute_group_metrics(df)
    assert "en" in result.index
    assert "hi" in result.index
    assert "F1-Score" in result.columns
    assert "FPR (%)"  in result.columns


def test_equalized_odds(cfg):
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "label":    np.random.randint(0, 2, n),
        "pred":     np.random.randint(0, 2, n),
        "language": np.random.choice(["en", "hi"], n),
    })
    metrics_df = compute_group_metrics(df)
    gaps = equalized_odds_gap(metrics_df)
    assert "fpr_gap" in gaps
    assert "fnr_gap" in gaps
    assert "f1_gap"  in gaps
    assert gaps["fpr_gap"] >= 0


def test_extract_features():
    texts = ["Hello world", "नमस्ते दुनिया", "URGENT!!! CLICK NOW!!!"]
    feats = extract_text_features(texts)
    assert feats.shape == (3, 5)
    assert feats[2, 3] > feats[0, 3]


def test_drift_no_drift(cfg):
    detector = DriftDetector(cfg)
    ref  = ["normal tweet about sports"] * 100
    same = ["normal tweet about sports"] * 50
    detector.fit(ref)
    result = detector.detect(same)
    assert "drift_detected" in result
    assert result["drift_detected"] == False


def test_drift_with_drift(cfg):
    detector = DriftDetector(cfg)
    ref     = ["normal tweet"] * 200
    drifted = ["ALERT!!! URGENT!!! BREAKING!!! FAKE!!!"] * 200
    detector.fit(ref)
    result = detector.detect(drifted)
    assert result["drift_detected"] == True


def test_drift_log(cfg):
    detector = DriftDetector(cfg)
    detector.fit(["text"] * 100)
    detector.detect(["something"] * 50)
    df = detector.get_drift_summary()
    assert len(df) == 1
    assert "Drift" in df.columns
