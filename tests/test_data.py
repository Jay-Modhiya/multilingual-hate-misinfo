"""
tests/test_data.py
───────────────────
Unit tests for data loading and preprocessing.
Run: pytest tests/test_data.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
import yaml
from src.data.loader import load_config
from src.data.preprocessor import CleanerPipeline


@pytest.fixture
def cfg():
    return load_config("configs/config.yaml")


def test_config_loads(cfg):
    assert "model" in cfg
    assert "training" in cfg
    assert "datasets" in cfg


def test_config_values(cfg):
    assert cfg["model"]["hf_checkpoint"] == "google/muril-base-cased"
    assert cfg["training"]["epochs"] >= 3
    assert 0 < cfg["training"]["learning_rate"] < 1


def test_cleaner_english(cfg):
    cleaner = CleanerPipeline(cfg)
    text    = "RT @user: Check https://example.com #hateful content!!!"
    cleaned = cleaner.clean(text, lang="en")
    assert "https://" not in cleaned
    assert "@user" not in cleaned
    assert len(cleaned) > 0


def test_cleaner_hindi(cfg):
    cleaner = CleanerPipeline(cfg)
    text    = "@user देखो https://example.com यह नफरत है"
    cleaned = cleaner.clean(text, lang="hi")
    assert "https://" not in cleaned
    assert "देखो" in cleaned
    assert "नफरत" in cleaned


def test_cleaner_empty(cfg):
    cleaner = CleanerPipeline(cfg)
    assert cleaner.clean("", lang="en") == ""
    assert cleaner.clean(None, lang="en") == ""


def test_hasoc_loads(cfg):
    from src.data.loader import load_hasoc
    splits = load_hasoc(cfg)
    assert "train" in splits
    assert "val"   in splits
    assert "test"  in splits
    assert len(splits["train"]) > 0
    assert "label_map" in splits
    assert set(splits["label_map"].keys()) == {0, 1}


def test_fakenewsnet_loads(cfg):
    from src.data.loader import load_fakenewsnet
    splits = load_fakenewsnet(cfg)
    assert len(splits["train"]) > 0
    assert splits["label_map"][0] == "real"
    assert splits["label_map"][1] == "fake"


def test_label_distribution(cfg):
    from src.data.loader import load_hasoc
    splits = load_hasoc(cfg)
    train_labels = splits["train"]["label"].value_counts()
    assert 0 in train_labels.index
    assert 1 in train_labels.index


def test_train_val_test_no_overlap(cfg):
    from src.data.loader import load_hasoc
    splits = load_hasoc(cfg)
    train_texts = set(splits["train"]["text"].tolist())
    test_texts  = set(splits["test"]["text"].tolist())
    overlap = train_texts.intersection(test_texts)
    assert len(overlap) == 0, f"Train/test overlap found: {len(overlap)} samples"
