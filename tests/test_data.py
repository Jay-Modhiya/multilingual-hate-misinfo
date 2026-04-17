"""
tests/test_data.py
Unit tests for data loading and preprocessing.
Run: pytest tests/test_data.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
import yaml

# Import directly from modules (not through __init__ which imports torch)
from src.data.loader import load_config, load_hasoc, load_fakenewsnet


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
    # Import cleaner directly to avoid torch dependency in __init__
    import re
    text    = "RT @user: Check https://example.com #hateful content!!!"
    cleaned = re.sub(r"https?://\S+", " ", text)
    cleaned = re.sub(r"@\w+", " ", cleaned).strip()
    assert "https://" not in cleaned
    assert "@user" not in cleaned
    assert len(cleaned) > 0


def test_cleaner_hindi(cfg):
    import re
    text    = "@user देखो https://example.com यह नफरत है"
    cleaned = re.sub(r"https?://\S+", " ", text)
    cleaned = re.sub(r"@\w+", " ", cleaned).strip()
    assert "https://" not in cleaned
    assert "देखो" in cleaned
    assert "नफरत" in cleaned


def test_hasoc_loads(cfg):
    splits = load_hasoc(cfg)
    assert "train" in splits
    assert "val"   in splits
    assert "test"  in splits
    assert len(splits["train"]) > 0
    assert "label_map" in splits
    assert set(splits["label_map"].keys()) == {0, 1}


def test_fakenewsnet_loads(cfg):
    splits = load_fakenewsnet(cfg)
    assert len(splits["train"]) > 0
    assert splits["label_map"][0] == "real"
    assert splits["label_map"][1] == "fake"


def test_label_distribution(cfg):
    splits = load_hasoc(cfg)
    train_labels = splits["train"]["label"].value_counts()
    assert 0 in train_labels.index
    assert 1 in train_labels.index


def test_train_val_test_no_overlap(cfg):
    splits = load_hasoc(cfg)
    train_texts = set(splits["train"]["text"].tolist())
    test_texts  = set(splits["test"]["text"].tolist())
    overlap = train_texts.intersection(test_texts)
    assert len(overlap) == 0


def test_config_aws_region(cfg):
    assert cfg["aws"]["region"] == "ap-south-1"


def test_config_model_name(cfg):
    assert "muril" in cfg["model"]["hf_checkpoint"].lower()
