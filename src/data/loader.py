"""
src/data/loader.py
──────────────────
Unified dataset loader for:
  1. Davidson  – English hate speech  (HuggingFace / CSV)
  2. HASOC     – Hindi/Hinglish hate speech (CSV)
  3. FakeNewsNet – English misinformation (CSV / GitHub)

Each loader returns a dict:
  {
    "train": pd.DataFrame,
    "val":   pd.DataFrame,
    "test":  pd.DataFrame,
    "label_map": {int: str}
  }
"""

import os
import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ─────────────────────────────────────────────
#  Config helper
# ─────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
#  Shared split utility
# ─────────────────────────────────────────────

def _split(df: pd.DataFrame, cfg: dict, label_col: str = "label") -> dict:
    """Stratified train / val / test split."""
    test_size = cfg["preprocessing"]["test_size"]
    val_size  = cfg["preprocessing"]["val_size"]
    seed      = cfg["project"]["seed"]

    df = df.dropna(subset=["text", label_col]).reset_index(drop=True)

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=seed
    )
    # val_size is relative to original → adjust
    adjusted_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=adjusted_val, stratify=train_val[label_col], random_state=seed
    )

    logger.info(f"Split → train:{len(train)}  val:{len(val)}  test:{len(test)}")
    return {"train": train.reset_index(drop=True),
            "val":   val.reset_index(drop=True),
            "test":  test.reset_index(drop=True)}


# ─────────────────────────────────────────────
#  1. Davidson Dataset
# ─────────────────────────────────────────────

def load_davidson(cfg: dict) -> dict:
    """
    Load Davidson hate speech dataset.
    Tries HuggingFace `tweet_eval/hate` first; falls back to local CSV.

    CSV format expected:
        tweet, class   (0=hate, 1=offensive, 2=neither)
    We binarise: 0 → hate(1), 1,2 → not_hate(0)
    """
    logger.info("Loading Davidson dataset …")
    label_map = {0: "not_hate", 1: "hate"}

    try:
        from datasets import load_dataset
        raw = load_dataset("tweet_eval", "hate", trust_remote_code=True)
        frames = []
        for split_name, split_data in raw.items():
            tmp = pd.DataFrame({"text": split_data["text"], "label": split_data["label"]})
            frames.append(tmp)
        df = pd.concat(frames, ignore_index=True)
        # tweet_eval hate: 0=not_hate, 1=hate  → already binary
        logger.info(f"Davidson loaded from HuggingFace  |  rows: {len(df)}")

    except Exception as e:
        logger.warning(f"HuggingFace load failed ({e}). Falling back to CSV.")
        csv_path = cfg["datasets"]["davidson"]["local_fallback"]
        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"Davidson CSV not found at {csv_path}. "
                "Download from: https://github.com/t-davidson/hate-speech-and-offensive-language"
            )
        df = pd.read_csv(csv_path)
        # Original Davidson: class 0=hate, 1=offensive, 2=neither
        df = df.rename(columns={"tweet": "text", "class": "label"})
        df["label"] = df["label"].apply(lambda x: 1 if x == 0 else 0)

    df = df[["text", "label"]].copy()
    df["language"] = "en"
    df["dataset"]  = "davidson"

    result = _split(df, cfg)
    result["label_map"] = label_map
    return result


# ─────────────────────────────────────────────
#  2. HASOC Dataset  (Hindi / Hinglish)
# ─────────────────────────────────────────────

def load_hasoc(cfg: dict) -> dict:
    """
    Load HASOC 2019 Hindi hate speech dataset from TSV files.

    Expected files (place in data/raw/):
      - hasoc_train.tsv  → columns: text_id, text, task_1, task_2, task_3
      - hasoc_test.tsv   → same format (gold labels)

    Labels: NOT=0 (not hate), HOF=1 (hate or offensive)
    """
    logger.info("Loading HASOC (Hindi/Hinglish) dataset from TSV files …")
    label_map = {0: "not_hate", 1: "hate_offensive"}

    train_path = "data/raw/hasoc_train.tsv"
    test_path  = "data/raw/hasoc_test.tsv"

    if Path(train_path).exists():
        # ── Load real TSV files ──
        train_df = pd.read_csv(train_path, sep="\t")
        test_df  = pd.read_csv(test_path,  sep="\t") if Path(test_path).exists() else None

        # Combine train + test into one pool → re-split stratified
        frames = [train_df]
        if test_df is not None:
            frames.append(test_df)
        df = pd.concat(frames, ignore_index=True)

        # Rename and encode
        df = df.rename(columns={"text": "text"})
        df["label"] = df["task_1"].apply(lambda x: 0 if str(x).strip() == "NOT" else 1)
        df = df[["text", "label"]].dropna()

        logger.info(
            f"HASOC TSV loaded  |  total rows: {len(df)}  |  "
            f"HOF: {(df['label']==1).sum()}  NOT: {(df['label']==0).sum()}"
        )
    else:
        logger.warning(
            "HASOC TSV not found. Using placeholder.\n"
            "→ Place hasoc_train.tsv and hasoc_test.tsv in data/raw/"
        )
        df = _create_hasoc_placeholder()

    df["language"] = "hi"
    df["dataset"]  = "hasoc"

    result = _split(df, cfg)
    result["label_map"] = label_map
    return result


def _create_hasoc_placeholder() -> pd.DataFrame:
    """Tiny placeholder so the pipeline runs end-to-end without real data."""
    hindi_not_hate = [
        "आज मौसम बहुत अच्छा है।",
        "मुझे क्रिकेट देखना पसंद है।",
        "यह फिल्म बहुत अच्छी है।",
        "भारत एक महान देश है।",
        "मेरे दोस्त बहुत अच्छे हैं।",
    ] * 60
    hindi_hate = [
        "तुम लोग बेकार हो।",
        "यह सब झूठे लोग हैं।",
        "इनको यहाँ से निकालो।",
        "नफरत है मुझे इनसे।",
        "ये देश के दुश्मन हैं।",
    ] * 60
    texts  = hindi_not_hate + hindi_hate
    labels = [0] * len(hindi_not_hate) + [1] * len(hindi_hate)
    df = pd.DataFrame({"text": texts, "label": labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ─────────────────────────────────────────────
#  3. FakeNewsNet Dataset
# ─────────────────────────────────────────────

def load_fakenewsnet(cfg: dict) -> dict:
    """
    Load FakeNewsNet misinformation dataset.
    Expected CSV: title (text), label (0=real, 1=fake)

    Download helper: see notebooks/kaggle_training.ipynb
    Place at: data/raw/fakenewsnet.csv
    """
    logger.info("Loading FakeNewsNet dataset …")
    label_map = {0: "real", 1: "fake"}

    csv_path = cfg["datasets"]["fakenewsnet"]["local_fallback"]

    try:
        from datasets import load_dataset
        raw = load_dataset("GonzaloA/fake_news", trust_remote_code=True)
        frames = []
        for split_name, split_data in raw.items():
            tmp = pd.DataFrame({
                "text":  split_data["text"],
                "label": split_data["label"]
            })
            frames.append(tmp)
        df = pd.concat(frames, ignore_index=True)
        logger.info(f"FakeNewsNet loaded from HuggingFace  |  rows: {len(df)}")

    except Exception as e:
        logger.warning(f"HuggingFace FakeNewsNet load failed ({e}). Trying local CSV …")
        if not Path(csv_path).exists():
            logger.warning(
                "FakeNewsNet CSV not found. Creating a small synthetic placeholder.\n"
                "→ Please download: https://github.com/KaiDMML/FakeNewsNet\n"
                "  and place at: data/raw/fakenewsnet.csv"
            )
            df = _create_fakenewsnet_placeholder()
        else:
            df = pd.read_csv(csv_path)
            if "title" in df.columns and "text" not in df.columns:
                df["text"] = df["title"]
            if "label" not in df.columns and "Label" in df.columns:
                df["label"] = df["Label"].apply(lambda x: 1 if str(x).lower() == "fake" else 0)

    df = df[["text", "label"]].copy()
    df = df[df["text"].str.strip().str.len() > 10]   # drop near-empty rows
    df["language"] = "en"
    df["dataset"]  = "fakenewsnet"

    result = _split(df, cfg)
    result["label_map"] = label_map
    return result


def _create_fakenewsnet_placeholder() -> pd.DataFrame:
    real_news = [
        "Scientists discover new species of deep-sea fish in Pacific Ocean.",
        "Government announces budget allocations for infrastructure projects.",
        "Central bank holds interest rates steady amid inflation concerns.",
        "Olympic committee selects host city for 2032 games.",
        "University researchers publish findings on climate change impact.",
    ] * 80
    fake_news = [
        "Aliens land in capital city, government covers up evidence.",
        "Secret cure for cancer suppressed by pharmaceutical companies.",
        "Microchips found in vaccines, experts claim.",
        "Famous celebrity announces they are actually a robot.",
        "Moon landing was filmed in a Hollywood studio, insider reveals.",
    ] * 80
    texts  = real_news + fake_news
    labels = [0] * len(real_news) + [1] * len(fake_news)
    df = pd.DataFrame({"text": texts, "label": labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ─────────────────────────────────────────────
#  Unified loader
# ─────────────────────────────────────────────

def load_all_datasets(cfg: dict) -> dict:
    """
    Load all three datasets and return as a combined dict.

    Returns:
        {
          "davidson":     { train, val, test, label_map },
          "hasoc":        { train, val, test, label_map },
          "fakenewsnet":  { train, val, test, label_map },
        }
    """
    return {
        "davidson":    load_davidson(cfg),
        "hasoc":       load_hasoc(cfg),
        "fakenewsnet": load_fakenewsnet(cfg),
    }


# ─────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    datasets = load_all_datasets(cfg)

    for name, splits in datasets.items():
        print(f"\n{'─'*40}")
        print(f"Dataset : {name}")
        print(f"Labels  : {splits['label_map']}")
        for split_name in ["train", "val", "test"]:
            df = splits[split_name]
            print(f"  {split_name:5s} → {len(df):5d} rows | "
                  f"label dist: {df['label'].value_counts().to_dict()}")
        print(f"  Sample text: {splits['train']['text'].iloc[0][:80]}")
