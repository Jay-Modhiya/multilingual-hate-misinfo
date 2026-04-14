"""
src/data/preprocessor.py
─────────────────────────
Text cleaning pipeline + MuRIL tokenization.

Two stages:
  Stage 1 – CleanerPipeline  → raw text  → cleaned text
  Stage 2 – MuRILTokenizer   → cleaned text → input_ids / attention_mask

DistilBERT insight from CA-3: raw text fed directly to WordPiece was
fine. MuRIL is the same — subword tokenizer handles multilingual text
without manual preprocessing. We still clean URLs/mentions for noise
reduction, but we keep Devanagari script and Hinglish intact.
"""

import re
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Stage 1 – Text Cleaning
# ─────────────────────────────────────────────

class CleanerPipeline:
    """
    Language-aware cleaning pipeline.
    
    For English:  remove URLs, mentions, extra whitespace
    For Hindi:    remove URLs, mentions — keep Devanagari unicode
    For Hinglish: same as Hindi
    """

    URL_RE      = re.compile(r"https?://\S+|www\.\S+")
    MENTION_RE  = re.compile(r"@\w+")
    HASHTAG_RE  = re.compile(r"#(\w+)")          # keep word, remove #
    MULTI_WS_RE = re.compile(r"\s+")
    RT_RE       = re.compile(r"^RT\s+")           # retweet marker

    def __init__(self, cfg: dict):
        self.remove_urls     = cfg["preprocessing"]["remove_urls"]
        self.remove_mentions = cfg["preprocessing"]["remove_mentions"]
        self.remove_hashtags = cfg["preprocessing"]["remove_hashtags"]

    def clean(self, text: str, lang: str = "en") -> str:
        if not isinstance(text, str):
            return ""

        text = self.RT_RE.sub("", text)

        if self.remove_urls:
            text = self.URL_RE.sub(" ", text)

        if self.remove_mentions:
            text = self.MENTION_RE.sub(" ", text)

        if self.remove_hashtags:
            text = re.sub(r"#\w+", " ", text)
        else:
            # keep the word, strip #
            text = self.HASHTAG_RE.sub(r"\1", text)

        # Remove non-printable / control characters (keep Devanagari)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Collapse whitespace
        text = self.MULTI_WS_RE.sub(" ", text).strip()

        return text

    def clean_dataframe(self, df: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
        df = df.copy()
        df["text"] = df["text"].apply(lambda t: self.clean(t, lang))
        df = df[df["text"].str.len() > 5].reset_index(drop=True)
        return df


# ─────────────────────────────────────────────
#  Stage 2 – MuRIL Tokenizer wrapper
# ─────────────────────────────────────────────

class MuRILTokenizerWrapper:
    """
    Thin wrapper around HuggingFace AutoTokenizer for google/muril-base-cased.
    Handles batched tokenization and returns tensors ready for the model.
    """

    def __init__(self, cfg: dict):
        checkpoint  = cfg["model"]["hf_checkpoint"]
        self.max_len = cfg["preprocessing"]["max_length"]

        logger.info(f"Loading tokenizer: {checkpoint}")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        logger.info("Tokenizer loaded.")

    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of strings → dict of tensors.
        Returns: { input_ids, attention_mask, token_type_ids }
        """
        encoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return encoding

    def tokenize_single(self, text: str) -> Dict[str, torch.Tensor]:
        return self.tokenize_batch([text])

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


# ─────────────────────────────────────────────
#  PyTorch Dataset wrapper
# ─────────────────────────────────────────────

class HateMisinfoDataset(Dataset):
    """
    Generic PyTorch Dataset for hate speech / misinformation tasks.

    Accepts a cleaned DataFrame with columns: text, label
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_wrapper: MuRILTokenizerWrapper,
        label_col: str = "label",
    ):
        self.texts  = df["text"].tolist()
        self.labels = df[label_col].tolist()
        self.tok    = tokenizer_wrapper

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tok.tokenize_single(self.texts[idx])
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─────────────────────────────────────────────
#  Full preprocessing pipeline (convenience)
# ─────────────────────────────────────────────

def preprocess_splits(
    splits: dict,
    cfg: dict,
    lang: str = "en",
    tokenizer: Optional[MuRILTokenizerWrapper] = None,
) -> Tuple[HateMisinfoDataset, HateMisinfoDataset, HateMisinfoDataset]:
    """
    Given a splits dict { train, val, test, label_map }, run:
      1. Text cleaning
      2. Wrap in PyTorch datasets

    Returns (train_ds, val_ds, test_ds)
    """
    cleaner = CleanerPipeline(cfg)
    if tokenizer is None:
        tokenizer = MuRILTokenizerWrapper(cfg)

    train_clean = cleaner.clean_dataframe(splits["train"], lang)
    val_clean   = cleaner.clean_dataframe(splits["val"],   lang)
    test_clean  = cleaner.clean_dataframe(splits["test"],  lang)

    train_ds = HateMisinfoDataset(train_clean, tokenizer)
    val_ds   = HateMisinfoDataset(val_clean,   tokenizer)
    test_ds  = HateMisinfoDataset(test_clean,  tokenizer)

    logger.info(
        f"Datasets ready → train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}"
    )
    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────────
#  Smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    cleaner = CleanerPipeline(cfg)

    samples = [
        ("en", "RT @user: Check out https://example.com this #hateful content!!!"),
        ("hi", "@user देखो https://example.com यह #नफरत वाली सामग्री है"),
        ("hi", "आज बहुत अच्छा मौसम है #खुशी"),
    ]

    print("\n── Cleaner output ─────────────────────────")
    for lang, text in samples:
        cleaned = cleaner.clean(text, lang)
        print(f"[{lang}] {text[:60]}")
        print(f"      → {cleaned}\n")
