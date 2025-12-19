"""Data loading utilities for the IMDB sentiment dataset."""
from __future__ import annotations

import re
from typing import Tuple

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer


def _clean_text(text: str) -> str:
    """Simple preprocessing: strip, lowercase, and remove HTML breaks."""
    text = text.replace("<br />", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_imdb(tokenizer_name: str = "distilbert-base-uncased", max_length: int = 256) -> Tuple[DatasetDict, AutoTokenizer]:
    """Load, preprocess, and tokenize IMDB reviews, returning torch-ready datasets.

    Args:
        tokenizer_name: HuggingFace model checkpoint for the tokenizer.
        max_length: Maximum sequence length for padding/truncation.

    Returns:
        A tuple of (tokenized_dataset, tokenizer) where tokenized_dataset is a DatasetDict
        with train/test splits formatted for PyTorch.
    """
    raw = load_dataset("stanfordnlp/imdb")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def preprocess(batch):
        return {"text": [_clean_text(t) for t in batch["text"]]}

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    cleaned = raw.map(preprocess, batched=True, desc="Cleaning text")
    tokenized = cleaned.map(tokenize, batched=True, remove_columns=["text"], desc="Tokenizing")
    tokenized = tokenized.rename_column("label", "labels")

    # Set torch format
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized, tokenizer
