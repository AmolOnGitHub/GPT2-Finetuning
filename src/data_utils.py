import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def build_tokenizer(model_name: str):
    """
    Build and return a tokenizer for the specified model.
    If the tokenizer does not have a pad token, set it to the eos token.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    return tokenizer


def build_datasets(tokenizer, max_length: int = 256, val_size: int = 2000, subsets=None):
    """
    Load and preprocess the AG News dataset.
    Tokenize the text and split into train, validation, and test sets.
    Optionally subset the datasets to the specified number of samples.
    Return the datasets and a data collator for padding.
    """
    subsets = subsets or {}
    ds = load_dataset("ag_news")

    def preprocess(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"] = examples["label"]
        return enc

    tokenized = ds.map(preprocess, batched=True, remove_columns=["text"])

    split = tokenized["train"].train_test_split(test_size=val_size, seed=42)
    train_ds, val_ds, test_ds = split["train"], split["test"], tokenized["test"]

    if subsets.get("train"):
        train_ds = train_ds.select(range(subsets["train"]))
    if subsets.get("val"):
        val_ds = val_ds.select(range(subsets["val"]))
    if subsets.get("test"):
        test_ds = test_ds.select(range(subsets["test"]))

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return train_ds, val_ds, test_ds, collator