"""Fine-tune DistilBERT on IMDB for binary sentiment classification using HuggingFace Trainer."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import load_imdb


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def get_training_args(output_dir: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )


def main() -> None:
    set_seed(42)

    model_name = "distilbert-base-uncased"
    max_length = 256

    dataset, tokenizer = load_imdb(tokenizer_name=model_name, max_length=max_length)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    output_dir = Path(__file__).resolve().parents[1] / "models" / "distilbert-imdb"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = get_training_args(output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
