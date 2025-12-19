"""Evaluate the fine-tuned sentiment model on the test split and generate reports."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_imdb


def evaluate_model(model_path: Path, results_path: Path) -> None:
    """Load fine-tuned model, evaluate on test set, and save plots.

    Args:
        model_path: Path to the saved model directory.
        results_path: Path to save evaluation results and plots.
    """
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load test dataset
    print("Loading test dataset...")
    dataset, _ = load_imdb(tokenizer_name=str(model_path), max_length=256)
    test_dataset = dataset["test"]
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []

    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of positive class

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Print classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        all_labels, all_preds, target_names=["Negative", "Positive"], digits=4
    )
    print(report)
    
    # Save classification report
    with open(results_path / "classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(f"              Predicted")
    print(f"              Neg    Pos")
    print(f"Actual Neg  {cm[0][0]:6d} {cm[0][1]:6d}")
    print(f"       Pos  {cm[1][0]:6d} {cm[1][1]:6d}")
    print("=" * 60 + "\n")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(results_path / "confusion_matrix.png", dpi=300)
    print(f"✓ Saved confusion matrix plot to {results_path / 'confusion_matrix.png'}")
    plt.close()

    # ROC curve
    roc_auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_path / "roc_curve.png", dpi=300)
    print(f"✓ Saved ROC curve plot to {results_path / 'roc_curve.png'}")
    plt.close()

    # Distribution of predicted probabilities
    plt.figure(figsize=(10, 5))
    plt.hist(
        all_probs[all_labels == 0],
        bins=50,
        alpha=0.6,
        label="Negative (True)",
        color="red",
    )
    plt.hist(
        all_probs[all_labels == 1],
        bins=50,
        alpha=0.6,
        label="Positive (True)",
        color="green",
    )
    plt.xlabel("Predicted Probability (Positive Class)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Predicted Probabilities", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_path / "probability_distribution.png", dpi=300)
    print(f"✓ Saved probability distribution plot to {results_path / 'probability_distribution.png'}")
    plt.close()

    print(f"\n✓ Evaluation complete! Results saved to {results_path}/")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "distilbert-imdb"
    results_path = project_root / "results"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using src/train.py")
        return

    evaluate_model(model_path, results_path)


if __name__ == "__main__":
    main()
