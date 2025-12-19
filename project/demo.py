"""Quick demo script - Tests the pipeline without full training."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def quick_demo():
    """Run a quick demo using the pre-trained DistilBERT (not fine-tuned)."""
    
    print("🎭 Movie Sentiment Analysis - Quick Demo\n")
    print("Loading pre-trained DistilBERT model (not fine-tuned on IMDB)...")
    
    # Use a pre-trained sentiment model from HuggingFace
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    classifier = pipeline("sentiment-analysis", model=model_name, device=0 if torch.cuda.is_available() else -1)
    
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"✓ Model loaded on {device}\n")
    
    # Test examples
    examples = [
        "This movie was absolutely fantastic! The cinematography was breathtaking.",
        "Terrible movie. Poor acting and a confusing plot.",
        "It was okay, not great but not terrible either.",
        "I loved every minute of it. A masterpiece!",
        "Complete waste of time and money. Very disappointing.",
    ]
    
    print("=" * 70)
    print("Testing sentiment predictions:")
    print("=" * 70)
    
    for text in examples:
        result = classifier(text)[0]
        label = result['label']
        score = result['score']
        
        emoji = "😊" if label == "POSITIVE" else "😞"
        print(f"\n📝 Review: {text[:60]}...")
        print(f"   {emoji} Sentiment: {label} (confidence: {score:.2%})")
    
    print("\n" + "=" * 70)
    print("\n✅ Demo completed successfully!")
    print("\nℹ️  This uses a pre-trained model. To get better results on IMDB reviews,")
    print("   train the custom model with: python -m src.train")


if __name__ == "__main__":
    quick_demo()
