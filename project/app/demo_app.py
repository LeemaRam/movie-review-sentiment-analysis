"""Gradio demo app using pre-trained DistilBERT (works without training first)."""
from __future__ import annotations

import gradio as gr
import torch
from transformers import pipeline


def load_model():
    """Load pre-trained sentiment model."""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("sentiment-analysis", model=model_name, device=device)
    print(f"✓ Model loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    return classifier


# Load model at startup
classifier = load_model()


def predict_sentiment(text: str) -> dict:
    """Predict sentiment for the given text.
    
    Args:
        text: Input movie review text.
    
    Returns:
        Dictionary with confidence scores for Negative and Positive classes.
    """
    if not text or not text.strip():
        return {"Negative 😞": 0.0, "Positive 😊": 0.0}
    
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    
    if label == "POSITIVE":
        return {"Negative 😞": 1 - score, "Positive 😊": score}
    else:
        return {"Negative 😞": score, "Positive 😊": 1 - score}


# Example reviews
examples = [
    ["This movie was absolutely fantastic! The cinematography was breathtaking and the performances were outstanding."],
    ["I loved every minute of it. A masterpiece that will be remembered for years to come."],
    ["Terrible movie. Poor acting and a confusing plot that made no sense."],
    ["I couldn't even finish watching it. Complete waste of time and money."],
    ["Incredible storytelling with brilliant acting. Highly recommended!"],
    ["Disappointing and boring. The worst film I've seen this year."],
]

# Device info
device_info = "🚀 GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"

# Create interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter a movie review here...",
        label="📝 Movie Review",
    ),
    outputs=gr.Label(
        num_top_classes=2,
        label="📊 Sentiment Prediction",
    ),
    title="🎭 Movie Review Sentiment Analysis",
    description=f"""
    **Powered by DistilBERT (Pre-trained SST-2 Model)**
    
    Enter a movie review and get instant sentiment predictions!
    
    **System:** {device_info} | **Model:** DistilBERT-SST2 | **Task:** Binary Sentiment Classification
    
    _Note: This uses a pre-trained model. For IMDB-specific fine-tuning, run `python -m src.train` first._
    """,
    examples=examples,
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
