"""Gradio UI for sentiment analysis with DistilBERT."""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add parent directory to path to import data module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None


def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer."""
    global model, tokenizer, device
    
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "distilbert-imdb"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please train the model first using: python -m src.train"
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully on {device}")


def predict_sentiment(text: str) -> dict:
    """Predict sentiment for the given text.
    
    Args:
        text: Input movie review text.
    
    Returns:
        Dictionary with confidence scores for Negative and Positive classes.
    """
    if not text or not text.strip():
        return {"Negative 😞": 0.0, "Positive 😊": 0.0}
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    
    neg_prob = probs[0].item()
    pos_prob = probs[1].item()
    
    return {
        "Negative 😞": neg_prob,
        "Positive 😊": pos_prob,
    }


def create_interface():
    """Create and configure the Gradio interface."""
    
    # Load model at startup
    load_model_and_tokenizer()
    
    # Example reviews
    examples = [
        ["This movie was absolutely fantastic! The cinematography was breathtaking and the performances were outstanding."],
        ["I loved every minute of it. A masterpiece that will be remembered for years to come."],
        ["Terrible movie. Poor acting and a confusing plot that made no sense."],
        ["I couldn't even finish watching it. Complete waste of time and money."],
        ["Incredible storytelling with brilliant acting. Highly recommended!"],
        ["Disappointing and boring. The worst film I've seen this year."],
    ]
    
    # Device info for description
    device_info = "🚀 GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
    
    # Create interface
    interface = gr.Interface(
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
        **Powered by DistilBERT fine-tuned on IMDB dataset**
        
        Enter a movie review and get instant sentiment predictions!
        
        **System:** {device_info} | **Model:** DistilBERT | **Task:** Binary Sentiment Classification
        """,
        examples=examples,
        theme=gr.themes.Soft(),
        allow_flagging="never",
    )
    
    return interface


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
