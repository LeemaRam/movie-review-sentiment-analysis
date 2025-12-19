"""Streamlit UI for sentiment analysis with DistilBERT."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add parent directory to path to import data module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@st.cache_resource
def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer (cached)."""
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "distilbert-imdb"
    
    if not model_path.exists():
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return None, None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    return model, tokenizer, device


def predict_sentiment(text: str, model, tokenizer, device) -> tuple[str, float, float]:
    """Predict sentiment for the given text.
    
    Returns:
        Tuple of (sentiment_label, positive_prob, negative_prob)
    """
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
    sentiment = "Positive 😊" if pos_prob > neg_prob else "Negative 😞"
    
    return sentiment, pos_prob, neg_prob


def main():
    st.set_page_config(
        page_title="Sentiment Analysis",
        page_icon="🎭",
        layout="wide",
    )
    
    st.title("🎭 Movie Review Sentiment Analysis")
    st.markdown("*Powered by DistilBERT fine-tuned on IMDB dataset*")
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer()
    
    if model is None:
        st.stop()
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ System Info")
        device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        device_emoji = "🚀" if torch.cuda.is_available() else "💻"
        st.info(f"**Device:** {device_emoji} {device_name}")
        
        if torch.cuda.is_available():
            st.success(f"**GPU:** {torch.cuda.get_device_name(0)}")
        
        st.markdown("---")
        st.header("📊 Model Info")
        st.write("**Model:** DistilBERT")
        st.write("**Task:** Binary Sentiment Classification")
        st.write("**Classes:** Positive / Negative")
        
        st.markdown("---")
        st.header("💡 Tips")
        st.write("• Enter a movie review")
        st.write("• Try sample examples")
        st.write("• Max 256 tokens")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("✍️ Enter Your Review")
        user_text = st.text_area(
            "Type or paste a movie review:",
            height=150,
            placeholder="e.g., This movie was absolutely amazing! The acting was superb and the plot kept me engaged throughout...",
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if predict_button and user_text.strip():
            with st.spinner("Analyzing sentiment..."):
                sentiment, pos_prob, neg_prob = predict_sentiment(
                    user_text, model, tokenizer, device
                )
            
            st.markdown("---")
            st.subheader("📈 Prediction Results")
            
            # Display sentiment with color
            if "Positive" in sentiment:
                st.success(f"### {sentiment}")
            else:
                st.error(f"### {sentiment}")
            
            # Probability bars
            st.markdown("**Confidence Scores:**")
            st.progress(pos_prob, text=f"Positive: {pos_prob:.2%}")
            st.progress(neg_prob, text=f"Negative: {neg_prob:.2%}")
            
            # Additional metrics
            confidence = max(pos_prob, neg_prob)
            st.metric("Confidence", f"{confidence:.2%}")
        
        elif predict_button:
            st.warning("⚠️ Please enter some text first!")
    
    with col2:
        st.header("📝 Sample Examples")
        st.markdown("Click to try:")
        
        positive_examples = [
            "This movie was absolutely fantastic! The cinematography was breathtaking and the performances were outstanding.",
            "I loved every minute of it. A masterpiece that will be remembered for years to come.",
            "Incredible storytelling with brilliant acting. Highly recommended!",
        ]
        
        negative_examples = [
            "Terrible movie. Poor acting and a confusing plot that made no sense.",
            "I couldn't even finish watching it. Complete waste of time and money.",
            "Disappointing and boring. The worst film I've seen this year.",
        ]
        
        st.markdown("**😊 Positive Reviews:**")
        for i, example in enumerate(positive_examples, 1):
            if st.button(f"Example {i}", key=f"pos_{i}", use_container_width=True):
                st.session_state.selected_text = example
                st.rerun()
        
        st.markdown("**😞 Negative Reviews:**")
        for i, example in enumerate(negative_examples, 1):
            if st.button(f"Example {i}", key=f"neg_{i}", use_container_width=True):
                st.session_state.selected_text = example
                st.rerun()
    
    # Handle selected example
    if "selected_text" in st.session_state:
        st.session_state.pop("selected_text")
    
    st.markdown("---")
    st.caption("Built with Streamlit • DistilBERT • HuggingFace Transformers")


if __name__ == "__main__":
    main()
