"""
Movie Review Sentiment Analysis - HuggingFace Spaces App
Using DistilBERT fine-tuned on SST-2 dataset
"""

import gradio as gr
from transformers import pipeline
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 512

# Initialize the sentiment analysis pipeline
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1  # Use CPU (-1), change to 0 for GPU
    )
    logger.info(f"Successfully loaded model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise


def analyze_sentiment(text: str) -> Tuple[str, float]:
    """
    Analyze the sentiment of a movie review.
    
    Args:
        text (str): The movie review text to analyze
        
    Returns:
        Tuple[str, float]: Sentiment label with emoji and confidence score as percentage
    """
    try:
        # Input validation
        if not text or not text.strip():
            return "⚠️ Please enter a movie review", 0.0
        
        # Truncate text if too long
        if len(text) > MAX_LENGTH:
            text = text[:MAX_LENGTH]
            logger.warning(f"Input truncated to {MAX_LENGTH} characters")
        
        # Perform sentiment analysis
        result = sentiment_pipeline(text)[0]
        
        # Extract sentiment and confidence
        label = result['label']
        confidence = result['score'] * 100  # Convert to percentage
        
        # Add emoji indicators
        if label == "POSITIVE":
            sentiment_text = "😊 POSITIVE"
        elif label == "NEGATIVE":
            sentiment_text = "😞 NEGATIVE"
        else:
            sentiment_text = f"🤔 {label}"
        
        logger.info(f"Analysis complete: {label} ({confidence:.2f}%)")
        return sentiment_text, confidence
        
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        return f"❌ Error: {str(e)}", 0.0


# Example reviews for users to try
examples = [
    ["This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. A masterpiece!"],
    ["Terrible waste of time. The story made no sense and the characters were poorly developed. Very disappointing."],
    ["An instant classic! Beautiful cinematography, compelling performances, and a story that stays with you long after the credits roll."],
    ["I couldn't even finish watching it. Boring, predictable, and poorly executed. Not recommended at all."],
    ["A decent movie with some good moments, though it could have been better. The ending was quite satisfying."]
]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎭 Movie Review Sentiment Analysis
        
        Analyze the sentiment of movie reviews using state-of-the-art AI! 
        This app uses **DistilBERT** fine-tuned on movie reviews to determine whether 
        a review expresses a **positive** or **negative** sentiment.
        
        Simply paste or type a movie review below and get instant sentiment analysis 
        with confidence scores! 🎬✨
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="📝 Enter Movie Review",
                placeholder="Type or paste a movie review here... (max 512 characters)",
                lines=5,
                max_lines=10
            )
            
            with gr.Row():
                clear_btn = gr.ClearButton(components=[input_text])
                submit_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")
        
        with gr.Column(scale=1):
            sentiment_output = gr.Label(
                label="🎯 Sentiment",
                num_top_classes=1
            )
            confidence_output = gr.Number(
                label="📊 Confidence Score (%)",
                precision=2
            )
    
    gr.Markdown("### 💡 Try These Examples:")
    gr.Examples(
        examples=examples,
        inputs=input_text,
        outputs=[sentiment_output, confidence_output],
        fn=analyze_sentiment,
        cache_examples=True
    )
    
    gr.Markdown(
        """
        ---
        
        ### ℹ️ About
        
        - **Model**: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
        - **Task**: Binary Sentiment Classification
        - **Labels**: POSITIVE 😊 / NEGATIVE 😞
        - **Max Input Length**: 512 characters
        
        ### 🚀 How It Works
        
        1. Enter your movie review in the text box
        2. Click "Analyze Sentiment" or press Enter
        3. Get instant results with sentiment label and confidence score
        
        The model has been fine-tuned on movie reviews and can accurately detect 
        positive and negative sentiments with high confidence scores!
        """
    )
    
    # Connect the button and input to the analysis function
    submit_btn.click(
        fn=analyze_sentiment,
        inputs=input_text,
        outputs=[sentiment_output, confidence_output],
        api_name="analyze"
    )
    
    input_text.submit(
        fn=analyze_sentiment,
        inputs=input_text,
        outputs=[sentiment_output, confidence_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )
