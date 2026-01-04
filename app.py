import gradio as gr
from transformers import pipeline
import torch

# Initialize sentiment analysis pipeline
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)

def analyze_sentiment(text):
    if not text or len(text.strip()) == 0:
        return "⚠️ Please enter a review", 0.0
    
    try:
        text = text[:512]
        result = classifier(text)[0]
        sentiment = result['label']
        confidence = result['score'] * 100
        emoji = "✅" if sentiment == "POSITIVE" else "❌"
        sentiment_text = f"{emoji} {sentiment}"
        return sentiment_text, confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        label="Enter Movie Review",
        placeholder="Type or paste a movie review here...",
        lines=5,
        max_lines=10
    ),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Number(label="Confidence Score (%)", precision=2)
    ],
    title="🎭 Movie Review Sentiment Analysis",
    description="""
    Analyze movie reviews using DistilBERT AI model.
    
    **Model:** DistilBERT fine-tuned on sentiment classification
    **Accuracy:** ~92-94%
    """,
    examples=[
        ["This movie was absolutely fantastic! The cinematography was breathtaking. "],
        ["Terrible movie. Poor acting and confusing plot. "],
        ["I loved every minute of it. A masterpiece!"],
        ["Complete waste of time and money. "],
        ["It was okay, not great but not terrible. "]
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__": 
    demo.launch()
