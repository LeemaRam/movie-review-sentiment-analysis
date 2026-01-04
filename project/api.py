"""Flask API for sentiment analysis."""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load the pre-trained sentiment model
print("Loading sentiment analysis model...")
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)
print("✓ Model loaded successfully")

device_type = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
print(f"✓ Running on: {device_type}\n")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "device": device_type
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict sentiment for the given review text.
    
    Expected JSON:
    {
        "text": "Your movie review here"
    }
    
    Returns:
    {
        "text": "Your movie review here",
        "sentiment": "POSITIVE" or "NEGATIVE",
        "score": 0.9999,
        "confidence": "99.99%"
    }
    """
    try:
        data = request.get_json()
        
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data["text"].strip()
        
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if len(text) > 512:
            return jsonify({"error": "Text too long (max 512 characters)"}), 400
        
        # Get prediction
        result = classifier(text)[0]
        
        return jsonify({
            "text": text,
            "sentiment": result["label"],
            "score": round(result["score"], 4),
            "confidence": f"{result['score']*100:.2f}%"
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    """Predict sentiment for multiple reviews.
    
    Expected JSON:
    {
        "reviews": ["review1", "review2", ...]
    }
    
    Returns:
    {
        "results": [
            {"text": "review1", "sentiment": "POSITIVE", "score": 0.99, "confidence": "99%"},
            ...
        ],
        "count": 2
    }
    """
    try:
        data = request.get_json()
        
        if not data or "reviews" not in data:
            return jsonify({"error": "Missing 'reviews' field in request"}), 400
        
        reviews = data["reviews"]
        
        if not isinstance(reviews, list):
            return jsonify({"error": "'reviews' must be a list"}), 400
        
        if len(reviews) > 100:
            return jsonify({"error": "Too many reviews (max 100)"}), 400
        
        # Filter empty reviews
        reviews = [r.strip() for r in reviews if r.strip()]
        
        if not reviews:
            return jsonify({"error": "No valid reviews provided"}), 400
        
        # Get predictions
        results = classifier(reviews)
        
        return jsonify({
            "results": [
                {
                    "text": review,
                    "sentiment": result["label"],
                    "score": round(result["score"], 4),
                    "confidence": f"{result['score']*100:.2f}%"
                }
                for review, result in zip(reviews, results)
            ],
            "count": len(reviews)
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎭 Sentiment Analysis API Server")
    print("="*60)
    print("\n📍 Starting Flask API on http://localhost:5000")
    print("\n📚 Available endpoints:")
    print("   GET  /api/health         - Health check")
    print("   POST /api/predict        - Single review prediction")
    print("   POST /api/batch          - Batch predictions")
    print("\n" + "="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False)
