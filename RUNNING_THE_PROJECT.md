# 🎭 Movie Review Sentiment Analysis - Project Running Guide

## 📋 Quick Overview

This is a **dual-stack web application** combining:
- **Frontend**: React application (Vite) running on port **3000**
- **Backend**: Python sentiment analysis service using DistilBERT with multiple deployment options
  - Demo CLI application
  - Streamlit web interface 
  - Gradio web interface

---

## 🚀 Quick Start (Copy-Paste Commands)

### For Development Environment

```bash
# 1. Install frontend dependencies
npm install

# 2. Install backend dependencies  
cd project
pip install -r requirements.txt
cd ..

# 3. Start the React frontend (runs on http://localhost:3000)
npm start

# 4. In another terminal, run the sentiment analysis demo
cd project
python demo.py
```

---

## 📦 System Requirements

- **Python**: 3.8+
- **Node.js**: 14+
- **npm**: 6+
- **RAM**: 4GB minimum (8GB+ recommended for model training)
- **Storage**: 2GB+ (for model downloads)
- **GPU** (optional): CUDA-capable GPU for faster training/inference

---

## 📁 Project Structure

```
movie-review-sentiment-analysis/
├── 📄 index.html              # Frontend entry point
├── 📄 package.json            # Frontend dependencies (React, Vite)
├── 📄 vite.config.js          # Vite build configuration
├── src/                       # React frontend source
│   ├── App.jsx               # Main React component
│   └── index.jsx             # React entry point
│
└── project/                   # Python backend
    ├── 📄 requirements.txt     # Python dependencies
    ├── 📄 demo.py             # Quick demo script (no training needed)
    ├── 📄 README.md           # Backend documentation
    ├── app/
    │   ├── app.py            # Streamlit web interface
    │   ├── gradio_app.py     # Gradio web interface
    │   └── demo_app.py       # Alternative demo app
    ├── src/
    │   ├── data.py           # Data loading & preprocessing
    │   ├── train.py          # Training pipeline
    │   └── evaluate.py       # Evaluation & metrics
    ├── data/                 # Dataset storage (auto-downloaded)
    ├── models/               # Trained model checkpoints
    └── results/              # Evaluation plots & reports
```

---

## 🔧 Installation & Setup

### 1. Frontend Setup

```bash
# Install Node.js dependencies
npm install

# Verify installation
npm run build --dry-run
```

**Dependencies Installed:**
- React 18.2.0
- Vite 6.3.6 (build tool)
- Testing libraries (Vitest, React Testing Library)

### 2. Backend Setup

```bash
cd project

# Install Python dependencies
pip install -r requirements.txt

# Verify installation (check for any missing packages)
python -c "import torch; import transformers; print('✓ All dependencies installed')"
```

**Python Dependencies:**
- PyTorch 2.7.0 (ML framework)
- Transformers 4.57.3 (HuggingFace models)
- Datasets 4.4.1 (Dataset loading)
- Streamlit 1.52.2 (Web interface 1)
- Gradio 6.1.0 (Web interface 2)
- scikit-learn 1.6.1 (ML utilities)
- pandas, numpy, matplotlib, seaborn (Data analysis)

---

## ▶️ Running the Application

### **Option 1: Frontend Only (React App)**

```bash
# Terminal 1 - Start the Vite development server
npm start
```

Runs on: **http://localhost:3000**

Features:
- Hot module replacement (HMR) for live code updates
- React component previews
- Test runner integration

---

### **Option 2: Quick Demo (No Training Required)**

```bash
cd project

# Run the quick demo with pre-trained DistilBERT
python demo.py
```

**Output:**
- Tests sentiment analysis on 5 sample movie reviews
- Shows classification predictions with confidence scores
- Uses `distilbert-base-uncased-finetuned-sst-2-english` (pre-trained, not IMDB-specific)

**Expected Output:**
```
🎭 Movie Sentiment Analysis - Quick Demo

✓ Model loaded on CPU

Testing sentiment predictions:
📝 Review: This movie was absolutely fantastic!...
   😊 Sentiment: POSITIVE (confidence: 99.99%)

📝 Review: Terrible movie. Poor acting...
   😞 Sentiment: NEGATIVE (confidence: 99.98%)
```

---

### **Option 3: Streamlit Web Interface**

```bash
cd project

# Start the Streamlit app
streamlit run app/app.py
```

Runs on: **http://localhost:8501**

**Features:**
- Interactive text input for movie review analysis
- Real-time sentiment predictions
- Confidence score visualization
- Supports GPU acceleration if available

**First Run:**
- Loads the trained model from `models/distilbert-imdb/`
- If model not found: will display error message
  - Solution: Train the model first (see Training section below)

---

### **Option 4: Gradio Web Interface**

```bash
cd project

# Start the Gradio app
python -m app.gradio_app
```

Runs on: **http://localhost:7860** (auto-opens in browser)

**Features:**
- Drag-and-drop text input
- Beautiful, mobile-friendly interface
- Share link generation for easy collaboration
- API endpoint for programmatic access

---

## 🎓 Training the Model

### **Full Training Pipeline**

```bash
cd project

# This will:
# 1. Download IMDB dataset (50,000 reviews)
# 2. Preprocess and tokenize the data
# 3. Fine-tune DistilBERT on IMDB reviews
# 4. Evaluate on test set
# 5. Save model to models/distilbert-imdb/

python -m src.train \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --device cuda  # Use 'cpu' if no GPU available
```

**Training Time:**
- GPU (NVIDIA Tesla T4+): ~15-30 minutes
- CPU: ~2-4 hours

**Output:**
- Trained model checkpoint in `models/distilbert-imdb/`
- Training metrics and plots in `results/`
- Model performance metrics

---

## 🧪 Testing & Evaluation

### **Run Evaluation on Test Set**

```bash
cd project

# Evaluate the fine-tuned model
python -m src.evaluate --model_path models/distilbert-imdb/
```

**Output:**
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- Classification report
- Performance plots in `results/`

---

## 💡 Testing Sentiment Analysis

### **Example 1: Using the Demo Script**

```bash
python demo.py
```

Automatically tests 5 predefined movie reviews.

---

### **Example 2: Interactive CLI Testing**

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Test with custom review
review = "This movie was absolutely fantastic!"
result = classifier(review)
print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.2%}")
```

---

### **Example 3: Test with Fine-Tuned Model**

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained(
    "project/models/distilbert-imdb"
)
tokenizer = AutoTokenizer.from_pretrained("project/models/distilbert-imdb")

# Prepare input
text = "Absolutely loved this film! Best movie I've seen."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.softmax(logits, dim=-1)
    
print(f"Negative: {prediction[0][0]:.2%}, Positive: {prediction[0][1]:.2%}")
```

---

## 🌐 Running Multiple Services

To run the full application stack simultaneously:

```bash
# Terminal 1 - Frontend
npm start
# Runs on http://localhost:3000

# Terminal 2 - Streamlit backend
cd project && streamlit run app/app.py
# Runs on http://localhost:8501

# Terminal 3 - Alternative: Gradio backend
# cd project && python -m app.gradio_app
# Runs on http://localhost:7860
```

---

## 🐛 Troubleshooting

### **Problem: "Model not found at models/distilbert-imdb"**

**Solution:**
```bash
cd project

# Option 1: Use the quick demo (no training needed)
python demo.py

# Option 2: Train the model
python -m src.train --epochs 1  # Quick test run
```

---

### **Problem: "No module named 'streamlit' / 'gradio' / 'torch'?"**

**Solution:**
```bash
# Reinstall all dependencies
cd project
pip install --upgrade pip
pip install -r requirements.txt
```

---

### **Problem: "CUDA out of memory" during training**

**Solution:**
```bash
# Reduce batch size
python -m src.train --batch_size 8

# Or use CPU instead (slower but works)
python -m src.train --device cpu
```

---

### **Problem: Port already in use (3000, 8501, 7860)**

**Solution:**
```bash
# Find process using the port
lsof -i :3000

# Kill the process
kill -9 <PID>

# Or use a different port
streamlit run app/app.py --server.port 8502
```

---

### **Problem: "SSL certificate verification failed"**

**Solution:**
```bash
# Disable SSL verification (temporary, for development only)
pip install --trusted-host pypi.python.org -r requirements.txt

# Or upgrade certificates
pip install --upgrade certifi
```

---

## 📊 Expected Output Examples

### Demo Script Output
```
🎭 Movie Sentiment Analysis - Quick Demo

✓ Model loaded on CPU

Testing sentiment predictions:

📝 Review: This movie was absolutely fantastic!...
   😊 Sentiment: POSITIVE (confidence: 99.99%)

📝 Review: Terrible movie. Poor acting...
   😞 Sentiment: NEGATIVE (confidence: 99.98%)

✅ Demo completed successfully!
```

### Streamlit Interface
- Interactive input text area for movie reviews
- Real-time sentiment prediction
- Confidence bars for positive/negative predictions
- Processing time display
- Model device info (CPU/GPU)

### Gradio Interface
- Clean text input field
- Submit button
- Results showing:
  - Sentiment label (POSITIVE/NEGATIVE)
  - Confidence scores
  - Processing time
- Shareable API endpoint

---

## 🚢 Deployment Options

### **1. Streamlit Cloud**
```bash
# Push code to GitHub, then deploy at:
# https://streamlit.io/cloud
```

### **2. Hugging Face Spaces**
```bash
# Create a Gradio app with HF Spaces
# Automatic deployment from GitHub
```

### **3. Docker Container**
```bash
# Build and run in Docker
docker build -t sentiment-analysis .
docker run -p 8501:8501 sentiment-analysis
```

### **4. AWS/GCP/Azure**
- See [DEPLOYMENT.md](project/DEPLOYMENT.md) for detailed instructions

---

## 📈 Model Information

**Model Used:** DistilBERT
- **Size:** ~268MB
- **Parameters:** 66M (40% smaller than BERT)
- **Speed:** 60% faster than BERT-base
- **Accuracy:** Retains 97% of BERT's performance

**Dataset:** IMDB Movie Reviews (50,000 samples)
- 25,000 training reviews
- 25,000 test reviews
- Binary classification (Positive/Negative)
- Perfectly balanced dataset

---

## 📚 Additional Resources

- **HuggingFace Model Card**: https://huggingface.co/distilbert-base-uncased
- **IMDB Dataset**: https://huggingface.co/datasets/imdb
- **Transformers Library Docs**: https://huggingface.co/docs/transformers/
- **PyTorch Docs**: https://pytorch.org/docs/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Gradio Docs**: https://www.gradio.app/

---

## 🔍 Advanced Usage

### **Using with GPU Acceleration**

```bash
# Check if GPU is available
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# Run training with GPU
python -m src.train --device cuda --epochs 3

# Specify GPU device (for multi-GPU systems)
CUDA_VISIBLE_DEVICES=0 python -m src.train
```

### **Batch Processing**

```python
from transformers import pipeline
import pandas as pd

classifier = pipeline("sentiment-analysis", device=0)

# Load CSV with reviews
df = pd.read_csv("reviews.csv")

# Process in batches
results = classifier(df['review'].tolist())

# Save results
df['sentiment'] = results
df.to_csv("reviews_with_sentiment.csv", index=False)
```

### **API Integration**

```bash
# Gradio creates automatic API endpoint
curl http://localhost:7860/api/predict -X POST \
  -H "Content-Type: application/json" \
  -d '{"data": ["Great movie!"]}'
```

---

## ✨ Next Steps

1. **Run the demo** to verify setup: `python demo.py`
2. **Start the frontend**: `npm start`
3. **Try Streamlit interface**: `streamlit run app/app.py`
4. **Train the model** (optional): `python -m src.train`
5. **Deploy to production** (see DEPLOYMENT.md)

---

## 📞 Support

For issues, check:
1. The troubleshooting section above
2. [Project README](project/README.md) for backend details
3. [Deployment Guide](project/DEPLOYMENT.md) for production setup
4. GitHub Issues on the repository

---

**Last Updated:** January 1, 2026
**Project Status:** Ready to Run ✅
