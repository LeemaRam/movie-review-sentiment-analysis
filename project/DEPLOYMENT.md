# 🚀 Deployment Guide

Complete guide for deploying your sentiment analysis model to production environments.

## Table of Contents
- [HuggingFace Spaces Deployment](#huggingface-spaces-deployment)
- [Push Model to HuggingFace Hub](#push-model-to-huggingface-hub)
- [GitHub Repository Setup](#github-repository-setup)
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
- [Docker Deployment](#docker-deployment)

---

## 🤗 HuggingFace Spaces Deployment

HuggingFace Spaces provides free hosting for ML demos with automatic GPU support.

### Prerequisites
- HuggingFace account (sign up at [huggingface.co](https://huggingface.co))
- Git installed locally
- Trained model (or use demo app with pre-trained model)

### Step 1: Create a New Space

1. **Navigate to Spaces:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"

2. **Configure Space:**
   ```
   Space name: movie-sentiment-analyzer
   License: MIT
   SDK: Gradio (or Streamlit)
   Visibility: Public (or Private)
   Hardware: CPU Basic (free) or GPU (paid)
   ```

3. **Click "Create Space"**

### Step 2: Clone Your Space Repository

```bash
# Clone the empty Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/movie-sentiment-analyzer
cd movie-sentiment-analyzer

# Configure git (if not already done)
git config user.email "your-email@example.com"
git config user.name "Your Name"
```

### Step 3: Prepare Files for Deployment

#### Option A: Using Pre-trained Model (Recommended for Quick Start)

```bash
# Copy demo app (uses pre-trained model, no training needed)
cp /path/to/project/app/demo_app.py app.py

# Create minimal requirements.txt
cat > requirements.txt << 'EOF'
transformers==4.57.3
torch==2.7.0
gradio==6.1.0
EOF

# Create README.md for your Space
cat > README.md << 'EOF'
---
title: Movie Sentiment Analyzer
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: false
---

# Movie Sentiment Analysis

Sentiment analysis for movie reviews using DistilBERT.

## Usage
Enter a movie review and get instant sentiment predictions (Positive/Negative).

## Model
- **Architecture:** DistilBERT
- **Task:** Binary Sentiment Classification
- **Pre-trained on:** SST-2 dataset
EOF
```

#### Option B: Using Your Fine-tuned Model

```bash
# Copy app that loads your fine-tuned model
cp /path/to/project/app/gradio_app.py app.py

# Copy trained model
cp -r /path/to/project/models/distilbert-imdb ./distilbert-imdb

# Create requirements.txt
cat > requirements.txt << 'EOF'
transformers==4.57.3
torch==2.7.0
gradio==6.1.0
EOF

# Update app.py to load local model
# Edit app.py: change model_path to "./distilbert-imdb"
```

### Step 4: Push to HuggingFace Spaces

```bash
# Add all files
git add .

# Commit changes
git commit -m "Initial deployment: Movie sentiment analyzer"

# Push to HuggingFace (uses git-lfs for large files)
git push
```

### Step 5: Monitor Build

- Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/movie-sentiment-analyzer`
- Watch the build logs in the "Logs" tab
- Build typically takes 2-5 minutes
- Once built, your app will be live!

### Troubleshooting HuggingFace Spaces

**Issue: Build fails with memory error**
```yaml
# Add .devcontainer/devcontainer.json
{
  "postCreateCommand": "pip install -r requirements.txt"
}
```

**Issue: App crashes on startup**
- Check logs for missing dependencies
- Ensure model files are included
- Verify Python version compatibility

**Issue: Slow loading**
- Use smaller models (DistilBERT is already optimized)
- Enable caching with `@st.cache_resource` or Gradio's built-in caching
- Consider upgrading to GPU hardware (paid)

---

## 📦 Push Model to HuggingFace Hub

Share your trained model on HuggingFace Model Hub for easy reuse.

### Step 1: Install HuggingFace CLI

```bash
pip install huggingface_hub
```

### Step 2: Login to HuggingFace

```bash
huggingface-cli login
```

Enter your access token (get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

### Step 3: Push Model Using Python

Create `push_model.py`:

```python
"""Push trained model to HuggingFace Hub."""
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load your trained model
model_path = Path("models/distilbert-imdb")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Your HuggingFace username and desired model name
repo_id = "YOUR_USERNAME/distilbert-imdb-sentiment"

# Push to hub
print(f"Pushing model to {repo_id}...")
model.push_to_hub(repo_id, commit_message="Add fine-tuned IMDB sentiment model")
tokenizer.push_to_hub(repo_id, commit_message="Add tokenizer")

print(f"✅ Model successfully pushed to https://huggingface.co/{repo_id}")
```

Run it:
```bash
cd project
python push_model.py
```

### Step 4: Add Model Card

Create a comprehensive model card on the HuggingFace website:

```markdown
---
language: en
license: apache-2.0
tags:
- sentiment-analysis
- text-classification
- distilbert
- imdb
datasets:
- imdb
metrics:
- accuracy
- f1
model-index:
- name: distilbert-imdb-sentiment
  results:
  - task:
      type: text-classification
      name: Sentiment Analysis
    dataset:
      name: IMDB
      type: imdb
    metrics:
    - type: accuracy
      value: 0.93
      name: Accuracy
    - type: f1
      value: 0.93
      name: F1
---

# DistilBERT IMDB Sentiment Classifier

Fine-tuned DistilBERT model for binary sentiment classification on IMDB movie reviews.

## Model Description

- **Base Model:** distilbert-base-uncased
- **Task:** Binary Sentiment Classification (Positive/Negative)
- **Dataset:** IMDB Movie Reviews (50k samples)
- **Accuracy:** 93%
- **F1 Score:** 93%

## Usage

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", 
                     model="YOUR_USERNAME/distilbert-imdb-sentiment")

result = classifier("This movie was fantastic!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9987}]
```

## Training

- Epochs: 3
- Batch size: 16
- Learning rate: 5e-5
- Optimizer: AdamW

## Limitations

- Trained specifically on movie reviews
- Binary classification only (no neutral sentiment)
- English language only
```

### Step 5: Update Your App to Use Hub Model

Update `app/gradio_app.py`:

```python
# Instead of loading from local path:
model = AutoModelForSequenceClassification.from_pretrained(
    "YOUR_USERNAME/distilbert-imdb-sentiment"
)
tokenizer = AutoTokenizer.from_pretrained(
    "YOUR_USERNAME/distilbert-imdb-sentiment"
)
```

---

## 🐙 GitHub Repository Setup

Share your code on GitHub for collaboration and version control.

### Step 1: Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `movie-sentiment-analysis`
3. Description: `Binary sentiment classification using DistilBERT on IMDB dataset`
4. Public or Private
5. Don't initialize with README (we already have one)
6. Click "Create repository"

### Step 2: Initialize Git Locally

```bash
cd /path/to/project

# Initialize git repository
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# HuggingFace
.cache/
wandb/

# Data and models (too large for git)
data/*
!data/.gitkeep
models/*
!models/.gitkeep
results/*
!results/.gitkeep

# Logs
*.log
training.log
demo.log

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Environment
.env
.env.local
EOF

# Create placeholder files to keep directories
touch data/.gitkeep models/.gitkeep results/.gitkeep
```

### Step 3: Add and Commit Files

```bash
# Add all files
git add .

# Commit
git commit -m "Initial commit: Sentiment analysis project with DistilBERT"
```

### Step 4: Push to GitHub

```bash
# Add remote origin
git remote add origin https://github.com/YOUR_USERNAME/movie-sentiment-analysis.git

# Push to main branch
git branch -M main
git push -u origin main
```

### Step 5: Add GitHub Repository Features

**Add Topics:**
- Go to repository settings
- Add topics: `nlp`, `sentiment-analysis`, `transformers`, `distilbert`, `huggingface`, `pytorch`, `machine-learning`

**Enable GitHub Actions (Optional CI/CD):**

Create `.github/workflows/test.yml`:

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run tests
      run: |
        python demo.py
```

---

## ☁️ Streamlit Cloud Deployment

Deploy your Streamlit app for free on Streamlit Cloud.

### Step 1: Prepare Streamlit App

Ensure your Streamlit app is ready:
- File: `app/app.py`
- Update model loading to use HuggingFace Hub model

### Step 2: Create Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### Step 3: Create `packages.txt` (if needed)

For system-level dependencies:

```bash
# packages.txt
```

Usually not needed for this project.

### Step 4: Optimize `requirements.txt` for Streamlit Cloud

Create `requirements_streamlit.txt`:

```txt
torch==2.7.0
transformers==4.57.3
streamlit==1.52.2
scikit-learn==1.6.1
```

### Step 5: Deploy on Streamlit Cloud

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Streamlit Cloud configuration"
   git push
   ```

2. **Go to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

3. **Deploy App:**
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/movie-sentiment-analysis`
   - Branch: `main`
   - Main file path: `app/app.py`
   - Advanced settings:
     - Python version: 3.10
     - Secrets: Add any API keys if needed
   - Click "Deploy"

4. **Wait for Deployment:**
   - Takes 5-10 minutes for first deployment
   - App will be available at: `https://YOUR-USERNAME-movie-sentiment-analysis.streamlit.app`

### Step 6: Add Streamlit Secrets (if needed)

In Streamlit Cloud dashboard:
- Go to App settings → Secrets
- Add secrets in TOML format:

```toml
# No secrets needed for this project
# But example for future use:
HUGGINGFACE_TOKEN = "your_token_here"
```

### Streamlit Cloud Troubleshooting

**Issue: App crashes with memory error**
- Streamlit Cloud has 1GB memory limit
- Use smaller models or optimize loading
- Clear cache periodically

**Issue: Slow cold starts**
- Use `@st.cache_resource` decorator
- Preload model in cached function
- Consider HuggingFace Spaces for better performance

**Issue: Deployment fails**
```bash
# Check logs in Streamlit Cloud dashboard
# Common fixes:
- Update requirements.txt versions
- Remove incompatible dependencies
- Check Python version compatibility
```

---

## 🐳 Docker Deployment

Containerize your application for consistent deployment anywhere.

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY src/ ./src/

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Create docker-compose.yml

```yaml
version: '3.8'

services:
  sentiment-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker build -t sentiment-analyzer:latest .

# Run container
docker run -p 8501:8501 sentiment-analyzer:latest

# Or use docker-compose
docker-compose up -d
```

### Push to Docker Hub

```bash
# Tag image
docker tag sentiment-analyzer:latest YOUR_USERNAME/sentiment-analyzer:latest

# Login
docker login

# Push
docker push YOUR_USERNAME/sentiment-analyzer:latest
```

---

## 🔧 Environment Variables

For production deployments, use environment variables:

```bash
# .env file (don't commit this!)
MODEL_NAME=YOUR_USERNAME/distilbert-imdb-sentiment
HUGGINGFACE_TOKEN=your_token_here
MAX_LENGTH=256
BATCH_SIZE=16
```

Load in Python:
```python
import os
from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("MODEL_NAME", "distilbert-base-uncased")
```

---

## 📊 Monitoring and Analytics

### Add Usage Analytics

```python
import streamlit as st

# Track usage
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0

# Increment on prediction
st.session_state.prediction_count += 1
```

### Error Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

---

## 🎉 Deployment Checklist

- [ ] Model trained and evaluated
- [ ] Code pushed to GitHub
- [ ] Model pushed to HuggingFace Hub (optional)
- [ ] HuggingFace Space created and deployed
- [ ] Streamlit Cloud app deployed (optional)
- [ ] README updated with deployment links
- [ ] Model card added to HuggingFace
- [ ] Error handling implemented
- [ ] Loading states added to UI
- [ ] Input validation added
- [ ] Rate limiting considered (for public apps)
- [ ] Analytics/monitoring set up (optional)

---

## 🔗 Quick Links Template

Add to your README.md:

```markdown
## 🌐 Live Demos

- **HuggingFace Space:** https://huggingface.co/spaces/YOUR_USERNAME/movie-sentiment-analyzer
- **Streamlit Cloud:** https://YOUR-USERNAME-movie-sentiment-analysis.streamlit.app
- **Model on Hub:** https://huggingface.co/YOUR_USERNAME/distilbert-imdb-sentiment
- **GitHub Repo:** https://github.com/YOUR_USERNAME/movie-sentiment-analysis
```

---

## 🆘 Support

If you encounter issues:
1. Check deployment platform docs (HuggingFace, Streamlit Cloud)
2. Review build logs for errors
3. Test locally first: `streamlit run app/app.py`
4. Check requirements.txt compatibility
5. Open an issue on GitHub

---

**Happy Deploying! 🚀**
