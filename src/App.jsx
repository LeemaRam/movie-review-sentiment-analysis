import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);
  const [history, setHistory] = useState([]);

  const API_URL = '';

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
    // Check health every 5 seconds
    const interval = setInterval(checkApiHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/api/health`);
      if (response.ok) {
        setApiHealth('connected');
      } else {
        setApiHealth('disconnected');
      }
    } catch (err) {
      setApiHealth('disconnected');
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();

    if (!review.trim()) {
      setError('Please enter a movie review');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: review }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }

      const data = await response.json();
      setResult(data);

      // Add to history
      setHistory([data, ...history.slice(0, 9)]);

      // Clear input
      setReview('');
    } catch (err) {
      setError(err.message || 'Failed to connect to API. Make sure the backend is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentEmoji = (sentiment) => {
    return sentiment === 'POSITIVE' ? '😊' : '😞';
  };

  const getSentimentColor = (sentiment) => {
    return sentiment === 'POSITIVE' ? '#4caf50' : '#f44336';
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎭 Movie Review Sentiment Analysis</h1>
        <p>Analyze the sentiment of movie reviews using DistilBERT AI</p>
      </header>

      <main className="App-main">
        {/* API Status */}
        <div className={`status-indicator ${apiHealth}`}>
          <span className="status-dot"></span>
          <span className="status-text">
            {apiHealth === 'connected'
              ? '✓ API Connected'
              : apiHealth === 'disconnected'
                ? '✗ API Disconnected - Start the backend with: python project/api.py'
                : '⏳ Checking...'}
          </span>
        </div>

        {/* Main Form */}
        <section className="analyzer-section">
          <form onSubmit={handlePredict} className="analyzer-form">
            <label htmlFor="review">Enter a Movie Review:</label>
            <textarea
              id="review"
              value={review}
              onChange={(e) => setReview(e.target.value)}
              placeholder="Example: This movie was absolutely fantastic! The cinematography was breathtaking and the acting was superb."
              maxLength="512"
              disabled={loading || apiHealth !== 'connected'}
              rows="4"
            />
            <div className="input-info">
              <span>{review.length}/512 characters</span>
            </div>
            <button
              type="submit"
              disabled={loading || apiHealth !== 'connected'}
              className="analyze-btn"
            >
              {loading ? '⏳ Analyzing...' : '🔍 Analyze Sentiment'}
            </button>
          </form>

          {/* Error Message */}
          {error && (
            <div className="error-message">
              <p>❌ {error}</p>
            </div>
          )}

          {/* Result Display */}
          {result && (
            <div className="result-box" style={{ borderColor: getSentimentColor(result.sentiment) }}>
              <div className="result-header">
                <h2>Analysis Result</h2>
                <span className="sentiment-emoji">{getSentimentEmoji(result.sentiment)}</span>
              </div>

              <div className="result-content">
                <div className="result-row">
                  <label>Review:</label>
                  <p className="result-text">"{result.text}"</p>
                </div>

                <div className="result-row">
                  <label>Sentiment:</label>
                  <p className="result-sentiment" style={{ color: getSentimentColor(result.sentiment) }}>
                    <strong>{result.sentiment}</strong>
                  </p>
                </div>

                <div className="result-row">
                  <label>Confidence:</label>
                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${result.score * 100}%`,
                        backgroundColor: getSentimentColor(result.sentiment)
                      }}
                    >
                      <span className="confidence-text">{result.confidence}</span>
                    </div>
                  </div>
                </div>

                <div className="result-row">
                  <label>Score:</label>
                  <p>{result.score.toFixed(4)}</p>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* History Section */}
        {history.length > 0 && (
          <section className="history-section">
            <h2>📋 Recent Analyses</h2>
            <div className="history-list">
              {history.map((item, index) => (
                <div key={index} className="history-item">
                  <div className="history-emoji">
                    {getSentimentEmoji(item.sentiment)}
                  </div>
                  <div className="history-content">
                    <p className="history-text">"{item.text.substring(0, 60)}{item.text.length > 60 ? '...' : ''}"</p>
                    <p className="history-sentiment" style={{ color: getSentimentColor(item.sentiment) }}>
                      {item.sentiment} • {item.confidence}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Examples Section */}
        {history.length === 0 && (
          <section className="examples-section">
            <h2>💡 Try These Examples</h2>
            <div className="examples-grid">
              {[
                "This movie was absolutely fantastic! The cinematography was breathtaking.",
                "Terrible movie. Poor acting and a confusing plot.",
                "I loved every minute of it. A masterpiece!",
                "Complete waste of time and money. Very disappointing."
              ].map((example, index) => (
                <button
                  key={index}
                  className="example-btn"
                  onClick={() => {
                    setReview(example);
                  }}
                  disabled={loading || apiHealth !== 'connected'}
                >
                  {example.substring(0, 50)}...
                </button>
              ))}
            </div>
          </section>
        )}
      </main>

      <footer className="App-footer">
        <p>🚀 Powered by DistilBERT • Flask API • React • Vite</p>
        <p>Model: distilbert-base-uncased-finetuned-sst-2-english</p>
      </footer>
    </div>
  );
}

export default App;
