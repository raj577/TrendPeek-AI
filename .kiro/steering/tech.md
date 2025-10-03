# Technology Stack & Build System

## Core Technologies
- **Python 3.7+**: Primary programming language
- **Streamlit**: Web application framework for the UI
- **scikit-learn**: Machine learning library (Logistic Regression, TF-IDF)
- **Google API Client**: YouTube Data API v3 integration
- **joblib**: Model serialization and loading

## Key Dependencies
```
scikit-learn>=1.0.0
pandas>=1.3.0
google-api-python-client>=2.0.0
psutil>=5.8.0
joblib>=1.0.0
numpy>=1.21.0
python-dotenv>=1.0.0
streamlit (for web app)
```

## Project Structure
- **Core ML Models**: Logistic Regression + TF-IDF vectorizer
- **API Integration**: YouTube Data API v3 for video/comment retrieval
- **Hybrid Detection**: Regex patterns + ML classification
- **Web Interface**: Streamlit app with card-based results display

## Common Commands

### Setup & Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Train/retrain the ML model
python train_model.py
```

### Development & Testing
```bash
# Run the Streamlit web app
streamlit run app.py

# Test the filter system
python demo_filter.py

# Performance testing and optimization
python performance_test.py

# Test individual components
python product_detector.py
python youtube_shorts_filter.py
```

### Model Management
```bash
# Generate optimized models (smaller size)
python performance_test.py  # Creates optimized_*.pkl files

# Test ML training pipeline
python tutorial_ml_training.py
```

## Configuration
- **API Keys**: Set in Streamlit secrets or environment variables
- **Model Paths**: Configurable in YouTubeShortsProductFilter constructor
- **Regex Patterns**: Customizable in product_detector.py
- **Search Parameters**: Adjustable max_results, days, queries

## Performance Optimization
- Use optimized model files for production (`optimized_*.pkl`)
- Batch processing for large datasets
- Memory management for low-RAM environments
- API rate limiting and error handling built-in