# Setup Guide for YouTube Shorts Product Filter

## Prerequisites

Before setting up the YouTube Shorts Product Filter, ensure you have:

1. **Python 3.7 or higher** installed on your system
2. **pip** package manager
3. **Internet connection** for API access
4. **YouTube Data API v3 key** (instructions below)

## Step 1: Get YouTube Data API Key

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the YouTube Data API v3:
   - Navigate to "APIs & Services" > "Library"
   - Search for "YouTube Data API v3"
   - Click on it and press "Enable"
4. Create credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "API Key"
   - Copy the generated API key
   - (Optional) Restrict the key to YouTube Data API v3 for security

## Step 2: Install Dependencies

### Option A: Using requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```

### Option B: Manual Installation
```bash
pip install scikit-learn pandas google-api-python-client psutil joblib numpy
```

## Step 3: Configure the System

1. **Set your API key** in the main script:
   ```python
   API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
   ```

2. **Train the model** (if not using pre-trained):
   ```bash
   python train_model.py
   ```

## Step 4: Test the Installation

### Run the Demo
```bash
python demo_filter.py
```

This should output:
```
ML model and vectorizer loaded successfully.
=== YouTube Shorts Product Filter Demo ===
...
```

### Run Performance Test
```bash
python performance_test.py
```

## Step 5: Basic Usage

### Example 1: Filter with Search Query
```python
from youtube_shorts_filter import YouTubeShortsProductFilter

# Initialize the filter
filter_system = YouTubeShortsProductFilter("YOUR_API_KEY")

# Search and filter
results = filter_system.run_filter("tech review", max_results=20)

# Print results
for short in results:
    print(f"Found: {short['title']}")
```

### Example 2: Test Individual Components
```python
from product_detector import detect_products_regex

# Test regex detection
text = "Check out this iPhone 15 review! Link in bio to buy."
patterns = detect_products_regex(text)
print(f"Detected patterns: {patterns}")
```

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Install missing dependencies
```bash
pip install [missing_module_name]
```

### Issue: "API Key Invalid"
**Solution**: 
1. Verify your API key is correct
2. Ensure YouTube Data API v3 is enabled
3. Check API quotas in Google Cloud Console

### Issue: "Model files not found"
**Solution**: Run the training script
```bash
python train_model.py
```

### Issue: High memory usage
**Solution**: Use the optimized model
```bash
python performance_test.py  # This creates optimized models
```

Then modify your code to use:
```python
filter_system = YouTubeShortsProductFilter(
    "YOUR_API_KEY",
    model_path="optimized_product_detection_model.pkl",
    vectorizer_path="optimized_tfidf_vectorizer.pkl"
)
```

## Advanced Configuration

### Custom Regex Patterns
Edit `product_detector.py` to add your own patterns:
```python
patterns = [
    r'your_brand_name',
    r'custom_product_pattern',
    # ... existing patterns
]
```

### Model Retraining
To retrain with your own data, modify the `DATA` list in `train_model.py`:
```python
DATA = [
    {"text": "your training text", "is_product": 1},
    # ... more training examples
]
```

### Batch Processing
For large datasets, process in batches:
```python
video_ids = ["id1", "id2", ...]  # Your video IDs
batch_size = 10

for i in range(0, len(video_ids), batch_size):
    batch = video_ids[i:i+batch_size]
    results = filter_system.filter_shorts_with_products(batch)
    # Process results
```

## Performance Optimization

### For Low RAM Systems (< 1GB)
1. Use optimized models
2. Process smaller batches
3. Clear variables after use:
   ```python
   del large_variable
   import gc
   gc.collect()
   ```

### For High Volume Processing
1. Implement caching for repeated API calls
2. Use multiprocessing for parallel processing
3. Consider database storage for results

## Next Steps

1. **Customize patterns** for your specific use case
2. **Collect training data** to improve ML model accuracy
3. **Implement caching** for production use
4. **Add logging** for monitoring and debugging
5. **Create a web interface** for easier use

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Verify all dependencies are installed
3. Ensure your API key has proper permissions
4. Test with the demo script first

