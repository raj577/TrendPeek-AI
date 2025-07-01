import psutil
import time
import os
import sys
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_model_memory_usage():
    """Test memory usage of the trained model"""
    print("=== Model Memory Usage Test ===")
    
    # Baseline memory
    baseline_memory = get_memory_usage()
    print(f"Baseline memory usage: {baseline_memory:.2f} MB")
    
    # Load model and vectorizer
    print("Loading model and vectorizer...")
    model = joblib.load("product_detection_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    
    after_load_memory = get_memory_usage()
    model_memory = after_load_memory - baseline_memory
    print(f"Memory after loading model: {after_load_memory:.2f} MB")
    print(f"Model memory usage: {model_memory:.2f} MB")
    
    # Test prediction performance
    test_texts = [
        "Check out my new iPhone 15 Pro Max! Link in bio to buy now.",
        "Unboxing the latest Samsung Galaxy S24 Ultra. So cool!",
        "My favorite Nike shoes for running. Get them on sale at adidas.com.",
        "Just a regular video, nothing to see here.",
        "This product review will show you everything you need to know.",
        "Found a great deal on amazon.com for only $25.99!",
        "A beautiful sunset view from my balcony.",
        "Cooking a delicious meal for dinner tonight.",
        "New gaming setup reveal! Check out the RTX 4090.",
        "Learning Python programming for data science."
    ] * 100  # Repeat to test with more data
    
    print(f"\nTesting prediction performance with {len(test_texts)} texts...")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Vectorize all texts
    text_vectors = vectorizer.transform(test_texts)
    
    # Make predictions
    predictions = model.predict(text_vectors)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    processing_time = end_time - start_time
    memory_increase = end_memory - start_memory
    
    print(f"Processing time: {processing_time:.4f} seconds")
    print(f"Memory increase during processing: {memory_increase:.2f} MB")
    print(f"Texts per second: {len(test_texts) / processing_time:.2f}")
    print(f"Predictions made: {len(predictions)}")
    print(f"Product detections: {sum(predictions)}")
    
    return {
        'model_memory': model_memory,
        'processing_time': processing_time,
        'memory_increase': memory_increase,
        'texts_per_second': len(test_texts) / processing_time
    }

def test_regex_performance():
    """Test performance of regex-based detection"""
    print("\n=== Regex Performance Test ===")
    
    from product_detector import detect_products_regex
    
    test_texts = [
        "Check out my new iPhone 15 Pro Max! Link in bio to buy now.",
        "Unboxing the latest Samsung Galaxy S24 Ultra. So cool!",
        "My favorite Nike shoes for running. Get them on sale at adidas.com.",
        "Just a regular video, nothing to see here.",
        "This product review will show you everything you need to know.",
        "Found a great deal on amazon.com for only $25.99!",
        "A beautiful sunset view from my balcony.",
        "Cooking a delicious meal for dinner tonight.",
        "New gaming setup reveal! Check out the RTX 4090.",
        "Learning Python programming for data science."
    ] * 100  # Repeat to test with more data
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    detections = []
    for text in test_texts:
        result = detect_products_regex(text)
        detections.append(len(result) > 0)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    processing_time = end_time - start_time
    memory_increase = end_memory - start_memory
    
    print(f"Processing time: {processing_time:.4f} seconds")
    print(f"Memory increase during processing: {memory_increase:.2f} MB")
    print(f"Texts per second: {len(test_texts) / processing_time:.2f}")
    print(f"Product detections: {sum(detections)}")
    
    return {
        'processing_time': processing_time,
        'memory_increase': memory_increase,
        'texts_per_second': len(test_texts) / processing_time
    }

def optimize_model():
    """Create an optimized version of the model with reduced features"""
    print("\n=== Model Optimization ===")
    
    # Load original training data (simplified for demo)
    DATA = [
        {"text": "Check out my new iPhone 15 Pro Max! Link in bio to buy now.", "is_product": 1},
        {"text": "Unboxing the latest Samsung Galaxy S24 Ultra. So cool!", "is_product": 1},
        {"text": "My favorite Nike shoes for running. Get them on sale at adidas.com.", "is_product": 1},
        {"text": "Just a regular video, nothing to see here.", "is_product": 0},
        {"text": "This product review will show you everything you need to know.", "is_product": 1},
        {"text": "Found a great deal on amazon.com for only $25.99!", "is_product": 1},
        {"text": "A beautiful sunset view from my balcony.", "is_product": 0},
        {"text": "Cooking a delicious meal for dinner tonight.", "is_product": 0},
        {"text": "New gaming setup reveal! Check out the RTX 4090.", "is_product": 1},
        {"text": "Learning Python programming for data science.", "is_product": 0},
        {"text": "Travel vlog from my trip to Japan.", "is_product": 0},
        {"text": "Reviewing the new Sony WH-1000XM5 headphones.", "is_product": 1},
        {"text": "Morning routine and healthy breakfast ideas.", "is_product": 0},
        {"text": "DIY home decor ideas on a budget.", "is_product": 0},
        {"text": "Testing out the new GoPro Hero 11 in action.", "is_product": 1},
        {"text": "My top 5 books to read this summer.", "is_product": 0},
        {"text": "Exploring the ancient ruins of Rome.", "is_product": 0},
        {"text": "Quick tutorial on how to use Adobe Photoshop.", "is_product": 0},
        {"text": "Get your limited edition t-shirt now!", "is_product": 1},
        {"text": "Building a custom PC from scratch.", "is_product": 1},
    ]
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.DataFrame(DATA)
    X = df["text"]
    y = df["is_product"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create optimized vectorizer with fewer features
    print("Creating optimized model with reduced features...")
    optimized_vectorizer = TfidfVectorizer(max_features=100, min_df=1, stop_words='english')
    X_train_vec = optimized_vectorizer.fit_transform(X_train)
    X_test_vec = optimized_vectorizer.transform(X_test)
    
    # Train optimized model
    optimized_model = LogisticRegression(max_iter=1000)
    optimized_model.fit(X_train_vec, y_train)
    
    # Save optimized model
    joblib.dump(optimized_model, "optimized_product_detection_model.pkl")
    joblib.dump(optimized_vectorizer, "optimized_tfidf_vectorizer.pkl")
    
    # Compare file sizes
    original_model_size = os.path.getsize("product_detection_model.pkl") / 1024
    original_vectorizer_size = os.path.getsize("tfidf_vectorizer.pkl") / 1024
    optimized_model_size = os.path.getsize("optimized_product_detection_model.pkl") / 1024
    optimized_vectorizer_size = os.path.getsize("optimized_tfidf_vectorizer.pkl") / 1024
    
    print(f"Original model size: {original_model_size:.2f} KB")
    print(f"Original vectorizer size: {original_vectorizer_size:.2f} KB")
    print(f"Optimized model size: {optimized_model_size:.2f} KB")
    print(f"Optimized vectorizer size: {optimized_vectorizer_size:.2f} KB")
    
    total_original = original_model_size + original_vectorizer_size
    total_optimized = optimized_model_size + optimized_vectorizer_size
    reduction = ((total_original - total_optimized) / total_original) * 100
    
    print(f"Total size reduction: {reduction:.1f}%")
    
    return {
        'original_size': total_original,
        'optimized_size': total_optimized,
        'reduction_percent': reduction
    }

def generate_performance_report():
    """Generate a comprehensive performance report"""
    print("=== YouTube Shorts Product Filter Performance Report ===\n")
    
    # Test model performance
    model_results = test_model_memory_usage()
    
    # Test regex performance
    regex_results = test_regex_performance()
    
    # Optimize model
    optimization_results = optimize_model()
    
    # Generate summary
    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"ML Model Memory Usage: {model_results['model_memory']:.2f} MB")
    print(f"ML Model Processing Speed: {model_results['texts_per_second']:.2f} texts/second")
    print(f"Regex Processing Speed: {regex_results['texts_per_second']:.2f} texts/second")
    print(f"Model Size Reduction: {optimization_results['reduction_percent']:.1f}%")
    
    # Recommendations
    print("\n=== OPTIMIZATION RECOMMENDATIONS ===")
    
    if model_results['model_memory'] > 50:
        print("⚠️  Model memory usage is high. Consider using the optimized model.")
    else:
        print("✅ Model memory usage is acceptable for low-RAM environments.")
    
    if regex_results['texts_per_second'] > model_results['texts_per_second']:
        print("✅ Regex detection is faster than ML model. Use regex for initial filtering.")
    else:
        print("ℹ️  ML model performance is comparable to regex.")
    
    print("💡 For optimal performance in low-RAM environments:")
    print("   1. Use regex detection for initial filtering")
    print("   2. Apply ML model only to regex-positive results")
    print("   3. Use the optimized model for production deployment")
    print("   4. Process videos in batches to manage memory usage")

if __name__ == "__main__":
    generate_performance_report()

