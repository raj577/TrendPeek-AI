#!/usr/bin/env python3
"""
Tutorial: Building the Machine Learning Component
Learn how to create a lightweight product detection model
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

def create_training_data():
    """Step 1: Create labeled training data"""
    # In real projects, you'd collect this from actual YouTube data
    training_data = [
        # Product-related texts (label = 1)
        {"text": "Check out my new iPhone 15 Pro Max! Link in bio to buy now.", "is_product": 1},
        {"text": "Unboxing the latest Samsung Galaxy S24 Ultra. So cool!", "is_product": 1},
        {"text": "My favorite Nike shoes for running. Get them on sale!", "is_product": 1},
        {"text": "This product review will show you everything you need to know.", "is_product": 1},
        {"text": "Found a great deal on amazon.com for only $25.99!", "is_product": 1},
        {"text": "New gaming setup reveal! Check out the RTX 4090.", "is_product": 1},
        {"text": "Reviewing the new Sony WH-1000XM5 headphones.", "is_product": 1},
        {"text": "Testing out the new GoPro Hero 11 in action.", "is_product": 1},
        {"text": "Get your limited edition t-shirt now!", "is_product": 1},
        {"text": "Building a custom PC from scratch with these parts.", "is_product": 1},
        
        # Non-product texts (label = 0)
        {"text": "Just a regular video, nothing to see here.", "is_product": 0},
        {"text": "A beautiful sunset view from my balcony.", "is_product": 0},
        {"text": "Cooking a delicious meal for dinner tonight.", "is_product": 0},
        {"text": "Learning Python programming for data science.", "is_product": 0},
        {"text": "Travel vlog from my trip to Japan.", "is_product": 0},
        {"text": "Morning routine and healthy breakfast ideas.", "is_product": 0},
        {"text": "DIY home decor ideas on a budget.", "is_product": 0},
        {"text": "My top 5 books to read this summer.", "is_product": 0},
        {"text": "Exploring the ancient ruins of Rome.", "is_product": 0},
        {"text": "Quick tutorial on how to use Adobe Photoshop.", "is_product": 0},
    ]
    
    return pd.DataFrame(training_data)

def train_lightweight_model():
    """Step 2: Train a lightweight ML model"""
    print("🤖 Training Product Detection Model...")
    
    # Load training data
    df = create_training_data()
    X = df["text"]
    y = df["is_product"]
    
    print(f"Training on {len(df)} examples")
    print(f"Product examples: {sum(y)}")
    print(f"Non-product examples: {len(y) - sum(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Step 3: Convert text to numerical features using TF-IDF
    print("\n📊 Converting text to features...")
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Limit features for lightweight model
        min_df=1,           # Include rare terms (small dataset)
        stop_words='english',
        ngram_range=(1, 2)  # Include single words and bigrams
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_vec.shape}")
    
    # Step 4: Train Logistic Regression (lightweight and fast)
    print("\n🧠 Training model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train_vec, y_train)
    
    # Step 5: Evaluate model
    print("\n📈 Evaluating model...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2%}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['No Product', 'Product']))
    
    # Step 6: Save model and vectorizer
    print("\n💾 Saving model...")
    joblib.dump(model, "tutorial_model.pkl")
    joblib.dump(vectorizer, "tutorial_vectorizer.pkl")
    
    # Check model size
    import os
    model_size = os.path.getsize("tutorial_model.pkl") / 1024  # KB
    vec_size = os.path.getsize("tutorial_vectorizer.pkl") / 1024  # KB
    
    print(f"Model size: {model_size:.1f} KB")
    print(f"Vectorizer size: {vec_size:.1f} KB")
    print(f"Total size: {model_size + vec_size:.1f} KB")
    
    return model, vectorizer

def test_model():
    """Step 7: Test the trained model"""
    print("\n🧪 Testing model with new examples...")
    
    # Load saved model
    model = joblib.load("tutorial_model.pkl")
    vectorizer = joblib.load("tutorial_vectorizer.pkl")
    
    test_texts = [
        "Amazing new gadget you need to see!",
        "Just sharing my morning coffee routine",
        "Unboxing my latest tech purchase",
        "Beautiful nature photography tips",
        "This product changed my life - link below!"
    ]
    
    for text in test_texts:
        # Convert text to features
        text_vec = vectorizer.transform([text])
        
        # Get prediction and probability
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec)[0]
        
        result = "PRODUCT" if prediction == 1 else "NO PRODUCT"
        confidence = max(probability) * 100
        
        print(f"Text: '{text}'")
        print(f"Prediction: {result} (confidence: {confidence:.1f}%)")
        print()

if __name__ == "__main__":
    # Train the model
    model, vectorizer = train_lightweight_model()
    
    # Test it
    test_model()
    
    print("✅ Tutorial complete! You now have a working ML model.")