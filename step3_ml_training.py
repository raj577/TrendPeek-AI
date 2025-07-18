#!/usr/bin/env python3
"""
Step 3: Machine Learning Model Training
Build a lightweight, optimized product detection model
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from typing import List, Tuple

class ProductDetectionTrainer:
    def __init__(self):
        """Initialize the ML training pipeline"""
        self.vectorizer = None
        self.model = None
        self.training_data = self._create_comprehensive_training_data()
        print(f"✅ Training data prepared: {len(self.training_data)} examples")
    
    def _create_comprehensive_training_data(self) -> List[Tuple[str, int]]:
        """Create comprehensive training dataset"""
        
        # Product-related examples (label = 1)
        product_texts = [
            # Tech products
            "Check out my new iPhone 15 Pro Max! Link in bio to buy now.",
            "Unboxing the latest Samsung Galaxy S24 Ultra. So cool!",
            "AirPods Pro review - are they worth the $249 price tag?",
            "New MacBook Air M2 first impressions. Get yours on apple.com",
            "Gaming setup reveal! RTX 4090 is a beast. Amazon link below.",
            
            # Fashion and lifestyle
            "My favorite Nike shoes for running. 30% off sale right now!",
            "Skincare routine that changed my life. Products linked in bio.",
            "Amazon fashion haul - under $50 finds! Swipe up for links.",
            "This makeup palette is everything! Use code SAVE20 for discount.",
            "Home decor haul from Target. Everything under $25!",
            
            # Shopping and deals
            "Found amazing deals on amazon.com - you need to see this!",
            "Black Friday haul! Saved over $200 on these products.",
            "Affiliate link in description - helps support the channel!",
            "Sponsored: Try this new product for 30% off with my code.",
            "Product review: Is this $100 gadget worth it? Let's find out.",
            
            # E-commerce and shopping
            "Shop with me at Sephora! New arrivals are incredible.",
            "Etsy finds that will blow your mind. Links in description.",
            "Walmart grocery haul on a budget. Under $50 for everything!",
            "Best Buy tech deals you can't miss. Limited time offers!",
            "Temu haul - cheap finds that actually work!",
            
            # Unboxing and reviews
            "Unboxing my Shein order - was it worth the hype?",
            "Product comparison: iPhone vs Samsung - which should you buy?",
            "Testing viral TikTok products so you don't have to.",
            "Honest review of this $20 Amazon gadget everyone's buying.",
            "First impressions of the new Tesla Model 3 accessories.",
            
            # Social commerce
            "DM me for the link to this amazing product!",
            "Comment 'LINK' and I'll send you the details!",
            "Tag a friend who needs this in their life!",
            "Get yours before they sell out - link in bio!",
            "Limited time offer - use my discount code SAVE15!",
        ]
        
        # Non-product examples (label = 0)
        non_product_texts = [
            # Lifestyle and personal
            "Just a regular day in my life. Nothing special happening.",
            "Beautiful sunset view from my balcony tonight.",
            "Cooking a delicious meal for dinner with family.",
            "Morning routine and healthy breakfast ideas for you.",
            "Travel vlog from my amazing trip to Japan last month.",
            
            # Educational content
            "Learning Python programming for data science projects.",
            "Quick tutorial on how to use Adobe Photoshop effectively.",
            "Math tips that will help you ace your exams.",
            "History lesson about ancient Rome and its culture.",
            "Science experiment you can do at home safely.",
            
            # Entertainment
            "Funny moments compilation from this week's adventures.",
            "Dancing to my favorite song - hope you enjoy!",
            "Singing cover of a popular song everyone loves.",
            "Comedy skit about everyday life situations we all face.",
            "Magic tricks that will amaze your friends and family.",
            
            # Personal stories
            "Sharing my journey of overcoming challenges in life.",
            "Motivational speech about following your dreams always.",
            "Story time: The most embarrassing thing that happened to me.",
            "Life update: What I've been up to lately.",
            "Reflecting on lessons learned this year so far.",
            
            # Hobbies and interests
            "My art process - painting a landscape from scratch.",
            "Gardening tips for beginners who want to start growing.",
            "Book recommendations for summer reading this year.",
            "Workout routine that keeps me motivated every day.",
            "Photography tips for capturing better portraits naturally.",
            
            # Random content
            "Exploring the ancient ruins of a historical site.",
            "Nature walk in the beautiful forest near my home.",
            "Meditation and mindfulness practices for inner peace.",
            "Climate change discussion and environmental awareness topics.",
            "Philosophy thoughts about the meaning of life itself.",
        ]
        
        # Combine and label the data
        training_data = []
        training_data.extend([(text, 1) for text in product_texts])
        training_data.extend([(text, 0) for text in non_product_texts])
        
        return training_data
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML model"""
        texts, labels = zip(*self.training_data)
        
        df = pd.DataFrame({
            'text': texts,
            'is_product': labels
        })
        
        print(f"📊 Dataset Statistics:")
        print(f"  Total examples: {len(df)}")
        print(f"  Product examples: {sum(df['is_product'])}")
        print(f"  Non-product examples: {len(df) - sum(df['is_product'])}")
        print(f"  Balance ratio: {sum(df['is_product']) / len(df):.1%} product")
        
        return df['text'].values, df['is_product'].values
    
    def create_optimized_vectorizer(self) -> TfidfVectorizer:
        """Create memory-optimized TF-IDF vectorizer"""
        return TfidfVectorizer(
            max_features=1000,      # Limit features for memory efficiency
            min_df=2,               # Remove very rare terms
            max_df=0.95,            # Remove very common terms
            stop_words='english',   # Remove common English words
            ngram_range=(1, 2),     # Include single words and bigrams
            lowercase=True,         # Normalize case
            strip_accents='ascii',  # Handle accented characters
            dtype=np.float32        # Use float32 for memory efficiency
        )
    
    def train_model(self) -> Tuple[LogisticRegression, TfidfVectorizer]:
        """Train the complete ML pipeline"""
        print("\n🤖 Training Product Detection Model...")
        
        # Prepare data
        X, y = self.prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and fit vectorizer
        print("📊 Creating TF-IDF features...")
        self.vectorizer = self.create_optimized_vectorizer()
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"  Feature matrix shape: {X_train_vec.shape}")
        print(f"  Memory usage: ~{X_train_vec.data.nbytes / 1024:.1f} KB")
        
        # Train model
        print("🧠 Training Logistic Regression...")
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',  # Handle class imbalance
            solver='liblinear',       # Good for small datasets
            C=1.0                     # Regularization strength
        )
        
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate model
        print("📈 Evaluating model performance...")
        y_pred = self.model.predict(X_test_vec)
        y_pred_proba = self.model.predict_proba(X_test_vec)
        
        print(f"\nTest Set Results:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['No Product', 'Product']))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_vec, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return self.model, self.vectorizer
    
    def optimize_model_size(self):
        """Optimize model for production deployment"""
        print("\n⚡ Optimizing model for production...")
        
        # Convert model coefficients to float32
        if hasattr(self.model, 'coef_'):
            self.model.coef_ = self.model.coef_.astype(np.float32)
        if hasattr(self.model, 'intercept_'):
            self.model.intercept_ = self.model.intercept_.astype(np.float32)
        
        print("✅ Model weights optimized to float32")
    
    def save_models(self, model_path="optimized_product_model.pkl", 
                   vectorizer_path="optimized_tfidf_vectorizer.pkl"):
        """Save trained models to disk"""
        print(f"\n💾 Saving models...")
        
        # Save model and vectorizer
        joblib.dump(self.model, model_path, compress=3)
        joblib.dump(self.vectorizer, vectorizer_path, compress=3)
        
        # Check file sizes
        model_size = os.path.getsize(model_path) / 1024  # KB
        vec_size = os.path.getsize(vectorizer_path) / 1024  # KB
        total_size = model_size + vec_size
        
        print(f"  Model size: {model_size:.1f} KB")
        print(f"  Vectorizer size: {vec_size:.1f} KB")
        print(f"  Total size: {total_size:.1f} KB (~{total_size/1024:.2f} MB)")
        
        return model_path, vectorizer_path
    
    def test_saved_model(self, model_path, vectorizer_path):
        """Test the saved model with new examples"""
        print(f"\n🧪 Testing saved model...")
        
        # Load models
        loaded_model = joblib.load(model_path)
        loaded_vectorizer = joblib.load(vectorizer_path)
        
        # Test examples
        test_examples = [
            "Amazing new gadget you absolutely need to see!",
            "Just sharing my peaceful morning coffee routine",
            "Unboxing my latest tech purchase from Amazon",
            "Beautiful nature photography tips and tricks",
            "This product changed my life - link in bio!",
            "Cooking dinner with fresh ingredients from garden",
            "Sponsored content: Try this skincare for 30% off",
            "Random thoughts about life and happiness today"
        ]
        
        print("Test Results:")
        for text in test_examples:
            # Predict
            text_vec = loaded_vectorizer.transform([text])
            prediction = loaded_model.predict(text_vec)[0]
            probability = loaded_model.predict_proba(text_vec)[0]
            
            result = "PRODUCT" if prediction == 1 else "NO PRODUCT"
            confidence = max(probability) * 100
            
            print(f"  '{text[:40]}...' → {result} ({confidence:.1f}%)")

def main():
    """Main training pipeline"""
    trainer = ProductDetectionTrainer()
    
    # Train the model
    model, vectorizer = trainer.train_model()
    
    # Optimize for production
    trainer.optimize_model_size()
    
    # Save models
    model_path, vectorizer_path = trainer.save_models()
    
    # Test saved models
    trainer.test_saved_model(model_path, vectorizer_path)
    
    print("\n✅ Training complete! Models ready for production use.")
    print(f"📁 Model files: {model_path}, {vectorizer_path}")

if __name__ == "__main__":
    main()