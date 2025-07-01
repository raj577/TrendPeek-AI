import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Dummy data for demonstration. In a real scenario, this would come from collected YouTube Shorts data.
# 'text' is the video description/comment, 'is_product' is 1 if it contains a product, 0 otherwise.
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

df = pd.DataFrame(DATA)

X = df["text"]
y = df["is_product"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use TF-IDF Vectorizer for text feature extraction
# min_df is set to 1 to allow more terms, as the dataset is small
vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Logistic Regression model
# Logistic Regression is chosen for its simplicity and low resource usage
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vec)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(model, "product_detection_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")

# Example of how to use the saved model for prediction
def predict_product_presence(text_to_predict):
    loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    loaded_model = joblib.load("product_detection_model.pkl")

    text_vec = loaded_vectorizer.transform([text_to_predict])
    prediction = loaded_model.predict(text_vec)
    return "Product detected" if prediction[0] == 1 else "No product detected"

if __name__ == '__main__':
    print("\n--- Prediction Examples ---")
    print(f"'This is a test for a new gadget.' -> {predict_product_presence('This is a test for a new gadget.')}")
    print(f"'Just talking about my day.' -> {predict_product_presence('Just talking about my day.')}")
    print(f"'Buy this amazing product now!' -> {predict_product_presence('Buy this amazing product now!')}")


