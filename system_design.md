
# System Design for YouTube Shorts Product Filtering

## 1. Introduction

This document outlines the system design for filtering YouTube Shorts based on the presence of products. The system will leverage the YouTube Data API to access video metadata and comments, employ regular expressions for initial product identification, and utilize a lightweight machine learning model for more sophisticated product detection, optimized for low RAM environments.

## 2. Data Sources and Collection Strategy

The primary data source will be the YouTube Data API v3. While there isn't a direct API endpoint to specifically filter for YouTube Shorts, we can identify Shorts by their aspect ratio (typically 9:16 vertical) or by checking for `#shorts` in the title or description. However, the most reliable method for identifying Shorts programmatically is to check the video's dimensions after retrieval, or to rely on the `contentDetails.duration` field (Shorts are typically under 60 seconds).

### 2.1. Video Metadata Retrieval

We will use the `videos.list` endpoint of the YouTube Data API to retrieve video metadata. Key fields to extract for product detection include:

*   `snippet.title`: Video title, which may contain product names or brands.
*   `snippet.description`: Video description, a rich source of product mentions, links, and hashtags.
*   `snippet.tags`: Tags associated with the video, potentially containing product-related keywords.
*   `contentDetails.duration`: To help identify Shorts (videos under 60 seconds).

### 2.2. Comment Thread Retrieval

Product mentions can also occur in the comments section. We will use the `commentThreads.list` endpoint to retrieve top-level comments and their replies for each video. We will focus on the `snippet.textDisplay` field of comments for product detection.

### 2.3. Data Collection Workflow

1.  **Identify Potential Shorts:** Initially, we can search for videos using keywords related to product reviews, hauls, unboxings, or general shopping terms. We can also filter by upload date to focus on recent content. Alternatively, if a list of channel IDs is available, we can iterate through their uploaded videos.
2.  **Filter for Shorts:** After retrieving video details, we will programmatically check if the video is a Short. This can be done by examining the video's duration (less than 60 seconds) and potentially its aspect ratio (though this might require additional processing or a separate API call if not directly available in the metadata).
3.  **Extract Relevant Text:** For each identified Short, extract the `title`, `description`, and all available `comment` texts.
4.  **Store Raw Data:** Store the extracted raw text data in a structured format (e.g., JSON or a simple database) for further processing and model training.

## 3. Product Detection Approaches

We will employ a hybrid approach combining regex-based pattern matching for explicit product mentions and a lightweight machine learning model for more nuanced detection.

### 3.1. Regex-based Product Detection

Regular expressions will be used to identify common patterns associated with product mentions, such as:

*   Brand names (e.g., 


Apple, Samsung, Nike)
*   Product names (e.g., iPhone 15, Galaxy S24, Air Jordan 1)
*   Keywords like "buy," "shop," "link in bio," "product review," "unboxing."
*   URLs pointing to e-commerce sites (e.g., amazon.com, shopify.com).

This approach is fast and efficient for initial filtering but may have limitations in identifying implicit product mentions or variations in naming conventions.

### 3.2. Lightweight Machine Learning Model

For more advanced product detection, we will train a lightweight machine learning model. Given the low RAM constraint, we will consider models suitable for edge devices or environments with limited resources. Potential candidates include:

*   **Text Classification Models:** Using techniques like TF-IDF or word embeddings combined with simple classifiers (e.g., Naive Bayes, Logistic Regression, or a small, optimized neural network like a shallow CNN or a simple Recurrent Neural Network (RNN) with GRU/LSTM units).
*   **Named Entity Recognition (NER):** Training a custom NER model to identify product names, brands, and related entities within the text. This would require a labeled dataset of product mentions.

#### 3.2.1. Model Training Data

The training data for the machine learning model will be derived from the collected YouTube Shorts metadata and comments. This will involve:

1.  **Annotation:** Manually annotating a subset of the collected text data to label product mentions. This is a crucial step for supervised learning.
2.  **Feature Extraction:** Converting text into numerical features that the model can understand. For low-RAM models, simpler features like TF-IDF or pre-trained, small word embeddings (e.g., GloVe, FastText) would be preferred over large transformer models.

#### 3.2.2. Model Optimization for Low RAM

To ensure the model runs efficiently on low RAM, we will focus on:

*   **Model Architecture:** Choosing models with fewer parameters and simpler structures.
*   **Quantization:** Reducing the precision of model weights (e.g., from float32 to int8) to decrease memory footprint and speed up inference.
*   **Pruning:** Removing less important connections or neurons from the neural network to reduce model size.
*   **Knowledge Distillation:** Training a smaller model to mimic the behavior of a larger, more complex model.

## 4. System Architecture

The system will consist of several modular components:

```mermaid
graph TD
    A[YouTube Data API] --> B{Data Ingestion Module}
    B --> C[Raw Data Storage (JSON/DB)]
    C --> D{Preprocessing Module}
    D --> E[Regex Product Detector]
    D --> F[ML Product Detector]
    E --> G[Filtered Shorts Output]
    F --> G
    G --> H[Reporting/Visualization]
```

### 4.1. Data Ingestion Module

This module will be responsible for interacting with the YouTube Data API, fetching video metadata and comments, and storing them in a raw data storage. It will handle API key management, rate limiting, and error handling.

### 4.2. Preprocessing Module

This module will clean and normalize the raw text data (titles, descriptions, comments). This includes:

*   Lowercasing text.
*   Removing punctuation and special characters.
*   Tokenization (splitting text into words).
*   Stop word removal (optional, depending on ML model).
*   Lemmatization/Stemming (optional).

### 4.3. Regex Product Detector

This module will apply the predefined regular expression patterns to the preprocessed text data to identify explicit product mentions. It will output a flag indicating the presence of products and potentially the identified product names.

### 4.4. ML Product Detector

This module will load the trained lightweight machine learning model and perform inference on the preprocessed text data. It will output a probability score or a binary classification indicating the likelihood of product presence.

### 4.5. Filtered Shorts Output

This module will combine the results from the Regex and ML detectors. It will apply a configurable threshold to determine if a Short contains products and then output the filtered list of YouTube Shorts. The output format can be a CSV, JSON, or direct integration with a display interface.

### 4.6. Reporting/Visualization

An optional module for generating reports or visualizations of the filtered Shorts, including statistics on product categories, brands, and detection accuracy.

## 5. Technology Stack

*   **Programming Language:** Python (due to its rich ecosystem for data science and machine learning).
*   **API Interaction:** `google-api-python-client` library for YouTube Data API.
*   **Regex:** Python's built-in `re` module.
*   **Machine Learning:** `scikit-learn` for traditional ML models, or `TensorFlow Lite`/`PyTorch Mobile` for optimized neural networks.
*   **Data Storage:** Simple JSON files for raw data, or SQLite for a lightweight database.

## 6. Future Considerations

*   **Real-time Processing:** Adapting the system for real-time or near real-time filtering of new YouTube Shorts.
*   **Visual Product Detection:** Integrating computer vision techniques to detect products directly from video frames. This would significantly increase complexity and resource requirements but offer higher accuracy.
*   **User Interface:** Developing a simple web interface for users to input search queries and view filtered results.

## 7. References

[1] YouTube Data API Overview: [https://developers.google.com/youtube/v3](https://developers.google.com/youtube/v3)
[2] YouTube Data API - Videos: list: [https://developers.google.com/youtube/v3/docs/videos/list](https://developers.google.com/youtube/v3/docs/videos/list)
[3] YouTube Data API - CommentThreads: list: [https://developers.google.com/youtube/v3/docs/commentThreads/list](https://developers.google.com/youtube/v3/docs/commentThreads/list)
[4] TensorFlow Lite: [https://www.tensorflow.org/lite](https://www.tensorflow.org/lite)
[5] PyTorch Mobile: [https://pytorch.org/mobile/](https://pytorch.org/mobile/)


