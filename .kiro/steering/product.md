# Product Overview

## YouTube Shorts Product Filter (TrendPeek-AI)

A lightweight system that filters YouTube Shorts for product mentions using a hybrid approach combining regex patterns and machine learning. The system is optimized for low-RAM environments and designed for product scouting, marketing research, and eCommerce lead generation.

### Key Features
- **Dual Detection**: Combines regex pattern matching with ML classification
- **Lightweight ML**: Uses Logistic Regression + TF-IDF (0.25 MB RAM usage)
- **High Performance**: Processes 100K+ texts/second
- **YouTube API Integration**: Leverages YouTube Data API v3
- **Streamlit Web Interface**: User-friendly web app for filtering and visualization

### Target Use Cases
- Product discovery and trend analysis
- Marketing research and competitor analysis
- eCommerce lead generation
- Content creator product placement analysis
- Brand mention monitoring

### Performance Characteristics
- RAM Usage: ~0.25 MB for ML model
- Processing Speed: 100K+ texts/second
- Model Size: 17% optimized from original
- API Quota: Dependent on YouTube Data API v3 limits