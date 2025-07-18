#!/usr/bin/env python3
"""
Step 4: Complete YouTube Shorts Product Filter System
Integrate all components into a production-ready system
"""

import os
import re
import json
import joblib
import time
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import List, Dict, Optional
import psutil

class YouTubeShortsProductFilter:
    def __init__(self, api_key: str, 
                 model_path: str = "optimized_product_model.pkl",
                 vectorizer_path: str = "optimized_tfidf_vectorizer.pkl"):
        """
        Initialize the complete product filter system
        
        Args:
            api_key: YouTube Data API v3 key
            model_path: Path to trained ML model
            vectorizer_path: Path to TF-IDF vectorizer
        """
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)
        
        # Load ML components
        self._load_ml_models(model_path, vectorizer_path)
        
        # Initialize regex patterns
        self._init_regex_patterns()
        
        # Performance tracking
        self.stats = {
            'videos_processed': 0,
            'shorts_found': 0,
            'products_detected': 0,
            'api_calls': 0,
            'processing_time': 0
        }
        
        print("✅ YouTube Shorts Product Filter initialized")
    
    def _load_ml_models(self, model_path: str, vectorizer_path: str):
        """Load trained ML model and vectorizer"""
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print("✅ ML model and vectorizer loaded successfully")
        except FileNotFoundError as e:
            print(f"⚠️ ML model files not found: {e}")
            print("   Only regex detection will be used")
            self.model = None
            self.vectorizer = None
    
    def _init_regex_patterns(self):
        """Initialize comprehensive regex patterns"""
        self.regex_patterns = [
            # Brand patterns
            r"iphone\s*\d+",
            r"galaxy\s*s\d+",
            r"macbook\s*(pro|air)?",
            r"airpods?\s*(pro|max)?",
            r"nike",
            r"adidas",
            
            # Shopping intent
            r"buy\s+now",
            r"shop\s+here",
            r"get\s+yours?",
            r"order\s+now",
            r"link\s+in\s+bio",
            r"swipe\s+up",
            
            # Content types
            r"product\s+review",
            r"unboxing",
            r"haul",
            r"first\s+impressions?",
            
            # E-commerce
            r"amazon\.com",
            r"amzn\.to",
            r"etsy\.com",
            r"shopify\.com",
            r"walmart\.com",
            r"target\.com",
            r"bestbuy\.com",
            
            # Pricing
            r"(\$|€|£)\d+(\.\d{2})?",
            r"discount",
            r"sale",
            r"deal",
            r"\d+%\s*off",
            
            # Affiliate
            r"affiliate\s+link",
            r"sponsored",
            r"#ad",
            r"promo\s*code"
        ]
    
    def detect_products_regex(self, text: str) -> List[str]:
        """Detect products using regex patterns"""
        found_patterns = []
        text_lower = text.lower()
        
        for pattern in self.regex_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                found_patterns.append(pattern)
        
        return found_patterns
    
    def detect_products_ml(self, text: str) -> Dict:
        """Detect products using ML model"""
        if self.model is None or self.vectorizer is None:
            return {"prediction": False, "confidence": 0.0}
        
        try:
            # Vectorize text
            text_vec = self.vectorizer.transform([text])
            
            # Get prediction and probability
            prediction = self.model.predict(text_vec)[0]
            probabilities = self.model.predict_proba(text_vec)[0]
            confidence = max(probabilities)
            
            return {
                "prediction": bool(prediction),
                "confidence": float(confidence)
            }
        except Exception as e:
            print(f"⚠️ ML prediction error: {e}")
            return {"prediction": False, "confidence": 0.0}
    
    def is_short_video(self, video_details: Dict) -> bool:
        """Check if video is a YouTube Short (under 60 seconds)"""
        duration = video_details.get("contentDetails", {}).get("duration", "")
        
        # Parse YouTube duration format: PT1M30S
        duration_match = re.match(r"PT(?:(\d+)M)?(?:(\d+)S)?", duration)
        if duration_match:
            minutes = int(duration_match.group(1) or 0)
            seconds = int(duration_match.group(2) or 0)
            total_seconds = minutes * 60 + seconds
            return total_seconds <= 60
        return False
    
    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """Get detailed video information from YouTube API"""
        try:
            request = self.youtube.videos().list(
                part="snippet,contentDetails",
                id=video_id
            )
            response = request.execute()
            self.stats['api_calls'] += 1
            
            return response["items"][0] if response["items"] else None
        except HttpError as e:
            print(f"❌ Error fetching video {video_id}: {e}")
            return None
    
    def get_video_comments(self, video_id: str, max_results: int = 5) -> List[str]:
        """Get top comments for a video"""
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                order="relevance"
            )
            response = request.execute()
            self.stats['api_calls'] += 1
            
            comments = []
            for item in response["items"]:
                comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment_text)
            
            return comments
        except HttpError as e:
            # Comments might be disabled
            return []
    
    def search_videos(self, query: str, max_results: int = 50, 
                     published_after: Optional[str] = None) -> List[str]:
        """Search for videos using YouTube API"""
        if not published_after:
            # Default to last 4 days
            published_after = (
                datetime.now(timezone.utc) - timedelta(days=4)
            ).isoformat(timespec="seconds").replace("+00:00", "Z")
        
        try:
            request = self.youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=max_results,
                order="date",
                publishedAfter=published_after
            )
            response = request.execute()
            self.stats['api_calls'] += 1
            
            video_ids = [item["id"]["videoId"] for item in response["items"]]
            return video_ids
        except HttpError as e:
            print(f"❌ Error searching videos: {e}")
            return []
    
    def analyze_video_content(self, video_id: str) -> Optional[Dict]:
        """Complete analysis of a single video"""
        start_time = time.time()
        
        # Get video details
        video_details = self.get_video_details(video_id)
        if not video_details:
            return None
        
        # Check if it's a Short
        if not self.is_short_video(video_details):
            return None
        
        self.stats['shorts_found'] += 1
        
        # Extract text content
        title = video_details["snippet"].get("title", "")
        description = video_details["snippet"].get("description", "")
        
        # Get comments (optional, can be slow)
        comments = self.get_video_comments(video_id, max_results=3)
        
        # Combine all text
        all_text = f"{title} {description} {' '.join(comments)}"
        
        # Detect products using both methods
        regex_patterns = self.detect_products_regex(all_text)
        ml_result = self.detect_products_ml(all_text)
        
        # Determine if product is detected
        has_product = len(regex_patterns) > 0 or ml_result["prediction"]
        
        if has_product:
            self.stats['products_detected'] += 1
        
        processing_time = time.time() - start_time
        self.stats['processing_time'] += processing_time
        
        return {
            "video_id": video_id,
            "title": title,
            "description": description[:200] + "..." if len(description) > 200 else description,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "has_product": has_product,
            "regex_patterns": regex_patterns,
            "ml_prediction": ml_result["prediction"],
            "ml_confidence": ml_result["confidence"],
            "processing_time": processing_time,
            "comment_count": len(comments)
        }
    
    def filter_shorts_batch(self, video_ids: List[str]) -> List[Dict]:
        """Process multiple videos efficiently"""
        results = []
        
        print(f"🔍 Analyzing {len(video_ids)} videos...")
        
        for i, video_id in enumerate(video_ids, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(video_ids)} videos processed")
            
            result = self.analyze_video_content(video_id)
            if result and result["has_product"]:
                results.append(result)
            
            self.stats['videos_processed'] += 1
            
            # Rate limiting - be nice to YouTube API
            time.sleep(0.1)
        
        return results
    
    def run_complete_filter(self, search_queries: List[str], 
                           max_results_per_query: int = 20) -> List[Dict]:
        """Run complete filtering pipeline"""
        print(f"🚀 Starting complete filter with {len(search_queries)} queries")
        
        all_results = []
        processed_video_ids = set()
        
        for query in search_queries:
            print(f"\n🔍 Processing query: '{query}'")
            
            # Search for videos
            video_ids = self.search_videos(query, max_results=max_results_per_query)
            
            # Remove duplicates
            new_video_ids = [vid for vid in video_ids if vid not in processed_video_ids]
            
            if not new_video_ids:
                print("  No new videos found")
                continue
            
            print(f"  Found {len(new_video_ids)} new videos to analyze")
            
            # Analyze videos
            query_results = self.filter_shorts_batch(new_video_ids)
            all_results.extend(query_results)
            
            # Track processed videos
            processed_video_ids.update(new_video_ids)
            
            print(f"  Found {len(query_results)} product-related Shorts")
        
        return all_results
    
    def get_performance_stats(self) -> Dict:
        """Get system performance statistics"""
        return {
            **self.stats,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "avg_processing_time": self.stats['processing_time'] / max(self.stats['videos_processed'], 1)
        }
    
    def export_results(self, results: List[Dict], filename: str = None) -> str:
        """Export results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"youtube_shorts_filter_results_{timestamp}.json"
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "total_results": len(results),
            "performance_stats": self.get_performance_stats(),
            "results": results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Results exported to: {filename}")
        return filename

# Example usage and testing
def main():
    """Demonstration of the complete system"""
    # Replace with your actual API key
    API_KEY = "YOUR_YOUTUBE_API_KEY_HERE"
    
    # Initialize the system
    filter_system = YouTubeShortsProductFilter(API_KEY)
    
    # Define search queries
    search_queries = [
        "amazon finds",
        "tech review",
        "product unboxing",
        "must have gadgets"
    ]
    
    print("🎯 YouTube Shorts Product Filter - Complete System Demo")
    print(f"Search queries: {', '.join(search_queries)}")
    
    # Run the complete filter
    results = filter_system.run_complete_filter(search_queries, max_results_per_query=10)
    
    # Display results
    print(f"\n✅ Analysis Complete!")
    print(f"Found {len(results)} product-related YouTube Shorts")
    
    # Show performance stats
    stats = filter_system.get_performance_stats()
    print(f"\n📊 Performance Statistics:")
    print(f"  Videos processed: {stats['videos_processed']}")
    print(f"  Shorts found: {stats['shorts_found']}")
    print(f"  Products detected: {stats['products_detected']}")
    print(f"  API calls made: {stats['api_calls']}")
    print(f"  Total processing time: {stats['processing_time']:.2f}s")
    print(f"  Average time per video: {stats['avg_processing_time']:.3f}s")
    print(f"  Memory usage: {stats['memory_usage_mb']:.1f} MB")
    
    # Show sample results
    if results:
        print(f"\n🎬 Sample Results:")
        for i, result in enumerate(results[:3], 1):
            print(f"{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Regex patterns: {len(result['regex_patterns'])}")
            print(f"   ML confidence: {result['ml_confidence']:.2%}")
            print()
    
    # Export results
    filename = filter_system.export_results(results)
    
    print(f"🎉 Demo complete! Check {filename} for detailed results.")

if __name__ == "__main__":
    main()