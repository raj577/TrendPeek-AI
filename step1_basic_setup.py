#!/usr/bin/env python3
"""
Step 1: Basic YouTube API Setup and Video Detection
Learn the fundamentals of YouTube API integration
"""

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re
from datetime import datetime, timedelta, timezone

class BasicYouTubeFilter:
    def __init__(self, api_key):
        """Initialize YouTube API client"""
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)
        print("✅ YouTube API client initialized")
    
    def search_videos(self, query, max_results=10):
        """Step 1: Search for videos using YouTube API"""
        try:
            # Calculate date 4 days ago for recent content
            published_after = (
                datetime.now(timezone.utc) - timedelta(days=4)
            ).isoformat(timespec="seconds").replace("+00:00", "Z")
            
            request = self.youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=max_results,
                order="date",
                publishedAfter=published_after
            )
            response = request.execute()
            
            video_ids = [item["id"]["videoId"] for item in response["items"]]
            print(f"🔍 Found {len(video_ids)} videos for query: '{query}'")
            return video_ids
            
        except HttpError as e:
            print(f"❌ API Error: {e}")
            return []
    
    def get_video_details(self, video_id):
        """Step 2: Get detailed video information"""
        try:
            request = self.youtube.videos().list(
                part="snippet,contentDetails",
                id=video_id
            )
            response = request.execute()
            return response["items"][0] if response["items"] else None
        except HttpError as e:
            print(f"❌ Error fetching video {video_id}: {e}")
            return None
    
    def is_short_video(self, video_details):
        """Step 3: Identify if video is a Short (under 60 seconds)"""
        duration = video_details.get("contentDetails", {}).get("duration", "")
        
        # Parse YouTube duration format: PT1M30S = 1 minute 30 seconds
        duration_match = re.match(r"PT(?:(\d+)M)?(?:(\d+)S)?", duration)
        if duration_match:
            minutes = int(duration_match.group(1) or 0)
            seconds = int(duration_match.group(2) or 0)
            total_seconds = minutes * 60 + seconds
            return total_seconds <= 60
        return False
    
    def basic_product_detection(self, text):
        """Step 4: Simple product detection using keywords"""
        product_keywords = [
            'buy', 'shop', 'product', 'review', 'unboxing', 
            'amazon', 'deal', 'sale', 'discount', 'link in bio'
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in product_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def analyze_shorts(self, query, max_results=10):
        """Step 5: Complete analysis pipeline"""
        print(f"\n🚀 Starting analysis for: '{query}'")
        
        # Search for videos
        video_ids = self.search_videos(query, max_results)
        if not video_ids:
            return []
        
        shorts_with_products = []
        
        for video_id in video_ids:
            # Get video details
            video = self.get_video_details(video_id)
            if not video:
                continue
            
            # Check if it's a Short
            if not self.is_short_video(video):
                continue
            
            # Extract text content
            title = video["snippet"].get("title", "")
            description = video["snippet"].get("description", "")
            combined_text = f"{title} {description}"
            
            # Detect products
            product_keywords = self.basic_product_detection(combined_text)
            
            if product_keywords:
                shorts_with_products.append({
                    "video_id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "description": description[:100] + "..." if len(description) > 100 else description,
                    "detected_keywords": product_keywords
                })
        
        print(f"✅ Found {len(shorts_with_products)} Shorts with product mentions")
        return shorts_with_products

# Example usage and testing
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = "YOUR_YOUTUBE_API_KEY_HERE"
    
    # Initialize filter
    filter_system = BasicYouTubeFilter(API_KEY)
    
    # Test with different queries
    test_queries = ["tech review", "amazon finds", "product unboxing"]
    
    for query in test_queries:
        results = filter_system.analyze_shorts(query, max_results=5)
        
        print(f"\n📊 Results for '{query}':")
        for i, short in enumerate(results, 1):
            print(f"{i}. {short['title']}")
            print(f"   Keywords: {', '.join(short['detected_keywords'])}")
            print(f"   URL: {short['url']}")
            print()
        
        if not results:
            print("   No product-related Shorts found.")
        
        print("-" * 50)