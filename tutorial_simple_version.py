#!/usr/bin/env python3
"""
Simple YouTube Shorts Product Filter Tutorial
Build this step by step to understand the core concepts
"""

import re
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class SimpleProductFilter:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)
    
    def detect_products(self, text):
        """Step 1: Simple product detection using regex"""
        patterns = [
            r'buy\s+now',
            r'shop\s+here', 
            r'link\s+in\s+bio',
            r'product\s+review',
            r'unboxing',
            r'amazon\.com',
            r'(\$|€|£)\d+',  # Price patterns
            r'discount|sale'
        ]
        
        found = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(pattern)
        return found
    
    def is_short_video(self, duration):
        """Step 2: Check if video is a Short (under 60 seconds)"""
        # YouTube duration format: PT1M30S (1 minute 30 seconds)
        duration_match = re.match(r"PT(?:(\d+)M)?(?:(\d+)S)?", duration)
        if duration_match:
            minutes = int(duration_match.group(1) or 0)
            seconds = int(duration_match.group(2) or 0)
            total_seconds = minutes * 60 + seconds
            return total_seconds <= 60
        return False
    
    def search_videos(self, query, max_results=10):
        """Step 3: Search YouTube for videos"""
        try:
            request = self.youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=max_results,
                order="date"
            )
            response = request.execute()
            return [item["id"]["videoId"] for item in response["items"]]
        except HttpError as e:
            print(f"Error searching videos: {e}")
            return []
    
    def get_video_details(self, video_id):
        """Step 4: Get detailed video information"""
        try:
            request = self.youtube.videos().list(
                part="snippet,contentDetails",
                id=video_id
            )
            response = request.execute()
            return response["items"][0] if response["items"] else None
        except HttpError as e:
            print(f"Error fetching video details: {e}")
            return None
    
    def filter_product_shorts(self, query):
        """Step 5: Main filtering logic"""
        print(f"🔍 Searching for: {query}")
        
        # Search for videos
        video_ids = self.search_videos(query)
        print(f"Found {len(video_ids)} videos")
        
        product_shorts = []
        
        for video_id in video_ids:
            # Get video details
            video = self.get_video_details(video_id)
            if not video:
                continue
            
            # Check if it's a Short
            duration = video.get("contentDetails", {}).get("duration", "")
            if not self.is_short_video(duration):
                continue
            
            # Check for product mentions
            title = video["snippet"].get("title", "")
            description = video["snippet"].get("description", "")
            text_to_analyze = f"{title} {description}"
            
            products = self.detect_products(text_to_analyze)
            
            if products:
                product_shorts.append({
                    "video_id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "detected_patterns": products
                })
        
        return product_shorts

# Example usage
if __name__ == "__main__":
    # You need to replace with your actual API key
    API_KEY = "YOUR_YOUTUBE_API_KEY"
    
    filter_system = SimpleProductFilter(API_KEY)
    results = filter_system.filter_product_shorts("tech review")
    
    print(f"\n✅ Found {len(results)} product-related Shorts:")
    for short in results:
        print(f"- {short['title']}")
        print(f"  URL: {short['url']}")
        print(f"  Patterns: {', '.join(short['detected_patterns'])}")
        print()