import os
import re
import json
import joblib
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# load_dotenv()
# API_KEY = os.getenv("YOUTUBE_API_KEY")
# if not API_KEY:
#     raise ValueError("Missing YouTube API key! Please check your .env file.")

import streamlit as st

API_KEY = st.secrets["API_KEY"]

class YouTubeShortsProductFilter:
    def __init__(self, api_key, model_path="product_detection_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)

        # Load the trained ML model and vectorizer
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print("✅ ML model and vectorizer loaded successfully.")
        except FileNotFoundError:
            print("⚠️ ML model or vectorizer not found. Only regex detection will be used.")
            self.model = None
            self.vectorizer = None

    def detect_products_regex(self, text):
        patterns = [
            r"iphone\s*\d+",
            r"galaxy\s*s\d+",
            r"nike",
            r"adidas",
            r"buy\s+now",
            r"shop\s+here",
            r"link\s+in\s+bio",
            r"product\s+review",
            r"unboxing",
            r"amazon\.com",
            r"etsy\.com",
            r"shopify\.com",
            r"walmart\.com",
            r"target\.com",
            r"bestbuy\.com",
            r"(\$|€|£)\d+(\.\d{2})?",
            r"discount",
            r"sale"
        ]

        found_products = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_products.append(pattern)
        return found_products

    def detect_products_ml(self, text):
        if self.model is None or self.vectorizer is None:
            return False
        text_vec = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vec)
        return prediction[0] == 1

    def is_short_video(self, video_details):
        duration = video_details.get("contentDetails", {}).get("duration", "")
        duration_match = re.match(r"PT(?:(\d+)M)?(?:(\d+)S)?", duration)
        if duration_match:
            minutes = int(duration_match.group(1) or 0)
            seconds = int(duration_match.group(2) or 0)
            total_seconds = minutes * 60 + seconds
            return total_seconds <= 60
        return False

    def get_video_details(self, video_id):
        try:
            request = self.youtube.videos().list(
                part="snippet,contentDetails",
                id=video_id
            )
            response = request.execute()
            return response["items"][0] if response["items"] else None
        except HttpError as e:
            print(f"❌ Error fetching video details: {e}")
            return None

    def get_video_comments(self, video_id, max_results=4):
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                order="relevance"
            )
            response = request.execute()
            return [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in response["items"]]
        except HttpError as e:
            print(f"❌ Error fetching comments: {e}")
            return []

    def search_videos(self, query, max_results=50, published_after=None):
        if not published_after:
            published_after = self._iso_4_days_ago()

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
            return [item["id"]["videoId"] for item in response["items"]]
        except HttpError as e:
            print(f"❌ Error searching videos: {e}")
            return []

    def filter_shorts_with_products(self, video_ids):
        filtered_shorts = []

        for video_id in video_ids:
            video_details = self.get_video_details(video_id)
            if not video_details or not self.is_short_video(video_details):
                continue

            title = video_details["snippet"].get("title", "")
            description = video_details["snippet"].get("description", "")
            comments = self.get_video_comments(video_id)
            all_text = f"{title} {description} {' '.join(comments)}"

            regex_products = self.detect_products_regex(all_text)
            ml_detection = self.detect_products_ml(all_text)

            if regex_products or ml_detection:
                filtered_shorts.append({
                    "video_id": video_id,
                    "title": title,
                    "description": description[:200] + "..." if len(description) > 200 else description,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "regex_patterns": regex_products,
                    "ml_detection": str(ml_detection)
                })

        return filtered_shorts

    def run_filter(self, search_queries, max_results=50, days=4):
        all_filtered_shorts = []
        processed_ids = set()

        published_after = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat(timespec="seconds").replace("+00:00", "Z")

        for query in search_queries:
            print(f"🔍 Searching '{query}' (after {published_after})")
            video_ids = self.search_videos(query, max_results=max_results, published_after=published_after)
            new_ids = [vid for vid in video_ids if vid not in processed_ids]
            if not new_ids:
                print("No new videos found.")
                continue
            print(f"🎯 {len(new_ids)} new videos, filtering…")
            filtered = self.filter_shorts_with_products(new_ids)
            all_filtered_shorts.extend(filtered)
            processed_ids.update(new_ids)

        print(f"\n✅ Total filtered Shorts with product mentions: {len(all_filtered_shorts)}")
        return all_filtered_shorts

    def _iso_4_days_ago(self):
        dt = datetime.now(timezone.utc) - timedelta(days=4)
        return dt.isoformat(timespec="seconds").replace("+00:00", "Z")
