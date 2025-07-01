import os
import re
import json
import joblib
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env file

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise ValueError("Missing YouTube API key! Please check your .env file.")


class YouTubeShortsProductFilter:
    def __init__(self, api_key, model_path="product_detection_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
        """
        Initialize the YouTube Shorts Product Filter.

        Args:
            api_key (str): YouTube Data API key
            model_path (str): Path to the trained ML model
            vectorizer_path (str): Path to the TF-IDF vectorizer
        """
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)

        # Load the trained ML model and vectorizer
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print("ML model and vectorizer loaded successfully.")
        except FileNotFoundError:
            print("Warning: ML model or vectorizer not found. Only regex detection will be used.")
            self.model = None
            self.vectorizer = None

    def detect_products_regex(self, text):
        """
        Detects product mentions in text using regular expressions.
        """
        patterns = [
            r"iphone\\s*\\d+",
            r"galaxy\\s*s\\d+",
            r"nike",
            r"adidas",
            r"buy\\s+now",
            r"shop\\s+here",
            r"link\\s+in\\s+bio",
            r"product\\s+review",
            r"unboxing",
            r"amazon\\.com",
            r"etsy\\.com",
            r"shopify\\.com",
            r"walmart\\.com",
            r"target\\.com",
            r"bestbuy\\.com",
            r"(\\$|€|£)\\d+(\\.\\d{2})?",
            r"discount",
            r"sale"
        ]

        found_products = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_products.append(pattern)
        return found_products

    def detect_products_ml(self, text):
        """
        Detects product mentions using the trained ML model.
        """
        if self.model is None or self.vectorizer is None:
            return False

        text_vec = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vec)
        return prediction[0] == 1

    def is_short_video(self, video_details):
        """
        Determines if a video is a YouTube Short based on duration.
        """
        duration = video_details.get("contentDetails", {}).get("duration", "")

        # Parse ISO 8601 duration format (e.g., PT1M30S for 1 minute 30 seconds)
        import re
        duration_match = re.match(r"PT(?:(\\d+)M)?(?:(\\d+)S)?", duration)
        if duration_match:
            minutes = int(duration_match.group(1) or 0)
            seconds = int(duration_match.group(2) or 0)
            total_seconds = minutes * 60 + seconds
            return total_seconds <= 60  # Shorts are typically 60 seconds or less

        return False

    def get_video_details(self, video_id):
        """
        Retrieves video details from YouTube API.
        """
        try:
            request = self.youtube.videos().list(
                part="snippet,contentDetails",
                id=video_id
            )
            response = request.execute()

            if response["items"]:
                return response["items"][0]
            else:
                return None
        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            return None

    def get_video_comments(self, video_id, max_results=4):
        """
        Retrieves comments for a video from YouTube API.
        """
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                order="relevance"
            )
            response = request.execute()

            comments = []
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            return comments
        except HttpError as e:
            print(f"An HTTP error occurred while fetching comments: {e}")
            return []

    def search_videos(self, query, max_results=50):
        """
        Searches for videos using YouTube API.
        """
        try:
            request = self.youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=max_results,
                order="relevance"
            )
            response = request.execute()

            video_ids = []
            for item in response["items"]:
                video_ids.append(item["id"]["videoId"])

            return video_ids
        except HttpError as e:
            print(f"An HTTP error occurred during search: {e}")
            return []

    def filter_shorts_with_products(self, video_ids):
        """
        Filters YouTube Shorts that contain product mentions.
        """
        filtered_shorts = []

        for video_id in video_ids:
            video_details = self.get_video_details(video_id)

            if video_details is None:
                continue

            # Check if it's a Short
            if not self.is_short_video(video_details):
                continue

            # Extract text data
            title = video_details["snippet"].get("title", "")
            description = video_details["snippet"].get("description", "")

            # Get comments
            comments = self.get_video_comments(video_id)
            all_text = f"{title} {description} {" ".join(comments)}"

            # Detect products using regex
            regex_products = self.detect_products_regex(all_text)

            # Detect products using ML model
            ml_detection = self.detect_products_ml(all_text)

            # If either method detects products, include the video
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

    def run_filter(self, search_queries, max_results=50):
        """
        Main method to run the filtering process.
        Accepts a list of search queries.
        """
        all_filtered_shorts = []
        processed_video_ids = set()  # To avoid duplicate processing of videos

        for query in search_queries:
            print(f"Searching for videos with query: \'{query}\'")
            video_ids = self.search_videos(query, max_results)

            # Filter out already processed video IDs
            new_video_ids = [vid for vid in video_ids if vid not in processed_video_ids]

            if new_video_ids:
                print(
                    f"Found {len(new_video_ids)} new videos for query \'{query}\'. Filtering for Shorts with products...")
                filtered_shorts_for_query = self.filter_shorts_with_products(new_video_ids)
                all_filtered_shorts.extend(filtered_shorts_for_query)
                processed_video_ids.update(new_video_ids)
            else:
                print(f"No new videos found for query \'{query}\'.")

        print(f"\nTotal found {len(all_filtered_shorts)} Shorts with product mentions across all queries.")
        return all_filtered_shorts


def main():
    # Note: You need to replace 'YOUR_API_KEY' with an actual YouTube Data API key
    # API_KEY = "AIzaSyDUlbs_7Vc1Jj1stkyLyZxvk8w6mE-MHng"

    if API_KEY == "YOUR_API_KEY":
        print("Please set a valid YouTube Data API key in the API_KEY variable.")
        return

    # Initialize the filter
    filter_system = YouTubeShortsProductFilter(API_KEY)

    # Example usage with a list of search queries
    search_queries = ["product review", "unboxing", "tech haul," "amazon finds",
   # "gadgets under 500",
    "must have products",
    "kitchen hacks",
   #  "cool gadgets",
   #  "viral amazon products",
   #  #"top amazon gadgets",
   #  "best amazon products",
   #  "useful home items",
   #  "amazon must haves",
   #  "tech gadgets amazon",
   #  #"amazon haul",
   #  "tiktok amazon products",
   #  #"amazon gadgets 2024",
   #  #"amazon home gadgets",
   #  "cool stuff on amazon",
   #  "unique amazon finds",
   #  "amazon gift ideas",
   #  "budget gadgets",
   #  "amazon tools",
   #  "affordable tech amazon",
   #  "amazon electronics under 1000",
   #  "amazon beauty products",
   #  "home organization amazon",
   #  "smart home gadgets",
   # # "amazon hacks",
   #  "cheap amazon finds",
   #  "mini gadgets amazon",
   #  "best kitchen gadgets"
                      ]
    results = filter_system.run_filter(search_queries, max_results=20)

    # Display results
    for i, short in enumerate(results, 1):
        print(f"\n{i}. {short["title"]}")
        print(f"   URL: {short["url"]}")
        print(f"   Regex patterns found: {short["regex_patterns"]}")
        print(f"   ML detection: {short["ml_detection"]}")
        print(f"   Description: {short["description"]}")

    # Save results to JSON file
    with open("filtered_shorts.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to 'filtered_shorts.json'")


if __name__ == "__main__":
    main()

