import json
from youtube_shorts_filter import YouTubeShortsProductFilter

# Demo script that simulates the filtering process without requiring a real API key
class DemoYouTubeShortsFilter(YouTubeShortsProductFilter):
    def __init__(self):
        # Initialize without API key for demo purposes
        self.api_key = "demo_key"
        self.youtube = None
        
        # Load the trained ML model and vectorizer
        try:
            import joblib
            self.model = joblib.load("product_detection_model.pkl")
            self.vectorizer = joblib.load("tfidf_vectorizer.pkl")
            print("ML model and vectorizer loaded successfully.")
        except FileNotFoundError:
            print("Warning: ML model or vectorizer not found. Only regex detection will be used.")
            self.model = None
            self.vectorizer = None
    
    def get_demo_video_data(self):
        """
        Returns demo video data simulating YouTube Shorts
        """
        return [
            {
                'video_id': 'demo1',
                'title': 'Unboxing the new iPhone 15 Pro Max! #shorts',
                'description': 'Check out my unboxing of the latest iPhone 15 Pro Max. Link in bio to buy now! Use code SAVE10 for discount.',
                'comments': [
                    'Where did you buy this?',
                    'Link in bio for purchase!',
                    'Amazing product review!',
                    'Love your content'
                ],
                'duration': 45  # seconds
            },
            {
                'video_id': 'demo2',
                'title': 'My morning routine #shorts',
                'description': 'Just showing my daily morning routine. Nothing special here.',
                'comments': [
                    'Great routine!',
                    'Thanks for sharing',
                    'Very inspiring'
                ],
                'duration': 30
            },
            {
                'video_id': 'demo3',
                'title': 'Nike Air Jordan 1 Review #shorts',
                'description': 'Reviewing the classic Nike Air Jordan 1. Get yours at nike.com for $120!',
                'comments': [
                    'These shoes are fire!',
                    'Where to buy?',
                    'Check amazon.com for deals'
                ],
                'duration': 55
            },
            {
                'video_id': 'demo4',
                'title': 'Beautiful sunset timelapse #shorts',
                'description': 'Captured this amazing sunset from my balcony. Nature is beautiful!',
                'comments': [
                    'Gorgeous!',
                    'Amazing colors',
                    'Love nature videos'
                ],
                'duration': 40
            },
            {
                'video_id': 'demo5',
                'title': 'Gaming setup reveal! RTX 4090 #shorts',
                'description': 'Finally got my hands on the RTX 4090! Building the ultimate gaming PC. Parts list in description.',
                'comments': [
                    'Sick setup!',
                    'How much did this cost?',
                    'Link to buy the GPU?',
                    'Check bestbuy.com for stock'
                ],
                'duration': 50
            }
        ]
    
    def demo_filter_process(self):
        """
        Demonstrates the filtering process using demo data
        """
        print("=== YouTube Shorts Product Filter Demo ===\n")
        
        demo_videos = self.get_demo_video_data()
        filtered_shorts = []
        
        for video in demo_videos:
            # Combine all text
            all_text = f"{video['title']} {video['description']} {' '.join(video['comments'])}"
            
            # Apply regex detection
            regex_products = self.detect_products_regex(all_text)
            
            # Apply ML detection
            ml_detection = self.detect_products_ml(all_text)
            
            print(f"Video: {video['title']}")
            print(f"Duration: {video['duration']} seconds (Short: {'Yes' if video['duration'] <= 60 else 'No'})")
            print(f"Regex patterns found: {regex_products}")
            print(f"ML detection: {'Product detected' if ml_detection else 'No product detected'}")
            
            # If either method detects products, include the video
            if regex_products or ml_detection:
                print("✅ FILTERED: Contains product mentions")
                filtered_shorts.append({
                    'video_id': video['video_id'],
                    'title': video['title'],
                    'description': video['description'][:200] + '...' if len(video['description']) > 200 else video['description'],
                    'url': f"https://www.youtube.com/watch?v={video['video_id']}",
                    'regex_patterns': regex_products,
                    'ml_detection': bool(ml_detection)
                })
            else:
                print("❌ NOT FILTERED: No product mentions detected")
            
            print("-" * 80)
        
        print(f"\n=== SUMMARY ===")
        print(f"Total videos processed: {len(demo_videos)}")
        print(f"Shorts with products found: {len(filtered_shorts)}")
        
        # Save results
        with open('demo_filtered_shorts.json', 'w') as f:
            json.dump(filtered_shorts, f, indent=2)
        
        print(f"Results saved to 'demo_filtered_shorts.json'")
        
        return filtered_shorts

def main():
    # Create demo filter instance
    demo_filter = DemoYouTubeShortsFilter()
    
    # Run the demo
    results = demo_filter.demo_filter_process()
    
    print(f"\n=== FILTERED RESULTS ===")
    for i, short in enumerate(results, 1):
        print(f"\n{i}. {short['title']}")
        print(f"   URL: {short['url']}")
        print(f"   Regex patterns: {short['regex_patterns']}")
        print(f"   ML detection: {short['ml_detection']}")

if __name__ == "__main__":
    main()

