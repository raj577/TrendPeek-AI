#!/usr/bin/env python3
"""
Step 2: Advanced Regex Pattern Detection Engine
Learn how to build sophisticated product detection patterns
"""

import re
from typing import List, Dict, Tuple

class AdvancedRegexDetector:
    def __init__(self):
        """Initialize with comprehensive regex patterns"""
        self.patterns = self._build_pattern_library()
        print(f"✅ Loaded {len(self.patterns)} detection patterns")
    
    def _build_pattern_library(self) -> Dict[str, List[str]]:
        """Build comprehensive pattern library organized by categories"""
        return {
            # Brand and Product Patterns
            "brands": [
                r"iphone\s*\d+",           # iPhone 15, iPhone15
                r"galaxy\s*s\d+",          # Galaxy S24, GalaxyS24
                r"macbook\s*(pro|air)?",   # MacBook, MacBook Pro
                r"airpods?\s*(pro|max)?",  # AirPods, AirPod Pro
                r"nintendo\s*switch",      # Nintendo Switch
                r"playstation\s*\d+",      # PlayStation 5
                r"xbox\s*(series\s*[xs]|one)?", # Xbox Series X
            ],
            
            # Shopping Intent Patterns
            "shopping_intent": [
                r"buy\s+now",
                r"shop\s+here",
                r"get\s+yours?",
                r"order\s+now",
                r"purchase\s+link",
                r"available\s+now",
                r"in\s+stock",
                r"limited\s+time",
                r"while\s+supplies\s+last",
            ],
            
            # Social Commerce Patterns
            "social_commerce": [
                r"link\s+in\s+bio",
                r"swipe\s+up",
                r"check\s+description",
                r"dm\s+me",
                r"comment\s+below",
                r"tag\s+a\s+friend",
            ],
            
            # Content Type Patterns
            "content_types": [
                r"product\s+review",
                r"unboxing",
                r"haul",
                r"try\s+on",
                r"first\s+impressions?",
                r"comparison",
                r"vs\s+",
                r"worth\s+it\??",
            ],
            
            # E-commerce URLs
            "ecommerce_urls": [
                r"amazon\.com",
                r"amzn\.to",
                r"etsy\.com",
                r"shopify\.com",
                r"walmart\.com",
                r"target\.com",
                r"bestbuy\.com",
                r"ebay\.com",
                r"aliexpress\.com",
                r"temu\.com",
            ],
            
            # Price and Deal Patterns
            "pricing": [
                r"(\$|€|£|¥)\s*\d+(?:\.\d{2})?",  # $19.99, €25.50
                r"\d+\s*(?:dollars?|euros?|pounds?)", # 20 dollars
                r"free\s+shipping",
                r"discount",
                r"sale",
                r"deal",
                r"coupon",
                r"promo\s*code",
                r"\d+%\s*off",
                r"save\s+\$?\d+",
            ],
            
            # Affiliate and Sponsored Content
            "affiliate": [
                r"affiliate\s+link",
                r"sponsored",
                r"ad",
                r"partnership",
                r"collab",
                r"gifted",
                r"#ad",
                r"#sponsored",
                r"#affiliate",
            ]
        }
    
    def detect_patterns(self, text: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Detect all patterns in text and return organized results
        Returns: {category: [(pattern, matched_text), ...]}
        """
        results = {}
        text_lower = text.lower()
        
        for category, patterns in self.patterns.items():
            category_matches = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    category_matches.append((pattern, match.group()))
            
            if category_matches:
                results[category] = category_matches
        
        return results
    
    def get_confidence_score(self, detection_results: Dict) -> float:
        """Calculate confidence score based on pattern matches"""
        weights = {
            "brands": 3.0,
            "shopping_intent": 2.5,
            "ecommerce_urls": 2.0,
            "pricing": 2.0,
            "content_types": 1.5,
            "social_commerce": 1.0,
            "affiliate": 1.0,
        }
        
        total_score = 0
        for category, matches in detection_results.items():
            weight = weights.get(category, 1.0)
            total_score += len(matches) * weight
        
        # Normalize to 0-1 scale
        max_possible_score = 20  # Reasonable maximum
        return min(total_score / max_possible_score, 1.0)
    
    def analyze_text(self, text: str) -> Dict:
        """Complete analysis of text with detailed results"""
        detection_results = self.detect_patterns(text)
        confidence = self.get_confidence_score(detection_results)
        
        # Flatten all matches for summary
        all_matches = []
        for category, matches in detection_results.items():
            for pattern, matched_text in matches:
                all_matches.append({
                    "category": category,
                    "pattern": pattern,
                    "matched_text": matched_text
                })
        
        return {
            "has_products": len(all_matches) > 0,
            "confidence_score": confidence,
            "total_matches": len(all_matches),
            "categories_detected": list(detection_results.keys()),
            "detailed_matches": detection_results,
            "summary_matches": all_matches
        }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts efficiently"""
        results = []
        for text in texts:
            results.append(self.analyze_text(text))
        return results

# Testing and demonstration
if __name__ == "__main__":
    detector = AdvancedRegexDetector()
    
    # Test cases with different types of product content
    test_texts = [
        "Check out my new iPhone 15 Pro Max! Link in bio to buy now for $999",
        "Unboxing the latest Samsung Galaxy S24 Ultra. So cool! Get yours on amazon.com",
        "My favorite Nike shoes for running. 30% off sale at nike.com - use code SAVE30",
        "Just a regular video, nothing to see here. Beautiful sunset today.",
        "Product review: AirPods Pro vs Sony WH-1000XM5. Which is worth it?",
        "Amazon haul! Found amazing deals - swipe up for links. #affiliate #ad",
        "DIY home decor on a budget. No products mentioned here.",
        "Sponsored: Try this new skincare routine! DM me for discount code",
    ]
    
    print("🧪 Testing Advanced Regex Detection Engine\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: '{text[:50]}...'")
        
        result = detector.analyze_text(text)
        
        print(f"  Product Detected: {'✅ YES' if result['has_products'] else '❌ NO'}")
        print(f"  Confidence: {result['confidence_score']:.2%}")
        print(f"  Total Matches: {result['total_matches']}")
        
        if result['categories_detected']:
            print(f"  Categories: {', '.join(result['categories_detected'])}")
            
            # Show top matches
            for match in result['summary_matches'][:3]:  # Show first 3
                print(f"    - {match['category']}: '{match['matched_text']}'")
        
        print()
    
    # Performance test
    print("⚡ Performance Test:")
    import time
    
    large_text_batch = test_texts * 1000  # 8,000 texts
    start_time = time.time()
    
    batch_results = detector.batch_analyze(large_text_batch)
    
    end_time = time.time()
    processing_time = end_time - start_time
    texts_per_second = len(large_text_batch) / processing_time
    
    print(f"Processed {len(large_text_batch)} texts in {processing_time:.2f}s")
    print(f"Speed: {texts_per_second:,.0f} texts/second")
    print(f"Product detection rate: {sum(1 for r in batch_results if r['has_products']) / len(batch_results):.1%}")