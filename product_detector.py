import re

def detect_products_regex(text):
    """
    Detects product mentions in text using regular expressions.
    """
    # Basic regex patterns for common product-related keywords and brands
    patterns = [
        r'iphone\s*\d+',  # iPhone followed by numbers (e.g., iPhone 15)
        r'galaxy\s*s\d+',  # Galaxy S followed by numbers (e.g., Galaxy S24)
        r'nike',  # Nike brand
        r'adidas',  # Adidas brand
        r'buy\s+now',
        r'shop\s+here',
        r'link\s+in\s+bio',
        r'product\s+review',
        r'unboxing',
        r'amazon\.com',
        r'etsy\.com',
        r'shopify\.com',
        r'walmart\.com',
        r'target\.com',
        r'bestbuy\.com',
        r'(\$|€|£)\d+(\.\d{2})?', # Currency followed by numbers
        r'discount',
        r'sale'
    ]

    found_products = []
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            found_products.append(pattern)
    return found_products

if __name__ == '__main__':
    test_texts = [
        "Check out my new iPhone 15 Pro Max! Link in bio to buy now.",
        "Unboxing the latest Samsung Galaxy S24 Ultra. So cool!",
        "My favorite Nike shoes for running. Get them on sale at adidas.com.",
        "Just a regular video, nothing to see here.",
        "This product review will show you everything you need to know.",
        "Found a great deal on amazon.com for only $25.99!"
    ]

    for text in test_texts:
        products = detect_products_regex(text)
        if products:
            print(f"Text: '{text}'\n  Detected product patterns: {', '.join(products)}\n")
        else:
            print(f"Text: '{text}'\n  No product patterns detected.\n")


