#!/usr/bin/env python3
"""
Tutorial: Building the Streamlit Web Interface
Learn how to create a user-friendly web app
"""

import streamlit as st
import json
from tutorial_simple_version import SimpleProductFilter

# Step 1: Page Configuration
st.set_page_config(
    page_title="YouTube Shorts Product Filter Tutorial",
    page_icon="🎯",
    layout="wide"
)

# Step 2: Custom CSS Styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.result-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}

.detection-badge {
    background: #28a745;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Step 3: Main Header
st.markdown("""
<div class="main-header">
    <h1>🎯 YouTube Shorts Product Filter</h1>
    <p>Find product mentions in YouTube Shorts using AI</p>
</div>
""", unsafe_allow_html=True)

# Step 4: Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "YouTube API Key",
        type="password",
        help="Get your API key from Google Cloud Console"
    )
    
    # Search parameters
    search_query = st.text_input(
        "Search Query",
        value="tech review",
        help="What to search for on YouTube"
    )
    
    max_results = st.slider(
        "Max Results",
        min_value=5,
        max_value=50,
        value=10,
        help="Number of videos to analyze"
    )
    
    # Run button
    run_search = st.button("🚀 Start Search", type="primary")

# Step 5: Main Content Area
if not api_key:
    st.warning("⚠️ Please enter your YouTube API key in the sidebar to get started.")
    st.info("""
    **How to get a YouTube API key:**
    1. Go to [Google Cloud Console](https://console.cloud.google.com/)
    2. Create a new project or select existing one
    3. Enable YouTube Data API v3
    4. Create credentials (API Key)
    5. Copy the key and paste it in the sidebar
    """)
else:
    if run_search:
        if not search_query:
            st.error("Please enter a search query!")
        else:
            # Step 6: Run the Analysis
            with st.spinner(f"🔍 Searching for '{search_query}'..."):
                try:
                    # Initialize the filter
                    filter_system = SimpleProductFilter(api_key)
                    
                    # Run the search
                    results = filter_system.filter_product_shorts(search_query)
                    
                    # Step 7: Display Results
                    if results:
                        st.success(f"✅ Found {len(results)} product-related Shorts!")
                        
                        # Results summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Shorts Found", len(results))
                        with col2:
                            total_patterns = sum(len(r['detected_patterns']) for r in results)
                            st.metric("Total Patterns", total_patterns)
                        with col3:
                            avg_patterns = total_patterns / len(results) if results else 0
                            st.metric("Avg Patterns/Short", f"{avg_patterns:.1f}")
                        
                        st.markdown("---")
                        
                        # Display each result
                        for i, short in enumerate(results, 1):
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>#{i} {short['title']}</h4>
                                <p><strong>URL:</strong> <a href="{short['url']}" target="_blank">{short['url']}</a></p>
                                <p><strong>Detected Patterns:</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show detected patterns as badges
                            pattern_cols = st.columns(len(short['detected_patterns']))
                            for j, pattern in enumerate(short['detected_patterns']):
                                with pattern_cols[j]:
                                    st.markdown(f"""
                                    <span class="detection-badge">{pattern}</span>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown("---")
                        
                        # Step 8: Download Results
                        st.subheader("📥 Download Results")
                        
                        # Convert to JSON
                        json_data = json.dumps(results, indent=2)
                        
                        st.download_button(
                            label="💾 Download as JSON",
                            data=json_data,
                            file_name=f"youtube_shorts_{search_query.replace(' ', '_')}.json",
                            mime="application/json"
                        )
                        
                        # Show raw JSON (collapsible)
                        with st.expander("🔍 View Raw JSON Data"):
                            st.json(results)
                    
                    else:
                        st.warning("😔 No product-related Shorts found. Try a different search query!")
                        
                except Exception as e:
                    st.error(f"❌ Error occurred: {str(e)}")
                    st.info("Make sure your API key is valid and has YouTube Data API v3 enabled.")

# Step 9: Footer Information
st.markdown("---")
st.markdown("""
### 🛠️ How This Works

This app combines two detection methods:

1. **Regex Patterns**: Fast detection of explicit product mentions
   - Brand names (Nike, Apple, etc.)
   - Shopping keywords (buy now, shop here)
   - E-commerce URLs (amazon.com, etc.)
   - Price patterns ($19.99, etc.)

2. **Machine Learning**: Smart contextual detection
   - TF-IDF text vectorization
   - Logistic regression classifier
   - Trained on product/non-product examples

**Built with:** Python, Streamlit, YouTube Data API v3, scikit-learn
""")

# Step 10: Debug Information (for development)
if st.checkbox("🐛 Show Debug Info"):
    st.subheader("Debug Information")
    st.write("**Session State:**")
    st.write(st.session_state)
    
    if 'results' in locals():
        st.write("**Last Results:**")
        st.write(f"Number of results: {len(results) if 'results' in locals() else 0}")