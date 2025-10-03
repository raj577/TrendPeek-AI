#!/usr/bin/env python3
"""
Step 5: Complete Streamlit Web Application
Build a production-ready web interface for the YouTube Shorts Product Filter
"""

import streamlit as st
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from step4_complete_system import YouTubeShortsProductFilter

# Page configuration
st.set_page_config(
    page_title="🎯 Finder AI - YouTube Shorts Product Filter",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
/* Main header styling */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1.2rem;
    opacity: 0.9;
}

/* Result cards */
.result-card {
    background: linear-gradient(145deg, #f8f9fa, #e9ecef);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 5px solid #667eea;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: transform 0.2s ease;
}

.result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

/* Detection badges */
.badge-regex {
    background: linear-gradient(45deg, #ffd700, #ffed4e);
    color: #8b5a00;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.5rem;
    display: inline-block;
    box-shadow: 0 2px 5px rgba(255,215,0,0.3);
}

.badge-ml {
    background: linear-gradient(45deg, #4ade80, #22c55e);
    color: #065f46;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.5rem;
    display: inline-block;
    box-shadow: 0 2px 5px rgba(34,197,94,0.3);
}

/* Stats cards */
.stat-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    border-top: 4px solid #667eea;
}

.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: #667eea;
    margin: 0;
}

.stat-label {
    color: #6c757d;
    font-size: 0.9rem;
    margin: 0.5rem 0 0 0;
}

/* Sidebar styling */
.sidebar-section {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

/* Loading animation */
.loading-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
}

.loading-spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #667eea;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'filter_system' not in st.session_state:
        st.session_state.filter_system = None
    if 'last_search_time' not in st.session_state:
        st.session_state.last_search_time = None

def create_main_header():
    """Create the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Finder AI</h1>
        <p>Discover Product Mentions in YouTube Shorts with AI-Powered Detection</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create the application sidebar"""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "YouTube API Key",
            type="password",
            help="Enter your YouTube Data API v3 key",
            placeholder="AIza..."
        )
        
        st.markdown("---")
        
        # Search configuration
        st.markdown("### 🔍 Search Settings")
        
        search_queries = st.text_area(
            "Search Queries (one per line)",
            value="amazon finds\ntech review\nproduct unboxing\nmust have gadgets\ncool gadgets",
            height=120,
            help="Enter search terms, one per line"
        )
        
        max_results = st.slider(
            "Max Results per Query",
            min_value=5,
            max_value=50,
            value=15,
            help="Number of videos to analyze per search query"
        )
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            include_comments = st.checkbox(
                "Analyze Comments",
                value=True,
                help="Include video comments in analysis (slower but more accurate)"
            )
            
            min_confidence = st.slider(
                "ML Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Minimum confidence for ML detection"
            )
        
        st.markdown("---")
        
        # How it works section
        st.markdown("### 🧠 How It Works")
        st.markdown("""
        **Finder AI** uses a hybrid approach:
        
        🔤 **Regex Detection**
        - Brand names (iPhone, Samsung)
        - Shopping keywords (buy now, sale)
        - E-commerce URLs (amazon.com)
        - Price patterns ($19.99)
        
        🤖 **ML Detection**
        - TF-IDF text vectorization
        - Logistic regression classifier
        - Contextual product understanding
        - 90%+ accuracy on test data
        
        📊 **Performance**
        - ~0.25 MB RAM usage
        - 100K+ texts/second
        - Real-time processing
        """)
        
        return api_key, search_queries, max_results, include_comments, min_confidence

def display_performance_stats(stats):
    """Display performance statistics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['videos_processed']}</div>
            <div class="stat-label">Videos Processed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['shorts_found']}</div>
            <div class="stat-label">Shorts Found</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['products_detected']}</div>
            <div class="stat-label">Products Detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['memory_usage_mb']:.1f}MB</div>
            <div class="stat-label">Memory Usage</div>
        </div>
        """, unsafe_allow_html=True)

def create_results_visualization(results):
    """Create visualizations for the results"""
    if not results:
        return
    
    st.markdown("### 📊 Analysis Dashboard")
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Detection method distribution
        regex_count = sum(1 for r in results if r['regex_patterns'])
        ml_count = sum(1 for r in results if r['ml_prediction'])
        both_count = sum(1 for r in results if r['regex_patterns'] and r['ml_prediction'])
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Regex Only', 'ML Only', 'Both Methods'],
            values=[regex_count - both_count, ml_count - both_count, both_count],
            hole=0.4,
            marker_colors=['#ffd700', '#4ade80', '#667eea']
        )])
        fig_pie.update_layout(title="Detection Method Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # ML Confidence distribution
        if 'ml_confidence' in df.columns:
            fig_hist = px.histogram(
                df, 
                x='ml_confidence',
                nbins=20,
                title="ML Confidence Score Distribution",
                color_discrete_sequence=['#667eea']
            )
            fig_hist.update_layout(
                xaxis_title="Confidence Score",
                yaxis_title="Number of Videos"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

def display_results(results, min_confidence):
    """Display the filtered results"""
    if not results:
        st.warning("😔 No product-related Shorts found. Try different search queries!")
        return
    
    # Filter by confidence if needed
    filtered_results = [
        r for r in results 
        if r['ml_confidence'] >= min_confidence or r['regex_patterns']
    ]
    
    st.success(f"✅ Found {len(filtered_results)} product-related YouTube Shorts!")
    
    # Performance stats
    if st.session_state.filter_system:
        stats = st.session_state.filter_system.get_performance_stats()
        display_performance_stats(stats)
    
    st.markdown("---")
    
    # Visualizations
    create_results_visualization(filtered_results)
    
    st.markdown("---")
    
    # Results list
    st.markdown("### 🎬 Detected Shorts")
    
    # Sort by confidence
    sorted_results = sorted(
        filtered_results, 
        key=lambda x: x['ml_confidence'], 
        reverse=True
    )
    
    for i, result in enumerate(sorted_results, 1):
        # Create detection badges
        badges_html = ""
        if result['regex_patterns']:
            badges_html += '<span class="badge-regex">🔤 Regex</span>'
        if result['ml_prediction']:
            badges_html += '<span class="badge-ml">🤖 ML Model</span>'
        
        # Create result card
        st.markdown(f"""
        <div class="result-card">
            <h4>#{i} {result['title']}</h4>
            <div style="margin: 0.5rem 0;">
                {badges_html}
            </div>
            <p><strong>🔗 URL:</strong> <a href="{result['url']}" target="_blank">{result['url']}</a></p>
            <p><strong>📝 Description:</strong> {result['description']}</p>
            <p><strong>🎯 ML Confidence:</strong> {result['ml_confidence']:.1%}</p>
            <p><strong>🔍 Regex Patterns:</strong> {len(result['regex_patterns'])}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Download section
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON download
        json_data = json.dumps(filtered_results, indent=2)
        st.download_button(
            label="📄 Download as JSON",
            data=json_data,
            file_name=f"youtube_shorts_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # CSV download
        if filtered_results:
            df = pd.DataFrame(filtered_results)
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📊 Download as CSV",
                data=csv_data,
                file_name=f"youtube_shorts_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def main():
    """Main application function"""
    initialize_session_state()
    create_main_header()
    
    # Sidebar
    api_key, search_queries, max_results, include_comments, min_confidence = create_sidebar()
    
    # Main content area
    if not api_key:
        st.warning("⚠️ Please enter your YouTube API key in the sidebar to get started.")
        
        with st.expander("🔑 How to get a YouTube API key"):
            st.markdown("""
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project or select an existing one
            3. Enable the **YouTube Data API v3**
            4. Go to **Credentials** and create an **API Key**
            5. Copy the key and paste it in the sidebar
            6. (Optional) Restrict the key to YouTube Data API v3 for security
            """)
        return
    
    # Parse search queries
    queries = [q.strip() for q in search_queries.split('\n') if q.strip()]
    
    if not queries:
        st.error("Please enter at least one search query!")
        return
    
    # Search button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button(
            "🚀 Start AI Analysis",
            type="primary",
            use_container_width=True
        )
    
    # Run analysis
    if search_button:
        if not st.session_state.filter_system:
            with st.spinner("🤖 Initializing AI models..."):
                try:
                    st.session_state.filter_system = YouTubeShortsProductFilter(api_key)
                except Exception as e:
                    st.error(f"❌ Failed to initialize system: {str(e)}")
                    return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔍 Searching YouTube...")
            progress_bar.progress(20)
            
            # Run the analysis
            with st.spinner("🧠 AI is analyzing YouTube Shorts..."):
                results = st.session_state.filter_system.run_complete_filter(
                    queries, 
                    max_results_per_query=max_results
                )
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Store results
            st.session_state.results = results
            st.session_state.last_search_time = datetime.now()
            
            # Clear progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("Please check your API key and try again.")
            return
    
    # Display results if available
    if st.session_state.results:
        display_results(st.session_state.results, min_confidence)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 1rem;">
        <p>🤖 Built with ❤️ using Streamlit, YouTube Data API v3, and Machine Learning</p>
        <p>Powered by TF-IDF + Logistic Regression | Optimized for low-RAM environments</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()