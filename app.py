import json
import streamlit as st
from youtube_shorts_filter import YouTubeShortsProductFilter, API_KEY

# ------------------- Page setup -------------------
st.set_page_config(page_title="Finder AI – YouTube Shorts Product Filter", layout="wide")

# ------------------- Mobile-Responsive Header -------------------
st.markdown("""
<style>
/* Mobile-first responsive design */
.footer-sticky {
    position: relative;
    padding: 8px 0;
    background: #0f172a;
    margin-bottom: 10px;
}
.footer-sticky .footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 16px;
    color: #94a3b8;
    flex-wrap: wrap;
    gap: 8px;
}
.footer-sticky a {
    color: #38bdf8;
    font-weight: bold;
    text-decoration: none;
}

/* Mobile styles */
@media (max-width: 768px) {
    .footer-sticky {
        padding: 6px 0;
    }
    .footer-sticky .footer {
        font-size: 14px;
        flex-direction: column;
        text-align: center;
        gap: 4px;
    }
    .footer-sticky .footer > div {
        width: 100%;
    }
}

/* Ensure main content is visible on mobile */
.main .block-container {
    padding-top: 1rem;
    max-width: 100%;
}

/* Mobile card adjustments */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Stack columns on mobile */
    .row-widget.stColumns {
        flex-direction: column !important;
    }
    
    .row-widget.stColumns > div {
        width: 100% !important;
        margin-bottom: 10px;
    }
    
    /* Adjust video cards for mobile */
    .video-card {
        margin-bottom: 20px !important;
    }
}

/* Mobile sidebar handling - hide completely on mobile */
@media (max-width: 768px) {
    /* Hide sidebar completely on mobile */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Ensure main content takes full width */
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Mobile-friendly button styling */
    .stButton > button {
        width: 100%;
        margin-top: 10px;
    }
    
    /* Stack columns on mobile */
    .row-widget.stColumns {
        flex-direction: column !important;
    }
    
    .row-widget.stColumns > div {
        width: 100% !important;
        margin-bottom: 15px;
    }
}

/* Improve mobile text input */
@media (max-width: 640px) {
    .stTextArea textarea {
        min-height: 100px !important;
        font-size: 16px !important; /* Prevent zoom on iOS */
    }
    
    .stSlider {
        margin-bottom: 20px;
    }
    
    /* Make buttons more touch-friendly */
    .stButton > button {
        min-height: 44px;
        font-size: 16px;
    }
}
</style>
<div class="footer-sticky">
    <div class="footer">
        <div>🤖 Developed with ❤️ by <b>Rajat Srivastav</b></div>
        <div>💻 <a href="https://github.com/raj577" target="_blank">GitHub</a></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------- Main Content Area -------------------
st.title("🎯 YouTube Shorts Product Filter")
st.markdown("Find product mentions in YouTube Shorts using AI-powered detection")

# ------------------- Mobile-First Layout -------------------
# Create two columns: settings and info (on desktop), stack on mobile
col1, col2 = st.columns([2, 1])

with col1:
    st.header("⚙️ Search Settings")
    query_input = st.text_area("Enter keywords (comma-separated):",  value="amazon finds, must have products, kitchen hacks, "
          "viral amazon products, best amazon products, useful home items"
    )
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        max_results = st.slider("🔍 Max results per query", min_value=5, max_value=50, value=5)
    with col_b:
        st.write("")  # Spacer
        run_btn = st.button("🚀 Run Finder AI", type="primary")

with col2:
    with st.expander("🧠 How Finder AI Works", expanded=False):
        st.markdown("""
**Finder AI** uses a hybrid pipeline:

- ✅ **TF‑IDF Vectorizer**: Converts text into feature vectors  
- 🧠 **ML Classifier**: Predicts if a Short is product-related  
- 🔍 **Regex Detection**: Matches product keywords and patterns  
- 🤖 Evaluates title, description, and comments
- 📈 Ideal for product scouting, marketing, eCommerce leads

Built with love for data-driven YouTube discovery 🚀
""")

# ------------------- Optional Sidebar for Desktop -------------------
with st.sidebar:
    st.markdown("### 💡 Tips")
    st.markdown("""
- Use specific product categories for better results
- Try brand names like "apple", "samsung", "nike"
- Include shopping terms like "review", "unboxing"
- Separate keywords with commas
""")
    
    st.markdown("### 📊 About")
    st.markdown("""
This tool combines regex patterns with machine learning to identify product mentions in YouTube Shorts.
""")
    
    st.markdown("---")
    st.markdown("🤖 **Developed by Rajat Srivastav**")
    st.markdown("[GitHub](https://github.com/raj577)")

# ------------------- Run Filter -------------------
if run_btn:
    queries = [q.strip() for q in query_input.split(",") if q.strip()]
    if not queries:
        st.error("Please enter at least one keyword.")
        st.stop()

    st.info("🔎 Scanning YouTube… please wait.")
    finder = YouTubeShortsProductFilter(API_KEY)
    results = finder.run_filter(queries, max_results=max_results)

    st.success(f"✅ Found {len(results)} relevant Shorts.")

    # ------------------ Display cards in responsive grid ------------------
    for i in range(0, len(results), 3):
        cols = st.columns([1, 1, 1])  # Equal width columns
        for col_idx, short in enumerate(results[i:i+3]):
            video_id = short["video_id"]
            title = short["title"]
            url = short["url"]
            desc = short["description"]
            thumb = f"https://img.youtube.com/vi/{video_id}/0.jpg"

            # Detection badges
            detected_by = []
            if short["regex_patterns"]:
                detected_by.append("Regex")
            if str(short["ml_detection"]).lower() == "true":
                detected_by.append("ML")

            badge_html = ""
            if "Regex" in detected_by:
                badge_html += """<span style="background:#fde68a;color:#92400e;font-size:12px;
                                  padding:4px 8px;border-radius:5px;margin-right:5px;
                                  font-weight:600;display:inline-block;">
                                  🔤 Regex</span>"""
            if "ML" in detected_by:
                badge_html += """<span style="background:#6ee7b7;color:#065f46;font-size:12px;
                                  padding:4px 8px;border-radius:5px;font-weight:600;
                                  display:inline-block;">
                                  🤖 ML Model</span>"""

            with cols[col_idx]:
                st.markdown(f"""
                    <div class="video-card" style="
                        background-color:#1e293b;
                        padding:10px;
                        border-radius:10px;
                        margin-bottom:15px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <a href="{url}" target="_blank">
                            <img src="{thumb}" width="100%" style="border-radius:8px; display:block;">
                        </a>
                        <a href="{url}" target="_blank" style="text-decoration:none;">
                            <h5 style="margin:8px 0 4px 0; font-size:16px; color:#f1f5f9; line-height:1.3;">{title}</h5>
                        </a>
                        <div style="margin:6px 0;">{badge_html}</div>
                        <p style="margin:6px 0 0 0; font-size:13px; color:#cbd5e1; line-height:1.4;">
                            {desc}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

    # ------------------ Download JSON ------------------
    st.download_button("💾 Download JSON", data=json.dumps(results, indent=2), file_name="filtered_shorts.json")
