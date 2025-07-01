import json
import streamlit as st
from youtube_shorts_filter import YouTubeShortsProductFilter, API_KEY

# ------------------- Page setup -------------------
st.set_page_config(page_title="Finder AI – YouTube Shorts Product Filter", layout="wide")

# ------------------- Sticky Header -------------------
st.markdown("""
<style>
.footer-sticky {
    position: sticky;
    top: 0;
    z-index: 999;
    padding: 10px 0 5px 0;
    background: #0f172a;
}
.footer-sticky .footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 20px;
    color: #94a3b8;
}
.footer-sticky a {
    color: #38bdf8;
    font-weight: bold;
    text-decoration: none;
}
</style>
<div class="footer-sticky">
    <div class="footer">
        <div>🤖 Developed with ❤️ by <b>Rajat Srivastav</b></div>
        <div>💻 <a href="https://github.com/raj577" target="_blank">GitHub</a></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------- Sidebar: AI Description -------------------
with st.sidebar:
    st.header("🧠 How Finder AI Works")
    st.markdown("""
**Finder AI** is powered by a hybrid pipeline of:

- ✅ **TF‑IDF Vectorizer**: Converts text into feature vectors  
- 🧠 **ML Classifier**: Predicts if a Short is product-related  
- 🔍 **Regex Detection**: Matches product keywords and patterns  
- 🤖 Evaluates title, description, and comments
- 📈 Ideal for product scouting, marketing, eCommerce leads

Built with love for data-driven YouTube discovery 🚀
""")

    # ------------------- Search Form -------------------
    st.header("⚙️ Search Settings")
    query_input = st.text_area("Enter keywords (comma-separated):",  value="amazon finds, must have products, kitchen hacks, cool gadgets, "
          "viral amazon products, best amazon products, useful home items, "
          # "amazon must haves, tech gadgets amazon, tiktok amazon products, "
          # "cool stuff on amazon, unique amazon finds, amazon gift ideas, "
          # "budget gadgets, amazon tools, affordable tech amazon, "
          # "amazon electronics under 1000, amazon beauty products, "
          # "home organization amazon, smart home gadgets, cheap amazon finds, "
          # "mini gadgets amazon, best kitchen gadgets"
    )
    max_results = st.slider("🔍 Max results per query", min_value=5, max_value=50, value=5)
    run_btn = st.button("🚀 Run Finder AI")

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

    # ------------------ Display cards in rows of 3 ------------------
    for i in range(0, len(results), 3):
        cols = st.columns(3)
        for col, short in zip(cols, results[i:i+3]):
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

            with col:
                st.markdown(f"""
                    <div style="background-color:#1e293b;padding:10px;border-radius:10px;">
                        <a href="{url}" target="_blank">
                            <img src="{thumb}" width="100%" style="border-radius:8px;">
                        </a>
                        <a href="{url}" target="_blank" style="text-decoration:none;">
                            <h5 style="margin:4px 0; font-size:16px; color:#f1f5f9;">{title}</h5>
                        </a>
                        <div style="margin:6px 0;">{badge_html}</div>
                        <p style="margin:6px 0 0 0; font-size:13px; color:#cbd5e1;">
                            {desc}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

    # ------------------ Download JSON ------------------
    st.download_button("💾 Download JSON", data=json.dumps(results, indent=2), file_name="filtered_shorts.json")
