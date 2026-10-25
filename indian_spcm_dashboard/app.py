import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="SPCM | Indian Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Demo data (clearly marked) ----------------
# These values are illustrative UI data, NOT live quotes or model predictions.
stocks = pd.DataFrame([
    {"Symbol": "RELIANCE", "Company": "Reliance Industries", "Price": 1202.00, "Change": -0.76, "Sentiment": "Neutral", "Score": 0.08},
    {"Symbol": "TCS", "Company": "Tata Consultancy Services", "Price": 2460.90, "Change": -0.17, "Sentiment": "Positive", "Score": 0.31},
    {"Symbol": "INFY", "Company": "Infosys", "Price": 1130.70, "Change": -2.63, "Sentiment": "Negative", "Score": -0.42},
    {"Symbol": "HDFCBANK", "Company": "HDFC Bank", "Price": 1684.25, "Change": 0.54, "Sentiment": "Positive", "Score": 0.22},
    {"Symbol": "ICICIBANK", "Company": "ICICI Bank", "Price": 1298.60, "Change": 0.83, "Sentiment": "Positive", "Score": 0.37},
])
demo_news = [
    ("RELIANCE", "Illustrative headline: Energy and retail business outlook", "Neutral", "Demo item — replace with Phase 3 feed"),
    ("TCS", "Illustrative headline: IT services demand and quarterly outlook", "Positive", "Demo item — replace with Phase 3 feed"),
    ("INFY", "Illustrative headline: Technology sector market commentary", "Negative", "Demo item — replace with Phase 3 feed"),
]

# ---------------- Styling ----------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background: #0b1020; color: #e8edf8; }
  [data-testid="stSidebar"] { background: #10172a; border-right: 1px solid #26314a; }
  .hero { padding: 22px 24px; border: 1px solid #263653; border-radius: 18px;
          background: linear-gradient(120deg, #172640 0%, #11192d 55%, #18233b 100%); margin-bottom: 18px; }
  .eyebrow { color: #78d8c3; font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
  .hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 32px; margin: 6px 0 4px; color: #f4f7ff; }
  .hero p { color: #aab7d0; margin: 0; font-size: 14px; }
  .metric-card { background: #121b30; border: 1px solid #263653; border-radius: 14px; padding: 16px 17px; min-height: 112px; }
  .metric-label { color: #93a2bd; font-size: 12px; margin-bottom: 8px; }
  .metric-value { color: #f4f7ff; font-family: 'Space Grotesk', sans-serif; font-size: 25px; font-weight: 700; }
  .metric-sub { color: #93a2bd; font-size: 11px; margin-top: 6px; }
  .section-title { font-family: 'Space Grotesk', sans-serif; color: #f4f7ff; font-size: 19px; font-weight: 600; margin: 12px 0 4px; }
  .muted { color: #93a2bd; font-size: 12px; }
  .pill { display:inline-block; border-radius:999px; padding:4px 9px; font-size:10px; font-weight:700; letter-spacing:.4px; }
  .pill-demo { background:#49391c; color:#ffd782; border:1px solid #79602b; }
  div[data-testid="stDataFrame"] { border: 1px solid #263653; border-radius: 12px; overflow: hidden; }
  .footer { color:#71819f; font-size:11px; text-align:center; padding:18px 0 4px; }
  .stTabs [data-baseweb="tab-list"] { gap: 12px; }
  .stTabs [data-baseweb="tab"] { color:#aab7d0; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📈 SPCM")
    st.caption("Indian Stock Intelligence")
    st.markdown("---")
    page = st.radio("NAVIGATION", ["Overview", "Stock Explorer", "News & Sentiment", "Methodology"], label_visibility="visible")
    st.markdown("---")
    st.markdown("**Data status**")
    st.warning("DEMO MODE")
    st.caption("Illustrative values only. Real market/news/model adapters are planned for backend integration.")
    st.markdown("---")
    st.caption("SPCM · Stock Price + Sentiment")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Stock Price & Sentiment · India</div>
  <h1>Market Intelligence Dashboard</h1>
  <p>Multimodal stock movement research combining market features, financial-news sentiment, and machine-learning inference.</p>
</div>
""", unsafe_allow_html=True)

now = datetime.now().strftime("%d %b %Y · %I:%M %p")
st.markdown(f'<span class="pill pill-demo">● DEMO MODE — NOT LIVE PREDICTIONS</span> &nbsp; <span class="muted">Dashboard session: {now}</span>', unsafe_allow_html=True)
st.write("")

if page == "Overview":
    st.markdown('<div class="section-title">Market snapshot</div><div class="muted">Illustrative sample values based on the project demo layout; not current quotes.</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    metrics = [
        ("WATCHLIST", "05 stocks", "NSE / BSE universe"),
        ("MARKET FEED", "Not connected", "Phase 1 adapter ready in project"),
        ("SENTIMENT", "Demo values", "Phase 3 integration pending"),
        ("MODEL STATUS", "Not connected", "Frozen artifact required"),
    ]
    for col, (label, value, sub) in zip(cols, metrics):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:21px">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.55, 1])
    with left:
        st.markdown('<div class="section-title">Watchlist performance</div>', unsafe_allow_html=True)
        selected = st.selectbox("Select stock for illustrative chart", stocks["Symbol"].tolist(), index=0)
        row = stocks[stocks["Symbol"] == selected].iloc[0]
        rng = np.random.default_rng(sum(map(ord, selected)))
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=30)
        returns = rng.normal(0.001, 0.012, len(dates))
        prices = row["Price"] * np.cumprod(1 + returns)
        prices[-1] = row["Price"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines", name=selected,
                                 line=dict(color="#69d7c3", width=2.5),
                                 fill="tozeroy", fillcolor="rgba(105,215,195,0.08)"))
        fig.update_layout(height=310, margin=dict(l=8,r=8,t=12,b=8), paper_bgcolor="#121b30",
                          plot_bgcolor="#121b30", font=dict(color="#aab7d0"),
                          xaxis=dict(showgrid=False, zeroline=False),
                          yaxis=dict(showgrid=True, gridcolor="#263653", zeroline=False, title="Illustrative price (₹)"),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Illustrative chart generated for the interface. Replace with fetched OHLCV history during integration.")
    with right:
        st.markdown('<div class="section-title">Model output preview</div>', unsafe_allow_html=True)
        st.info("Real model inference is not connected yet.")
        st.markdown("**Target workflow**")
        st.markdown("1. Validate latest OHLCV data\n2. Generate training-compatible lag features\n3. Align point-in-time news sentiment\n4. Apply frozen preprocessing and model\n5. Rank eligible stocks")
        st.markdown("---")
        st.markdown("**Prediction fields planned**")
        st.markdown("- Predicted class / direction\n- Model probability\n- Observation timestamp\n- Feed and model status")
        st.caption("No illustrative probability is presented as a genuine model output.")

    st.markdown('<div class="section-title">Watchlist</div>', unsafe_allow_html=True)
    display = stocks[["Symbol", "Company", "Price", "Change", "Sentiment"]].copy()
    display["Price"] = display["Price"].map(lambda x: f"₹{x:,.2f}")
    display["Change"] = display["Change"].map(lambda x: f"{x:+.2f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)

elif page == "Stock Explorer":
    st.markdown('<div class="section-title">Stock explorer</div>', unsafe_allow_html=True)
    symbol = st.selectbox("Choose a company", stocks["Symbol"].tolist())
    r = stocks[stocks["Symbol"] == symbol].iloc[0]
    a,b,c = st.columns(3)
    a.metric("Illustrative price", f"₹{r.Price:,.2f}", f"{r.Change:+.2f}%")
    b.metric("Demo sentiment", r.Sentiment)
    c.metric("Sentiment score", f"{r.Score:+.2f}")
    st.warning("These are demonstration values, not verified live prices or predictions.")
    st.markdown("**Integration checklist**")
    st.checkbox("Fetch validated OHLCV bars", value=False, disabled=True)
    st.checkbox("Build exact training-compatible feature schema", value=False, disabled=True)
    st.checkbox("Load frozen model + fitted preprocessing", value=False, disabled=True)
    st.checkbox("Generate real prediction and timestamp", value=False, disabled=True)

elif page == "News & Sentiment":
    st.markdown('<div class="section-title">News & sentiment monitor</div>', unsafe_allow_html=True)
    st.caption("Illustrative placeholders only. No real articles are fetched in demo mode.")
    for symbol, headline, sentiment, status in demo_news:
        with st.container(border=True):
            c1, c2 = st.columns([4,1])
            c1.markdown(f"**{symbol}** · {headline}")
            c1.caption(status)
            c2.markdown(f"**{sentiment}**")
    st.info("Phase 3 will replace these placeholders with source headlines, publication timestamps, deduplication, sentiment scores, and provider status.")

elif page == "Methodology":
    st.markdown('<div class="section-title">Project methodology</div>', unsafe_allow_html=True)
    st.markdown("""
    **Objective:** classify stock movement by combining price-derived signals and financial-news sentiment.

    **Current project milestones**
    - **Phase 1:** Indian market data fetching and validation.
    - **Phase 2:** inference-time price feature generation, lag features, and schema checks.
    - **Phase 3:** news ingestion and point-in-time sentiment integration — in progress.
    - **Next:** frozen-model inference, ranked outputs, and backend-connected dashboard.

    **Integrity safeguards**
    - Historical/delayed data must not be described as real-time.
    - Demo data must not be interpreted as live market data.
    - Predictions require compatible trained artifacts and fitted preprocessing.
    - News must be aligned by publication/availability time to avoid future-information leakage.
    """)
    st.markdown("**Planned integration contract**")
    st.code("""market_data(symbols) -> prices, timestamp, feed_status
news_sentiment(symbols) -> articles, scores, publication_times, source_status
predict(features, frozen_artifacts) -> class, probability, model_version
dashboard -> displays outputs + provenance; never fabricates missing values""", language="python")

st.markdown('<div class="footer">SPCM · Indian Stock Movement Prediction · Demo UI · Illustrative data clearly labelled</div>', unsafe_allow_html=True)
