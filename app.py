import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ---------- Page Config (FIRST COMMAND) ----------
st.set_page_config(page_title="Social Media RAG — Trend Explainer", page_icon="🔥", layout="wide")

st.title("🔥 Social Media RAG — Trend Explainer")
st.caption("Mock ‘live’ data; backend is API-ready.")

# ---------- Session State for Refresh ----------
if "refresh_key" not in st.session_state:
    st.session_state["refresh_key"] = random.randint(0, 999999)

# Sidebar
with st.sidebar:
    st.header("Settings")
    n_topics = st.slider("Number of topics", 8, 15, 10)
    posts_per_topic = st.slider("Posts per topic (range)", 2, 6, 3)
    st.markdown("---")
    if st.button("🔄 Refresh data"):
        st.session_state["refresh_key"] = random.randint(0, 999999)
        st.rerun()

# Randomize each refresh
random.seed(st.session_state["refresh_key"])
np.random.seed(st.session_state["refresh_key"])

# ---------- Topic & copy banks ----------
topic_bank = [
    "#HeatWave", "NPC Meme", "Swift Tour", "AI Chip Shortage", "Transfer Deadline",
    "#ClimateStrike", "SpaceX Launch", "Meme Coins", "Met Gala", "Game Update 2.0",
    "Open Source Drama", "HBM Prices", "Indie Film Drop", "Esports Finals", "AI Code Copilot"
]
summary_bank = {
    "#HeatWave": "Record temperatures and grid strain warnings",
    "NPC Meme": "Creators reviving NPC-style streams and catchphrases",
    "Swift Tour": "Surprise acoustic set causing fan buzz",
    "AI Chip Shortage": "GPU backorders impacting labs and indie builders",
    "Transfer Deadline": "Last-minute negotiations pushing valuations",
    "#ClimateStrike": "Youth-led protests for climate action",
    "SpaceX Launch": "Successful deployment of new satellite batch",
    "Meme Coins": "Volatile price swings spark online jokes",
    "Met Gala": "Celebrity outfits dominating social feeds",
    "Game Update 2.0": "Major patch with new maps and weapons",
    "Open Source Drama": "Maintainers debating license changes",
    "HBM Prices": "Rumors of tight supply and price hikes",
    "Indie Film Drop": "Festival favorite finally hits OTT",
    "Esports Finals": "Underdog run triggers highlight reels",
    "AI Code Copilot": "Dev productivity debates after new release"
}
platforms = ["Twitter", "Reddit", "YouTube", "TikTok"]

# pick topics
topics = random.sample(topic_bank, n_topics)

# ---------- Generate “live” posts ----------
def gen_posts():
    rows = []
    now = datetime.now()
    for t in topics:
        k = random.randint(max(2, posts_per_topic-1), posts_per_topic+1)
        for _ in range(k):
            ts = now - timedelta(minutes=random.randint(2, 360))
            likes = random.randint(60, 1200)
            reposts = random.randint(10, 400)
            comments = random.randint(5, 200)
            rows.append({
                "platform": random.choice(platforms),
                "topic": t,
                "likes": likes,
                "reposts": reposts,
                "comments": comments,
                "engagement": likes + reposts + comments,
                "summary": summary_bank.get(t, "Conversation gaining traction"),
                "timestamp": ts
            })
    return pd.DataFrame(rows)

df = gen_posts()

# ---------- Trend scoring ----------
now = df["timestamp"].max()
def score_topic(g: pd.DataFrame):
    volume = len(g)
    recent = (g["timestamp"] > now - timedelta(hours=6)).sum()
    engagement = g["engagement"].sum()
    return 0.7 * np.log1p(engagement) + 0.2 * recent + 0.1 * volume

scores = df.groupby("topic").apply(score_topic).sort_values(ascending=False).rename("score").to_frame()

# ---------- Top Cards ----------
st.subheader("🔥 Top Trending Topics")
top3 = scores.head(3).join(df.groupby("topic")["engagement"].sum().rename("total_eng"))
cols = st.columns(3)
for i, (topic, row) in enumerate(top3.iterrows()):
    with cols[i]:
        st.metric(
            label=topic,
            value=f"{int(row['total_eng']):,} engagements",
            delta=f"+{random.randint(4,18)}% vs last hr"
        )

# ---------- All topics table ----------
st.subheader("📊 All Trending Topics (Mock Live)")
table = df.sort_values("engagement", ascending=False)[
    ["platform", "topic", "likes", "reposts", "comments", "engagement", "summary", "timestamp"]
]
st.dataframe(table, use_container_width=True, hide_index=True)

# ---------- Detail panel ----------
st.subheader("🧭 Topic Explanation")
selected = st.selectbox("Select a topic", options=list(scores.index))
topic_df = df[df["topic"] == selected].copy().sort_values(["engagement", "timestamp"], ascending=[False, False])

# confidence proxy
median_eng = int(topic_df["engagement"].median()) if not topic_df.empty else 0
conf = round(min(0.98, 0.5 + (np.log10(median_eng+1)/3)), 2)

left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown(f"### {selected}")
    st.write("**What it is:**", summary_bank.get(selected, "Conversation gaining traction across platforms."))
    st.write("**Why it matters:** Rising engagement and cross-platform spread indicate momentum.")
    st.write("**Context:** Activity clustered in the last few hours with multiple high-impact posts.")
    st.markdown(f"**RAG Confidence (proxy):** `{conf}`")
    st.caption("Confidence is a simple proxy using median engagement — replace with real eval metric later.")

    # mini trend line
    st.markdown("**Trend growth (last 12 hours)**")
    hours = pd.date_range(end=datetime.now(), periods=12, freq="H")
    base = max(50, topic_df["engagement"].sum() // 20)
    noise = np.abs(np.random.normal(0, base*0.2, size=len(hours))).astype(int)
    walk = np.maximum(0, np.cumsum(noise) + base).astype(int)
    trend_df = pd.DataFrame({"time": hours, "engagement": walk}).set_index("time")
    st.line_chart(trend_df)

with right:
    st.markdown("**Top Posts**")
    show_cols = ["platform", "likes", "reposts", "comments", "engagement", "summary", "timestamp"]
    st.dataframe(topic_df[show_cols], use_container_width=True, hide_index=True)

# ---------- Footer ----------
st.markdown("---")
st.caption(
    "Mock live data for demo — backend is API-ready. Replace generator with Reddit/Twitter/YouTube fetchers and integrate vector DB for RAG."
)
