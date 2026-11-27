"""Per-aspect sentiment report (product-manager view). Run:  streamlit run app.py
Reads results/aggregate_clothing_<method>.csv produced by `python -m src.aggregate --method <method>`.
"""
import glob, os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Aspect sentiment report", layout="wide")
st.title("Per-aspect sentiment report: clothing reviews")
files = sorted(glob.glob("results/aggregate_clothing_*.csv"))
if not files:
    st.warning("No aggregate files found. Run: python -m src.aggregate --method baseline")
    st.stop()
choice = st.sidebar.selectbox("Extractor", [os.path.basename(f)[len("aggregate_clothing_"):-4] for f in files])
min_m = st.sidebar.slider("Minimum mentions", 5, 200, 20)
df = pd.read_csv(f"results/aggregate_clothing_{choice}.csv")
df = df[df["mentions"] >= min_m].sort_values("net_sentiment")
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Net sentiment per aspect  (positive - negative) / mentions")
    st.bar_chart(df.set_index("aspect")["net_sentiment"], horizontal=True, height=max(300, 22 * len(df)))
with c2:
    st.subheader("Counts")
    st.dataframe(df[["aspect", "positive", "negative", "neutral", "mentions", "net_sentiment"]].round(2), hide_index=True)
st.caption("Aspects are canonical categories; each review counts once per category (dominant sentiment), so mentions = number of reviews.")
