import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

st.set_page_config(page_title="ChartIntel", page_icon="📈", layout="wide")
st.title("📈 ChartIntel — Image-based Chart Helper")
st.write("Upload a TradingView screenshot. We'll estimate trend/swing levels and propose an entry, stop, and targets.")

# Sidebar inputs
with st.sidebar:
    st.header("Options")
    rr1 = st.number_input("RR #1 (e.g., 1.5)", value=1.5, min_value=0.5, max_value=5.0, step=0.1)
    rr2 = st.number_input("RR #2 (e.g., 3.0)", value=3.0, min_value=0.5, max_value=10.0, step=0.1)
    last_price = st.number_input("Optional: Last traded price", value=0.0, min_value=0.0, step=0.01, format="%.6f")
    st.caption("If provided, we convert pixel distances to price distances using the last close as a reference.")

# File uploader
uploaded = st.file_uploader("Upload TradingView chart image", type=["png", "jpg", "jpeg", "webp"])
if not uploaded:
    st.info("⬆️ Choose an image to analyze.")
    st.stop()

# Load image
image = Image.open(uploaded).convert("RGB")
w, h = image.size

# Placeholder logic — replace with real analysis later
direction = "long"
confidence = 0.85
entry = 100.25
stop = 98.75
tp1 = 102.50
tp2 = 105.00

# Annotate image with horizontal lines
annot = image.copy()
draw = ImageDraw.Draw(annot)
for y in [h//2, int(h*0.6), int(h*0.4), int(h*0.3)]:
    draw.line([(0, y), (w, y)], fill=(255, 0, 0), width=2)

# Show images
col1, col2 = st.columns(2)
with col1:
    st.subheader("Uploaded Chart")
    st.image(image, use_column_width=True)
with col2:
    st.subheader("Annotated Plan")
    st.image(annot, use_column_width=True)

# Summary
st.markdown("---")
st.subheader("Analysis Summary")
st.write({
    "direction": direction,
    "confidence": confidence,
    "entry": entry,
    "stop": stop,
    "tp1": tp1,
    "tp2": tp2,
})

st.caption("This tool is for educational use only. Validate with real data before trading.")
