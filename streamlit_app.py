python
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import streamlit as st

# =====================================
# ChartIntel — Streamlit App (Latest-Bar Entry + Volatility Stops)
# =====================================

st.set_page_config(page_title="ChartIntel", page_icon="📈", layout="wide")
st.title("📈 ChartIntel — Image-based Chart Helper")

with st.expander("How to use (3 quick steps)", expanded=True):
    st.markdown(
        """
        1) Upload a chart screenshot (with the main price pane visible).
        
        2) Enter the **Last traded price** so entry is anchored at the latest bar.
        
        3) Read the **Trade Plan** for Entry/Stop/TPs and a BUY/SELL/WAIT call.
        """
    )

with st.sidebar:
    st.header("Options")
    rr1 = st.number_input("RR #1", value=1.5, step=0.1)
    rr2 = st.number_input("RR #2", value=3.0, step=0.1)
    atr_window = st.number_input("Vol lookback (bars)", value=14, min_value=5, max_value=200, step=1)
    atr_mult = st.number_input("Stop volatility ×", value=1.8, min_value=0.5, max_value=5.0, step=0.1)
    decision_threshold = st.slider("Decision threshold (confidence)", 0.30, 0.90, 0.55, 0.01)
    last_price = st.number_input("Last traded price (required)", value=0.0, min_value=0.0, step=0.0001, format="%.6f")

uploaded = st.file_uploader("Upload TradingView chart image", type=["png","jpg","jpeg","webp"])
if not uploaded:
    st.info("⬆️ Choose an image to analyze.")
    st.stop()
if last_price <= 0:
    st.error("Please enter the **Last traded price** to proceed.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
w, h = image.size

def detect_bias(img):
    arr = np.asarray(img)
    x1 = int(arr.shape[1] * 0.65)
    roi = arr[:, x1:]
    roi_gray = Image.fromarray(roi).convert("L")
    edges = roi_gray.filter(ImageFilter.FIND_EDGES)
    e = np.asarray(edges)
    mask = e > 20
    ys, xs = np.where(mask)
    if ys.size < 300:
        return "flat", 0.35
    xs = xs + x1
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    if slope < -0.02:
        conf = min(1.0, 0.35 + 0.65 * min(1.0, abs(slope) / 0.15))
        return "buy", conf
    if slope > 0.02:
        conf = min(1.0, 0.35 + 0.65 * min(1.0, abs(slope) / 0.15))
        return "sell", conf
    return "flat", 0.45

def extract_price_path(img):
    arr = np.asarray(img)
    x0 = int(arr.shape[1] * 0.60)
    roi = arr[:, x0:]
    gray = Image.fromarray(roi).convert("L")
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES)).astype(float)
    H, W = edges.shape
    y_path = [int(np.argmax(edges[:, j])) for j in range(W)]
    y_path = np.array(y_path, dtype=float)
    if len(y_path) >= 7:
        y_path = np.convolve(y_path, np.ones(7)/7.0, mode="same")
    return y_path

def find_recent_pivots(y_path, lookback_frac=0.35, window=5):
    n = len(y_path)
    if n == 0:
        return None, None
    start = int(n * (1 - lookback_frac))
    lo_idx = hi_idx = None
    for i in range(max(start, window), n - window):
        seg = y_path[i-window:i+window+1]
        if np.argmin(seg) == window:
            lo_idx = i
        if np.argmax(seg) == window:
            hi_idx = i
    return lo_idx, hi_idx

def atr_pixels(y_path, window=14):
    if y_path is None or len(y_path) < 2:
        return 6.0
    diffs = np.abs(np.diff(y_path))
    if len(diffs) >= window:
        atr = np.convolve(diffs, np.ones(window)/window, mode="valid")[-1]
    else:
        atr = np.mean(diffs) if len(diffs) else 6.0
    return max(4.0, float(atr))

action, confidence = detect_bias(image)
y_path = extract_price_path(image)
lo_idx, hi_idx = find_recent_pivots(y_path)
if lo_idx is None:
    lo_idx = max(0, len(y_path) - 8)
if hi_idx is None:
    hi_idx = max(0, len(y_path) - 12)

entry_y = int(y_path[-1]) if len(y_path) else int(h * 0.5)
price_entry = float(last_price)

if action == "buy":
    swing_stop_y = int(y_path[lo_idx]) if len(y_path) else entry_y + int(0.02 * h)
elif action == "sell":
    swing_stop_y = int(y_path[hi_idx]) if len(y_path) else entry_y - int(0.02 * h)
else:
    swing_stop_y = entry_y + int(0.02 * h)

atr_px = atr_pixels(y_path, int(atr_window))
vol_dist = int(round(atr_mult * atr_px))
if action == "buy":
    stop_y = max(swing_stop_y, entry_y + vol_dist)
elif action == "sell":
    stop_y = min(swing_stop_y, entry_y - vol_dist)
else:
    stop_y = entry_y + vol_dist

risk_px = abs(entry_y - stop_y)
risk_px = max(risk_px, max(4, int(0.015 * h)))
if action == "buy":
    tp1_y = int(entry_y - rr1 * risk_px)
    tp2_y = int(entry_y - rr2 * risk_px)
elif action == "sell":
    tp1_y = int(entry_y + rr1 * risk_px)
    tp2_y = int(entry_y + rr2 * risk_px)
else:
    tp1_y = int(entry_y - rr1 * risk_px)
    tp2_y = int(entry_y - rr2 * risk_px)

last_len = max(10, int(0.25 * len(y_path))) if len(y_path) else 10
local_std = float(np.std(y_path[-last_len:])) if len(y_path) else (h * 0.02)
pixels_per_percent = max(4.0, local_std / 2.5)
px_to_price_delta = lambda px: (px / pixels_per_percent) * (price_entry * 0.01)

price_stop = price_entry - px_to_price_delta(stop_y - entry_y)
price_tp1 = price_entry - px_to_price_delta(tp1_y - entry_y)
price_tp2 = price_entry - px_to_price_delta(tp2_y - entry_y)

entry, stop, tp1, tp2 = price_entry, price_stop, price_tp1, price_tp2

annot = image.copy()
draw = ImageDraw.Draw(annot)
for label, y_val in {"ENTRY": entry_y, "STOP": stop_y, "TP1": tp1_y, "TP2": tp2_y}.items():
    draw.line([(0, y_val), (w, y_val)], fill=(0, 255, 0), width=2)

recommendation = "WAIT"
if confidence >= decision_threshold:
    if action == "buy":
        recommendation = "BUY"
    elif action == "sell":
        recommendation = "SELL"

st.markdown(f"## {recommendation} at {entry:.4f}")
st.image(annot, use_container_width=True)

summary_text = "\n".join([
    "TRADE PLAN (EDU)",
    f"Recommendation: {recommendation}",
    f"Confidence: {int(confidence*100)}%",
    f"Entry: {entry:.6f}",
    f"Stop: {stop:.6f}",
    f"TP1: {tp1:.6f}",
    f"TP2: {tp2:.6f}",
    f"ATR(px): {atr_px:.2f} × {atr_mult} → buffer {vol_dist} px",
])

st.download_button("Download Plan (.txt)", data=summary_text, file_name="chartintel_plan.txt")
st.caption("Educational tool — not financial advice.")
