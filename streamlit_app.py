python
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import streamlit as st

st.set_page_config(page_title="ChartIntel", page_icon="📈", layout="wide")
st.title("📈 ChartIntel — Image-based Chart Helper")

with st.expander("How to use (3 quick steps)", expanded=True):
    st.markdown(
        """
        1) Upload a chart screenshot (price panel visible).
        2) Enter **Last traded price** so entry is anchored at the latest bar.
        3) Read the **Trade Plan** for Entry/Stop/TPs and a BUY/SELL/WAIT call.
        """
    )

with st.sidebar:
    st.header("Options")
    rr1 = st.number_input("RR #1", value=1.5, step=0.1)
    rr2 = st.number_input("RR #2", value=3.0, step=0.1)
    decision_threshold = st.slider("Decision threshold (confidence)", 0.30, 0.90, 0.55, 0.01)
    last_price = st.number_input("Last traded price (required)", value=0.0, min_value=0.0, step=0.01)

uploaded = st.file_uploader("Upload TradingView chart image", type=["png","jpg","jpeg","webp"])
if not uploaded:
    st.stop()
if last_price <= 0:
    st.error("Please enter the **Last traded price** in the sidebar to proceed.")
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
        return "buy", 0.7
    if slope > 0.02:
        return "sell", 0.7
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
    stop_y = int(y_path[lo_idx])
elif action == "sell":
    stop_y = int(y_path[hi_idx])
else:
    stop_y = entry_y + int(0.02 * h)

risk_px = max(abs(entry_y - stop_y), 4)
if action == "buy":
    tp1_y = int(entry_y - rr1 * risk_px)
    tp2_y = int(entry_y - rr2 * risk_px)
elif action == "sell":
    tp1_y = int(entry_y + rr1 * risk_px)
    tp2_y = int(entry_y + rr2 * risk_px)
else:
    tp1_y = int(entry_y - rr1 * risk_px)
    tp2_y = int(entry_y - rr2 * risk_px)

pixels_per_percent = max(4.0, np.std(y_path[-10:]) / 2.5)
px_to_price_delta = lambda px: (px / pixels_per_percent) * (price_entry * 0.01)

price_stop = price_entry - px_to_price_delta(stop_y - entry_y)
price_tp1  = price_entry - px_to_price_delta(tp1_y - entry_y)
price_tp2  = price_entry - px_to_price_delta(tp2_y - entry_y)

entry, stop, tp1, tp2 = price_entry, price_stop, price_tp1, price_tp2

annot = image.copy()
draw = ImageDraw.Draw(annot)
for label, y in zip(["ENTRY", "STOP", "TP1", "TP2"], [entry_y, stop_y, tp1_y, tp2_y]):
    draw.line([(0, y), (w, y)], fill=(0, 180, 0), width=2)

recommendation = "WAIT"
if confidence >= decision_threshold:
    if action == "buy":
        recommendation = "BUY"
    elif action == "sell":
        recommendation = "SELL"

st.image(annot, use_container_width=True)
summary_text = "\n".join([
    "TRADE PLAN (EDU)",
    f"Recommendation: {recommendation}",
    f"Entry: {entry:.6f}",
    f"Stop: {stop:.6f}",
    f"TP1: {tp1:.6f}",
    f"TP2: {tp2:.6f}",
])

st.download_button("Download Plan (.txt)", data=summary_text, file_name="chartintel_plan.txt")
