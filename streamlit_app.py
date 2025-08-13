import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import streamlit as st

st.set_page_config(page_title="ChartIntel", page_icon="📈", layout="wide")
st.title("📈 ChartIntel — Image-based Chart Helper")

with st.expander("How to use (3 quick steps)", expanded=True):
    st.markdown(
        """
        1) Upload a chart screenshot (main price pane visible).
        2) Enter the **Last traded price** — this anchors entry at the latest bar.
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
    st.stop()
if last_price <= 0:
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
        conf = float(min(1.0, 0.35 + 0.65 * min(1.0, abs(slope) / 0.15)))
        return "buy", conf
    if slope > 0.02:
        conf = float(min(1.0, 0.35 + 0.65 * min(1.0, abs(slope) / 0.15)))
        return "sell", conf
    return "flat", 0.45

def extract_price_path(img):
    arr = np.asarray(img)
    x0 = int(arr.shape[1] * 0.60)
    roi = arr[:, x0:]
    gray = Image.fromarray(roi).convert("L")
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES)).astype(float)
    y_path = np.array([int(np.argmax(edges[:, j])) for j in range(edges.shape[1])], dtype=float)
    if len(y_path) >= 7:
        y_path = np.convolve(y_path, np.ones(7)/7.0, mode="same")
    return y_path

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
entry_y = int(y_path[-1]) if len(y_path) else int(h * 0.5)
price_entry = float(last_price)
atr_px = atr_pixels(y_path, int(atr_window))
vol_dist = int(round(atr_mult * atr_px))
if action == "buy":
    stop_y = entry_y + vol_dist
elif action == "sell":
    stop_y = entry_y - vol_dist
else:
    stop_y = entry_y + vol_dist
risk_px = abs(entry_y - stop_y)
if action == "buy":
    tp1_y = entry_y - rr1 * risk_px
    tp2_y = entry_y - rr2 * risk_px
elif action == "sell":
    tp1_y = entry_y + rr1 * risk_px
    tp2_y = entry_y + rr2 * risk_px
else:
    tp1_y = entry_y - rr1 * risk_px
    tp2_y = entry_y - rr2 * risk_px
local_std = float(np.std(y_path[-max(10, int(0.25 * len(y_path))):])) if len(y_path) else (h * 0.02)
pixels_per_percent = max(4.0, local_std / 2.5)
px_to_price_delta = lambda px: (px / pixels_per_percent) * (price_entry * 0.01)
price_stop = float(price_entry - px_to_price_delta(stop_y - entry_y))
price_tp1  = float(price_entry - px_to_price_delta(tp1_y - entry_y))
price_tp2  = float(price_entry - px_to_price_delta(tp2_y - entry_y))
entry, stop, tp1, tp2 = price_entry, price_stop, price_tp1, price_tp2
annot = image.copy()
draw = ImageDraw.Draw(annot)
ys_positions = {"ENTRY": entry_y, "STOP": stop_y, "TP1": tp1_y, "TP2": tp2_y}
colors = {"ENTRY": (0, 180, 0), "STOP": (200, 0, 0), "TP1": (0, 120, 255), "TP2": (0, 120, 255)}
for label in ["ENTRY", "STOP", "TP1", "TP2"]:
    y = int(np.clip(ys_positions[label], 0, h - 1))
    draw.line([(0, y), (w, y)], fill=colors[label], width=2)
    val = entry if label == "ENTRY" else stop if label == "STOP" else tp1 if label == "TP1" else tp2
    draw.text((w - 260, max(0, y - 14)), f"{label}: {val:.4f}", fill=colors[label])
if confidence >= decision_threshold:
    recommendation = "BUY" if action == "buy" else "SELL" if action == "sell" else "WAIT"
else:
    recommendation = "WAIT"
st.image(image, use_container_width=True)
st.image(annot, use_container_width=True)
st.write(f"Recommendation: {recommendation}")
st.write(f"Entry: {entry:.6f}, Stop: {stop:.6f}, TP1: {tp1:.6f}, TP2: {tp2:.6f}")

