import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import streamlit as st

st.set_page_config(page_title="ChartIntel", page_icon="📈", layout="wide")
st.title("📈 ChartIntel — Image-based Chart Helper (Short-Term Mode)")

with st.expander("How to use (3 quick steps)", expanded=True):
    st.markdown("""
        1) Upload a chart screenshot (main price pane visible).
        2) Enter the **Last traded price** — this anchors entry at the latest bar.
        3) Read the **Trade Plan** for Entry/Stop/TPs and a BUY/SELL/WAIT call.
        """)

with st.sidebar:
    st.header("Options")
    rr1 = st.number_input("RR #1", value=1.0, step=0.1)
    rr2 = st.number_input("RR #2", value=1.5, step=0.1)
    atr_window = st.number_input("Vol lookback (bars)", value=8, min_value=3, max_value=50, step=1)
    atr_mult = st.number_input("Stop volatility ×", value=1.2, min_value=0.5, max_value=3.0, step=0.1)
    last_price = st.number_input("Last traded price (required)", value=0.0, min_value=0.0, step=0.0001, format="%.6f")

uploaded = st.file_uploader("Upload TradingView chart image", type=["png","jpg","jpeg","webp"])
if not uploaded:
    st.info("⬆️ Choose an image to analyze.")
    st.stop()
else:
    st.success(f"✅ Loaded file: **{uploaded.name}**")

if last_price <= 0:
    st.error("Please enter the **Last traded price** (> 0) in the sidebar, then rerun.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
    w, h = image.size
except Exception as e:
    st.error(f"❌ Could not open image: {e}")
    st.stop()

def extract_price_path(img):
    arr = np.asarray(img)
    x0 = int(arr.shape[1] * 0.60)
    roi = arr[:, x0:]
    gray = Image.fromarray(roi).convert("L")
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES)).astype(float)
    H, W = edges.shape
    y_path = [int(np.argmax(edges[:, j])) for j in range(W)]
    y_path = np.array(y_path, dtype=float)
    if len(y_path) >= 5:
        y_path = np.convolve(y_path, np.ones(5)/5.0, mode="same")
    return y_path

def atr_pixels(y_path, window=8):
    if y_path is None or len(y_path) < 2:
        return 4.0
    diffs = np.abs(np.diff(y_path))
    if len(diffs) >= window:
        atr = np.convolve(diffs, np.ones(window)/window, mode="valid")[-1]
    else:
        atr = np.mean(diffs) if len(diffs) else 4.0
    return max(2.0, float(atr))

y_path = extract_price_path(image)

action = "flat"
if len(y_path) >= 3:
    recent = np.diff(y_path[-min(6, len(y_path)):])
    recent_slope = float(np.mean(recent))
    if recent_slope < -0.01:
        action = "buy"
    elif recent_slope > 0.01:
        action = "sell"

entry_y = int(y_path[-1]) if len(y_path) else int(h * 0.5)
price_entry = float(last_price)

atr_px = atr_pixels(y_path, int(atr_window))
vol_dist = int(round(atr_mult * atr_px))

if action == "buy":
    stop_y = entry_y + vol_dist
    risk_px = np.clip(abs(entry_y - stop_y), 2, int(0.06 * h))
    tp1_y = int(entry_y - rr1 * risk_px)
    tp2_y = int(entry_y - rr2 * risk_px)
elif action == "sell":
    stop_y = entry_y - vol_dist
    risk_px = np.clip(abs(entry_y - stop_y), 2, int(0.06 * h))
    tp1_y = int(entry_y + rr1 * risk_px)
    tp2_y = int(entry_y + rr2 * risk_px)
else:
    stop_y = entry_y + vol_dist
    risk_px = np.clip(vol_dist, 2, int(0.06 * h))
    tp1_y = int(entry_y - rr1 * risk_px)
    tp2_y = int(entry_y - rr2 * risk_px)

local_std = float(np.std(y_path[-5:])) if len(y_path) else (h * 0.01)
pixels_per_percent = max(2.0, local_std / 3.0)

def px_to_price_delta(px: float) -> float:
    return (px / pixels_per_percent) * (price_entry * 0.01)

price_stop = float(price_entry - px_to_price_delta(stop_y - entry_y))
price_tp1  = float(price_entry - px_to_price_delta(tp1_y - entry_y))
price_tp2  = float(price_entry - px_to_price_delta(tp2_y - entry_y))

annot = image.copy()
draw = ImageDraw.Draw(annot)
colors = {"ENTRY": (0, 180, 0), "STOP": (200, 0, 0), "TP1": (0, 120, 255), "TP2": (0, 120, 255)}
for label, y_val, val in [("ENTRY", entry_y, price_entry), ("STOP", stop_y, price_stop), ("TP1", tp1_y, price_tp1), ("TP2", tp2_y, price_tp2)]:
    yv = int(np.clip(y_val, 0, h - 1))
    draw.line([(0, yv), (w, yv)], fill=colors[label], width=2)
    draw.text((w - 260, max(0, yv - 14)), f"{label}: {val:.4f}", fill=colors[label])

recommendation = "WAIT"
if action == "buy":
    recommendation = "BUY"
elif action == "sell":
    recommendation = "SELL"

emoji = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⏸️"}
color = {"BUY": "#16a34a", "SELL": "#dc2626", "WAIT": "#6b7280"}
bg = {"BUY": "#ecfdf5", "SELL": "#fef2f2", "WAIT": "#f3f4f6"}
st.markdown(f"""
    <div style='padding:14px;border-radius:12px;border:2px solid {color[recommendation]};
               background:{bg[recommendation]};text-align:center;font-size:26px;font-weight:700;'>
      {emoji[recommendation]} {recommendation}
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Uploaded Chart")
    st.image(image, use_container_width=True)
with col2:
    st.subheader("Annotated Plan")
    st.image(annot, use_container_width=True)

st.markdown("---")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("ATR (px)", f"{atr_px:.1f}")
with m2:
    st.metric("Stop buffer (px)", f"{vol_dist}")
with m3:
    rr_display = abs(price_entry - price_tp1) / max(1e-9, abs(price_entry - price_stop))
    st.metric("RR to TP1", f"{rr_display:.2f}x")

if recommendation == "WAIT":
    st.info("**WAIT** — latest bars are flat; no clear bias.")
else:
    st.success(f"{recommendation} **NOW** at **{price_entry:.6f}**. Stop-Loss: **{price_stop:.6f}**. TP1: **{price_tp1:.6f}**. TP2: **{price_tp2:.6f}**.")

summary_text = "\n".join([
    "TRADE PLAN (EDU)",
    f"Recommendation: {recommendation}",
    f"Entry: {price_entry:.6f}",
    f"Stop: {price_stop:.6f}",
    f"TP1: {price_tp1:.6f}",
    f"TP2: {price_tp2:.6f}",
    f"ATR(px): {atr_px:.2f} × {atr_mult} → buffer {vol_dist} px",
])

st.download_button("Download Plan (.txt)", data=summary_text, file_name="chartintel_plan.txt")

st.caption("Educational tool — not financial advice. Validate signals and manage risk.")
