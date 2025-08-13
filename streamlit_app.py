import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import streamlit as st

# =====================================
# ChartIntel — Streamlit (clean version)
# =====================================
# - Upload a TradingView screenshot
# - Detect simple bias from right-side slope (no OpenCV)
# - Find recent swing low/high from edge ridge
# - Recommend BUY / SELL / WAIT
# - Compute Entry (last price), Stop (recent swing), TP1/TP2 (RR multiples)
# - Optional: enter last price to output real prices (otherwise relative)
# =====================================

st.set_page_config(page_title="ChartIntel", page_icon="📈", layout="wide")
st.title("📈 ChartIntel — Image-based Chart Helper")
st.write(
    "Upload a TradingView screenshot and get a clear **Buy/Sell/Wait** call with a stop loss and two take profits. "
    "Add the **last price** in the sidebar to get actual price levels. This is **educational only** — not financial advice."
)

with st.expander("How to use (3 quick steps)", expanded=True):
    st.markdown(
        """
        1) Upload a chart screenshot (price panel visible).

        2) *(Optional)* Enter **Last traded price** so levels are in real prices.

        3) Read the **Trade Plan** for Entry/Stop/TPs and a BUY/SELL/WAIT call.
        """
    )

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Options")
    rr1 = st.number_input("RR #1", value=1.5, step=0.1)
    rr2 = st.number_input("RR #2", value=3.0, step=0.1)
    decision_threshold = st.slider(
        "Decision threshold (confidence)", 0.30, 0.90, 0.55, 0.01,
        help="Min confidence required to issue BUY/SELL; otherwise WAIT."
    )
    last_price = st.number_input(
        "Optional: Last traded price", value=0.0, min_value=0.0, step=0.01, format="%.6f"
    )
    st.caption("Provide last price to convert pixel distances into price distances.")

# --------------- File upload ---------------
uploaded = st.file_uploader("Upload TradingView chart image", type=["png","jpg","jpeg","webp"]) 
if not uploaded:
    st.info("⬆️ Choose an image to analyze.")
    st.stop()

# --------------- Load image ---------------
image = Image.open(uploaded).convert("RGB")
w, h = image.size

# --------------- Helpers ---------------

def detect_bias(img: Image.Image):
    """Estimate bias from slope of edges in the right ~35% of the image.
    Returns (action, confidence) where action ∈ {"buy","sell","flat"}.
    """
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
    # Screen y grows downward. Negative slope = down-right = bullish.
    if slope < -0.02:
        base = min(1.0, abs(slope) / 0.15)
        conf = float(min(1.0, 0.35 + 0.65 * base))
        return "buy", conf
    if slope > 0.02:
        base = min(1.0, abs(slope) / 0.15)
        conf = float(min(1.0, 0.35 + 0.65 * base))
        return "sell", conf
    return "flat", 0.45


def extract_price_path(img: Image.Image):
    """Return (x_abs, y_path) sampling the strongest edge per column in right ~40%."""
    arr = np.asarray(img)
    x0 = int(arr.shape[1] * 0.60)
    roi = arr[:, x0:]
    gray = Image.fromarray(roi).convert("L")
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES)).astype(float)
    H, W = edges.shape
    y_path, x_abs = [], []
    for j in range(W):
        col = edges[:, j]
        y = int(np.argmax(col))
        y_path.append(y)
        x_abs.append(x0 + j)
    y_path = np.array(y_path, dtype=float)
    if len(y_path) >= 7:
        y_path = np.convolve(y_path, np.ones(7)/7.0, mode="same")
    return np.array(x_abs), y_path


def find_recent_pivots(y_path, lookback_frac=0.35, window=5):
    """Return (lo_idx, hi_idx) for recent swing low/high in last fraction of columns."""
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

# --------------- Core logic ---------------
action, confidence = detect_bias(image)

x_abs, y_path = extract_price_path(image)
lo_idx, hi_idx = find_recent_pivots(y_path)

# Fallbacks if missing
if lo_idx is None:
    lo_idx = max(0, len(y_path) - 8)
if hi_idx is None:
    hi_idx = max(0, len(y_path) - 12)

# Pixel levels (screen y)
entry_y = int(y_path[-1]) if len(y_path) else int(h * 0.5)
if action == "buy":
    stop_y = int(y_path[lo_idx]) if len(y_path) else entry_y + int(0.02 * h)  # below swing low
elif action == "sell":
    stop_y = int(y_path[hi_idx]) if len(y_path) else entry_y - int(0.02 * h)  # above swing high
else:
    stop_y = entry_y + int(0.02 * h)

# Risk (px) and TPs from RR
risk_px = abs(entry_y - stop_y)
risk_px = max(risk_px, max(4, int(0.015 * h)))  # guardrail
if action == "buy":
    tp1_y = int(entry_y - rr1 * risk_px)
    tp2_y = int(entry_y - rr2 * risk_px)
elif action == "sell":
    tp1_y = int(entry_y + rr1 * risk_px)
    tp2_y = int(entry_y + rr2 * risk_px)
else:
    tp1_y = int(entry_y - rr1 * risk_px)
    tp2_y = int(entry_y - rr2 * risk_px)

# Convert pixel deltas → price if last_price provided
price_entry = price_stop = price_tp1 = price_tp2 = None
if last_price > 0 and len(y_path):
    last_len = max(10, int(0.25 * len(y_path)))
    local_std = float(np.std(y_path[-last_len:]))  # recent vertical volatility
    pixels_per_percent = max(4.0, local_std / 2.5)  # heuristic scale
    def px_to_price_delta(px):
        return (px / pixels_per_percent) * (last_price * 0.01)
    price_entry = float(last_price)
    price_stop = float(last_price - px_to_price_delta(stop_y - entry_y))
    price_tp1  = float(last_price - px_to_price_delta(tp1_y - entry_y))
    price_tp2  = float(last_price - px_to_price_delta(tp2_y - entry_y))

# Values used for display / sizing
if price_entry is not None:
    entry, stop, tp1, tp2 = price_entry, price_stop, price_tp1, price_tp2
else:
    # Relative units if no last_price; numbers still vary with structure
    entry = 1.0000
    norm = max(1.0, h * 0.02)
    d = (stop_y - entry_y) / norm
    if action == "buy":
        stop = entry - abs(d)
        tp1 = entry + rr1 * abs(entry - stop)
        tp2 = entry + rr2 * abs(entry - stop)
    elif action == "sell":
        stop = entry + abs(d)
        tp1 = entry - rr1 * abs(stop - entry)
        tp2 = entry - rr2 * abs(stop - entry)
    else:
        stop = entry - 0.02
        tp1 = entry + rr1 * (entry - stop)
        tp2 = entry + rr2 * (entry - stop)

# --------------- Visuals ---------------
annot = image.copy()
draw = ImageDraw.Draw(annot)
ys_positions = {"ENTRY": entry_y, "STOP": stop_y, "TP1": tp1_y, "TP2": tp2_y}
colors = {"ENTRY": (0, 180, 0), "STOP": (200, 0, 0), "TP1": (0, 120, 255), "TP2": (0, 120, 255)}
for label in ["ENTRY", "STOP", "TP1", "TP2"]:
    y = int(np.clip(ys_positions[label], 0, h - 1))
    draw.line([(0, y), (w, y)], fill=colors[label], width=2)
    if price_entry is not None:
        val = entry if label == "ENTRY" else stop if label == "STOP" else tp1 if label == "TP1" else tp2
        txt = f"{label}: {val:.4f}"
    else:
        val = entry if label == "ENTRY" else stop if label == "STOP" else tp1 if label == "TP1" else tp2
        txt = f"{label}: rel {val:.4f}"
    draw.text((w - 240, max(0, y - 14)), txt, fill=colors[label])

# --------------- Recommendation ---------------
if confidence >= decision_threshold:
    if action == "buy":
        recommendation = "BUY"
    elif action == "sell":
        recommendation = "SELL"
    else:
        recommendation = "WAIT"
else:
    recommendation = "WAIT"

emoji = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⏸️"}
color = {"BUY": "#16a34a", "SELL": "#dc2626", "WAIT": "#6b7280"}
bg = {"BUY": "#ecfdf5", "SELL": "#fef2f2", "WAIT": "#f3f4f6"}
st.markdown(
    f"""
    <div style='padding:14px;border-radius:12px;border:2px solid {color[recommendation]};
               background:{bg[recommendation]};text-align:center;font-size:26px;font-weight:700;'>
      {emoji[recommendation]} {recommendation}
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------- Layout ---------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Uploaded Chart")
    st.image(image, use_container_width=True)
with col2:
    st.subheader("Annotated Plan")
    st.image(annot, use_container_width=True)

st.markdown("---")

# --------------- Metrics ---------------
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Confidence", f"{int(confidence*100)}%")
with m2:
    rr_display = abs(entry - tp1) / abs(entry - stop) if stop != entry else None
    st.metric("RR to TP1", f"{rr_display:.2f}x" if rr_display is not None else "—")
with m3:
    st.metric("Risk distance", f"{abs(entry-stop):.4f}" if price_entry is not None else "relative")

# --------------- Order text ---------------
if recommendation == "WAIT":
    st.info("**WAIT** — confidence below threshold or no clear trend detected.")
else:
    side_word = "BUY" if recommendation == "BUY" else "SELL"
    st.success(
        f"Place **{side_word}** at **{entry:.4f}**, Stop-Loss at **{stop:.4f}**, TP1 **{tp1:.4f}**, TP2 **{tp2:.4f}**."
    )

# --------------- Downloadable plan ---------------
summary_text = "\n".join([
    "TRADE PLAN (EDU)",
    f"Recommendation: {recommendation}",
    f"Confidence: {int(confidence*100)}%",
    f"Entry: {entry:.6f}",
    f"Stop: {stop:.6f}",
    f"TP1: {tp1:.6f}",
    f"TP2: {tp2:.6f}",
])

st.download_button("Download Plan (.txt)", data=summary_text, file_name="chartintel_plan.txt")

st.caption("Educational tool — not financial advice. Validate signals and manage risk.")
