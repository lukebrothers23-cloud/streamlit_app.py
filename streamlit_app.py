import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import streamlit as st

st.set_page_config(page_title="ChartIntel", page_icon="📈", layout="wide")
st.title("📈 ChartIntel — Image-based Chart Helper")
st.write(
    "Upload a TradingView screenshot and get a clear **Buy/Sell signal** with stop loss, take profit targets, and position sizing. This is **educational only** — not financial advice."
)

with st.expander("How to use (quick 3 steps)", expanded=True):
    st.markdown(
        """
        1. **Upload a TradingView screenshot** (make sure the price pane is visible).
        2. *(Optional)* Enter **Account Size** and **Risk %** so we calculate **position size** for you.
        3. Read your **Trade Plan** box to see if you should Buy or Sell, and where to set stop and take profit.
        """
    )

with st.sidebar:
    st.header("Options")
    rr1 = st.number_input("RR #1 (e.g., 1.5)", value=1.5, min_value=0.5, max_value=5.0, step=0.1)
    rr2 = st.number_input("RR #2 (e.g., 3.0)", value=3.0, min_value=0.5, max_value=10.0, step=0.1)
    decision_threshold = st.slider("Decision threshold (confidence)", min_value=0.3, max_value=0.9, value=0.55, step=0.01, help="Minimum confidence required to issue a BUY/SELL. Below this, the app will say WAIT.")
    last_price = st.number_input("Optional: Last traded price", value=0.0, min_value=0.0, step=0.01, format="%.6f")
    st.caption("If provided, we convert pixel distances to price distances using the last close as a reference.")

    st.divider()
    st.header("Risk Settings (optional)")
    account_size = st.number_input("Account size ($)", value=0.0, min_value=0.0, step=100.0)
    risk_pct = st.number_input("Risk per trade (%)", value=1.0, min_value=0.1, max_value=5.0, step=0.1)
    instrument = st.selectbox("Instrument type", ["Stocks/Shares", "Futures", "Crypto"])
    if instrument == "Futures":
        tick_value = st.number_input("Tick value ($/tick)", value=12.5, step=0.1)
        ticks_per_point = st.number_input("Ticks per point", value=4, step=1)
        point_value = tick_value * ticks_per_point
    else:
        point_value = 1.0

uploaded = st.file_uploader("Upload TradingView chart image", type=["png", "jpg", "jpeg", "webp"])
if not uploaded:
    st.info("⬆️ Choose an image to analyze.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
w, h = image.size

def detect_bias(img: Image.Image):
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
    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    if a < -0.02:
        return "buy", min(1.0, abs(a) / 0.15)
    elif a > 0.02:
        return "sell", min(1.0, abs(a) / 0.15)
    else:
        return "flat", 0.4

action, confidence = detect_bias(image)

if last_price > 0:
    entry = float(last_price)
    risk_points = max(0.005 * entry, 0.01 * entry)
    if action == "buy":
        stop = entry - risk_points
        tp1 = entry + (entry - stop) * rr1
        tp2 = entry + (entry - stop) * rr2
    elif action == "sell":
        stop = entry + risk_points
        tp1 = entry - (stop - entry) * rr1
        tp2 = entry - (stop - entry) * rr2
    else:
        stop = entry - risk_points
        tp1 = entry + (entry - stop) * rr1
        tp2 = entry + (entry - stop) * rr2
else:
    entry = 100.25
    if action == "buy":
        stop = 98.75
    elif action == "sell":
        stop = 101.75
    else:
        stop = 99.75
    tp1 = entry + (entry - stop) * (rr1 if action != "sell" else -rr1)
    tp2 = entry + (entry - stop) * (rr2 if action != "sell" else -rr2)

annot = image.copy()
draw = ImageDraw.Draw(annot)
levels = {
    "ENTRY": (entry, (0, 180, 0)),
    "STOP": (stop, (200, 0, 0)),
    "TP1": (tp1, (0, 120, 255)),
    "TP2": (tp2, (0, 120, 255)),
}
ys = [int(h * 0.55), int(h * 0.7), int(h * 0.4), int(h * 0.3)] if action != "sell" else [int(h * 0.45), int(h * 0.3), int(h * 0.6), int(h * 0.7)]
for (label, (val, color)), y in zip(levels.items(), ys):
    draw.line([(0, y), (w, y)], fill=color, width=2)
    draw.text((w - 220, y - 14), f"{label}: {val:.4f}", fill=color)

def calc_position_size(account: float, risk_percent: float, entry_p: float, stop_p: float, per_point_value: float = 1.0):
    if account <= 0 or risk_percent <= 0 or entry_p <= 0 or stop_p <= 0 or per_point_value <= 0:
        return 0, 0.0
    risk_dollars = account * (risk_percent / 100.0)
    stop_distance_points = abs(entry_p - stop_p)
    if stop_distance_points == 0:
        return 0, 0.0
    units = risk_dollars / (stop_distance_points * per_point_value)
    return int(max(0, np.floor(units))), risk_dollars

units, risk_dollars = calc_position_size(account_size, risk_pct, entry, stop, point_value)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Uploaded Chart")
    st.image(image, use_container_width=True)
with col2:
    st.subheader("Annotated Plan")
    st.image(annot, use_container_width=True)

st.markdown("---")
st.subheader("Your Trade Plan (educational)")

# --- Recommendation block (BUY/SELL/WAIT) ---
if confidence >= decision_threshold:
    if direction == "long":
        recommendation = "BUY"
    elif direction == "short":
        recommendation = "SELL"
    else:
        recommendation = "WAIT"
else:
    recommendation = "WAIT"

colA, colB, colC = st.columns(3)
with colA:
    st.metric("Recommendation", recommendation)
with colB:
    st.metric("Confidence", f"{int(confidence*100)}%")
with colC:
    rr_display = abs(entry - tp1) / abs(entry - stop) if stop != entry else None
    st.metric("RR to TP1", f"{rr_display:.2f}x" if rr_display is not None else "—")

# Order instructions
if recommendation == "BUY":
    order_text = f"Place **BUY** stop/limit at **{entry:.4f}**, Stop-Loss at **{stop:.4f}**, Take-Profits at **{tp1:.4f}** and **{tp2:.4f}**."
elif recommendation == "SELL":
    order_text = f"Place **SELL** stop/limit at **{entry:.4f}**, Stop-Loss at **{stop:.4f}**, Take-Profits at **{tp1:.4f}** and **{tp2:.4f}**."
else:
    order_text = "**WAIT** — confidence below threshold or no clear trend."

st.info(order_text)

plan_lines = [
    f"Bias: **{direction.upper()}**  |  Confidence: **{int(confidence*100)}%**",
    f"Entry: **{entry:.4f}**",
    f"Stop: **{stop:.4f}**  (risk distance: {(abs(entry-stop)):.4f})",
    f"Targets: **TP1 {tp1:.4f}** (RR≈{rr1}x), **TP2 {tp2:.4f}** (RR≈{rr2}x)",
]
if account_size > 0 and units > 0:
    unit_label = "contracts" if instrument == "Futures" else ("shares" if instrument == "Stocks/Shares" else "coins")
    plan_lines.append(
        f"Position size: **{units} {unit_label}** (risking ≈ **${risk_dollars:,.2f}**, {risk_pct}% of ${account_size:,.2f})."
    )
else:
    plan_lines.append("(Add Account Size and Risk % in the sidebar to calculate position size.)")

st.markdown("
".join(["- " + ln for ln in plan_lines]))

summary_text = "
".join(
    [
        "TRADE PLAN (EDU)",
        f"Recommendation: {recommendation}",
        f"Bias: {direction}",
        f"Confidence: {int(confidence*100)}%",
        f"Entry: {entry:.6f}",
        f"Stop: {stop:.6f}",
        f"TP1: {tp1:.6f}",
        f"TP2: {tp2:.6f}",
        (f"Position size: {units}" if units else "Position size: (set account & risk%)"),
    ]
)

st.download_button("Download Plan (.txt)", data=summary_text, file_name="chartintel_plan.txt")", data=summary_text, file_name="chartintel_plan.txt")

with st.expander("What to do now (example workflow)", expanded=True):
    st.markdown(
        """
        **If you choose to trade this setup:**
        1. If Action = **BUY**, place a *buy stop order* at the Entry price.
           If Action = **SELL**, place a *sell stop/limit order* at the Entry price.
        2. Set a **Stop Loss** at the Stop level shown.
        3. Set **Take Profits** at TP1 and TP2.
        4. Size the position so the dollar risk ≤ your risk settings.
        5. If price hits Stop first — exit without widening.
        6. If TP1 hits, consider moving stop to breakeven.
        """
    )

st.caption("Educational tool — not financial advice. Validate signals and manage risk.")
