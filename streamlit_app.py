import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import streamlit as st

st.set_page_config(page_title="ChartIntel", page_icon="📈", layout="wide")
st.title("📈 ChartIntel — Image-based Chart Helper")
st.write(
    "Upload a TradingView screenshot and get a **plain‑English trade plan**: direction bias, entry, stop, two targets, and risk-based position sizing. This is **educational only** — not financial advice."
)

with st.expander("How to use (quick 3 steps)", expanded=True):
    st.markdown(
        """
        1. **Upload a TradingView screenshot** (make sure the price pane is visible).
        2. *(Optional)* Enter **Account Size** and **Risk %** so we calculate **position size** for you.
        3. Read your **Trade Plan** box and decide if it fits your rules.
        """
    )

with st.sidebar:
    st.header("Options")
    rr1 = st.number_input("RR #1 (e.g., 1.5)", value=1.5, min_value=0.5, max_value=5.0, step=0.1)
    rr2 = st.number_input("RR #2 (e.g., 3.0)", value=3.0, min_value=0.5, max_value=10.0, step=0.1)
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
        base_conf = min(1.0, abs(a) / 0.15)
        conf = float(min(1.0, 0.3 + 0.7 * (0.5 * base_conf + 0.5 * min(1.0, ys.size / 5000.0))))
        return "long", conf
    elif a > 0.02:
        base_conf = min(1.0, abs(a) / 0.15)
        conf = float(min(1.0, 0.3 + 0.7 * (0.5 * base_conf + 0.5 * min(1.0, ys.size / 5000.0))))
        return "short", conf
    else:
        return "flat", 0.4

direction, confidence = detect_bias(image)

if last_price > 0:
    entry = float(last_price)
    risk_points = max(0.005 * entry, 0.01 * entry)
    if direction == "long":
        stop = entry - risk_points
        tp1 = entry + (entry - stop) * rr1
        tp2 = entry + (entry - stop) * rr2
    elif direction == "short":
        stop = entry + risk_points
        tp1 = entry - (stop - entry) * rr1
        tp2 = entry - (stop - entry) * rr2
    else:
        stop = entry - risk_points
        tp1 = entry + (entry - stop) * rr1
        tp2 = entry + (entry - stop) * rr2
else:
    entry = 100.25
    if direction == "long":
        stop = 98.75
    elif direction == "short":
        stop = 101.75
    else:
        stop = 99.75
    tp1 = entry + (entry - stop) * (rr1 if direction != "short" else -rr1)
    tp2 = entry + (entry - stop) * (rr2 if direction != "short" else -rr2)

annot = image.copy()
draw = ImageDraw.Draw(annot)
levels = {
    "ENTRY": (entry, (0, 180, 0)),
    "STOP": (stop, (200, 0, 0)),
    "TP1": (tp1, (0, 120, 255)),
    "TP2": (tp2, (0, 120, 255)),
}
ys = [int(h * 0.55), int(h * 0.7), int(h * 0.4), int(h * 0.3)] if direction != "short" else [int(h * 0.45), int(h * 0.3), int(h * 0.6), int(h * 0.7)]
for (label, (val, color)), y in zip(levels.items(), ys):
    draw.line([(0, y), (w, y)], fill=color, width=2)
    try:
        draw.text((w - 220, y - 14), f"{label}: {val:.4f}", fill=color)
    except Exception:
        draw.text((w - 120, y - 14), label, fill=color)

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

st.markdown("\n".join(["- " + ln for ln in plan_lines]))

summary_text = "\n".join(
    [
        "TRADE PLAN (EDU)",
        f"Bias: {direction}",
        f"Entry: {entry:.6f}",
        f"Stop: {stop:.6f}",
        f"TP1: {tp1:.6f}",
        f"TP2: {tp2:.6f}",
        (f"Position size: {units}" if units else "Position size: (set account & risk%)"),
    ]
)

st.download_button("Download Plan (.txt)", data=summary_text, file_name="chartintel_plan.txt")

with st.expander("What to do now (example workflow)", expanded=True):
    st.markdown(
        """
        **If you choose to trade this setup:**
        1. Place a *stop order* at the **Entry** price in the direction of the bias.
        2. Set a **Stop-Loss** at the Stop level shown.
        3. Set **Take-Profits** at TP1 and TP2.
        4. Size the position so the dollar risk ≤ your risk settings.
        5. If price fails and hits Stop first — exit without widening.
        6. If TP1 hits, consider moving stop to breakeven (your rules).

        *This app is educational — you must decide if the setup fits your own plan.*
        """
    )

st.caption("Educational tool — not financial advice. Validate signals and manage risk.")
