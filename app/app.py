import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
import pytz

# Add project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


def get_market_status():
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    market_open = time(9, 15)
    market_close = time(15, 30)

    # Weekend check
    if now.weekday() >= 5:  # Saturday (5), Sunday (6)
        return "CLOSED", "Weekend – Market Closed"

    # Before market open
    if now.time() < market_open:
        return "CLOSED", "Market has not opened yet"

    # Market open
    if market_open <= now.time() <= market_close:
        return "OPEN", "Market is currently open"

    # After market close
    return "CLOSED", "Market closed for the day"



@st.cache_data(ttl=1800)  # refresh every 30 minutes
def fetch_latest_data_and_update_csv(symbol):
    df = yf.download(
        symbol,
        start="2018-01-01",
        end=datetime.today().strftime("%Y-%m-%d"),
        progress=False
    )
    df.reset_index(inplace=True)

    # SAFETY CHECK
    if df.empty:
        raise ValueError("Yahoo Finance returned empty data")

    # Save ONLY valid data
    data_path = os.path.join(BASE_DIR, "data", "raw_data.csv")
    df.to_csv(data_path, index=False)

    return df

symbol = "RELIANCE.NS"

try:
    df = fetch_latest_data_and_update_csv(symbol)
    source = "Live"
except Exception:
    st.warning("Live data unavailable. Using last saved CSV.")
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw_data.csv"))
    source = "CSV"
st.caption(f"📡 Data source: {source}")

# Convert Date column to datetime (safety)
df['Date'] = pd.to_datetime(df['Date'])



from src.feature_engineering import add_technical_indicators
from src.model import train_model

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Movement Predictor")
st.markdown("Predicts whether the stock price will go **UP or DOWN** the next day.")
st.markdown("⚠️ *For educational purposes only*")

status, reason = get_market_status()

if status == "OPEN":
    st.success("🟢 Market is OPEN")
elif "not opened" in reason.lower():
    st.info("🕒 Market has not opened yet")
else:
    st.error("🔴 Market is CLOSED")

st.caption(reason)

# Fix yfinance MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Clean numeric columns
price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
for col in price_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)

# Feature engineering
df = add_technical_indicators(df)

# Target
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)
MIN_ROWS = 120


# Train model
if len(df) < MIN_ROWS:
    st.error(
        f"Not enough data to train the model "
        f"({len(df)} rows available, minimum required: {MIN_ROWS})."
    )
    st.stop()

model, accuracy = train_model(df)


# Latest data point
latest = df.iloc[-1:]
features = ['SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'Return', 'Volatility_20']

prediction = model.predict(latest[features])[0]
probability = model.predict_proba(latest[features])[0].max()

# Display result
st.subheader("🔮 Prediction for Next Trading Day")

if prediction == 1:
    st.success(f"📈 Stock is likely to go UP (Confidence: {probability:.2f})")
else:
    st.error(f"📉 Stock is likely to go DOWN (Confidence: {probability:.2f})")

st.metric("Model Accuracy", f"{accuracy*100:.2f}%")


# Set Date as index for plotting
df_plot = df.set_index('Date')
st.subheader("📊 Price Analysis")
st.caption("Candlestick chart with EMA & SMA indicators")

fig = go.Figure()

# Candlestick chart
fig.add_trace(go.Candlestick(
    x=df_plot.index,
    open=df_plot['Open'],
    high=df_plot['High'],
    low=df_plot['Low'],
    close=df_plot['Close'],
    name="Price"
))

# EMA 20
fig.add_trace(go.Scatter(
    x=df_plot.index,
    y=df_plot['EMA_20'],
    mode='lines',
    name='EMA 20',
    line=dict(color='orange', width=1.5)
))

# SMA 20
fig.add_trace(go.Scatter(
    x=df_plot.index,
    y=df_plot['SMA_20'],
    mode='lines',
    name='SMA 20',
    line=dict(color='red', width=1.5)
))

fig.update_layout(
    title="Stock Price (Candlestick) with Moving Averages\n",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    template="plotly_dark",
    height=600,
    dragmode="pan",
    margin=dict(l=60, r=40, t=60, b=90),

    # Enable zoom & range controls
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        ),
        rangeslider=dict(visible=True),  # 👈 THIS IS THE KEY
        type="date"
    ),
    yaxis=dict(
        title="Price (₹)",
        autorange=True,
        fixedrange=False  # 👈 allows Y-axis zoom too
    )
)


st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "scrollZoom": True,     # 🖱️ mouse wheel zoom
        "displayModeBar": True,
        "displaylogo": False
    }
)