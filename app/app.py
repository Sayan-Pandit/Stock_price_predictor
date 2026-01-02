import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px


# Add project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.feature_engineering import add_technical_indicators
from src.model import train_model

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Movement Predictor")
st.markdown("Predicts whether the stock price will go **UP or DOWN** the next day.")
st.markdown("⚠️ *For educational purposes only*")

# Load data
DATA_PATH = os.path.join(BASE_DIR, "data", "raw_data.csv")
df = pd.read_csv(DATA_PATH)

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

# Train model
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

# Convert Date column to datetime (safety)
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as index for plotting
df_plot = df.set_index('Date')

# Plot with Date on X-axis
fig = px.line(
    df_plot,
    x=df_plot.index,
    y=['Close', 'EMA_20', 'SMA_20'],
    labels={
        "value": "Stock Price (₹)",
        "Date": "Date",
        "variable": "Legend"
    },
    title="Stock Price with 20-Day Moving Averages"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Stock Price (₹)",
    legend_title="Indicators",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

