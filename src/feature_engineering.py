import pandas as pd
import ta

def add_technical_indicators(df):
    df = df.copy()

    # Ensure Close is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    df.dropna(inplace=True)

    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)

    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    
    df['MACD'] = ta.trend.macd(df['Close'])

    df['Return'] = df['Close'].pct_change()
    
    df['Volatility_20'] = df['Return'].rolling(window=20).std()

    df.dropna(inplace=True)
    return df
