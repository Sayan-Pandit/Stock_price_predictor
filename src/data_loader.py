import yfinance as yf
import pandas as pd
import os

def download_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

if __name__ == "__main__":
    stock_symbol = "RELIANCE.NS"
    start = "2018-01-01"
    end = "2025-12-31"

    # Absolute project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Data directory
    DATA_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    df = download_stock_data(stock_symbol, start, end)

    file_path = os.path.join(DATA_DIR, "raw_data.csv")
    df.to_csv(file_path, index=False)

    print("✅ Data saved successfully at:")
    print(file_path)
