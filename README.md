# 📈 Stock Price Movement Predictor (ML + Web App)

A machine learning–based web application that predicts whether a stock’s price will **go UP or DOWN the next trading day**, using historical market data and technical indicators.  
The project integrates **data science, machine learning, and a Streamlit web interface**.

> ⚠️ This project is for **educational purposes only** and is **not financial advice**.

---
🔗 **Live Demo:** https://your-username-stock-price-predictor.streamlit.app

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-red?logo=streamlit)](https://stock-pricepredictor.streamlit.app/)

---
## 🔍 Project Overview

Stock markets are highly dynamic and noisy.  
This project aims to:
- Analyze historical stock data
- Extract meaningful technical indicators
- Train a machine learning model
- Provide next-day **price movement prediction (Up/Down)** via a web app

---

## 🗂️ Project Structure

```

Stock_price_predictor/
│
├── app/
│   └── app.py              # Streamlit web application
│
├── data/
│   └── raw_data.csv        # Historical stock data
│
├── notebooks/
│   └── EDA.ipynb           # Exploratory Data Analysis
│
├── src/
│   ├── data_loader.py      # Data collection script
│   ├── feature_engineering.py  # Technical indicators
│   └── model.py            # ML model training
│
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

```

---

## 📊 Dataset

- **Source:** Yahoo Finance (`yfinance`)
- **Date Range:**  
  **01-01-2018 to 31-12-2025**
- **Features:**
  - Date
  - Open
  - High
  - Low
  - Close
  - Adj Close
  - Volume

---

## 🧹 Data Preprocessing

- Converted all price columns to numeric values
- Removed non-numeric and invalid rows
- Ensured data suitability for rolling calculations
- Prevented data leakage by using only past data

---

## ⚙️ Feature Engineering

The following **technical indicators** were generated using the `ta` library:

| Indicator | Purpose |
|---------|--------|
| SMA (20) | Medium-term trend |
| EMA (20) | Short-term momentum |
| RSI (14) | Overbought / oversold |
| MACD | Trend & momentum |
| Daily Return | Price direction |
| Volatility (20) | Risk measurement |

These features help capture **trend, momentum, direction, and risk**.

---

## 🎯 Target Variable

The problem is framed as a **binary classification task**:

- `1` → Price goes **UP** the next trading day  
- `0` → Price goes **DOWN or remains same**

Target logic:
```

Target = Close(t+1) > Close(t)

````

---

## 🤖 Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **Train/Test Split:** Time-series aware (no shuffling)
- **Evaluation Metric:** Accuracy

The model predicts the **next-day price movement** based on engineered features.

---

## 🌐 Web Application (Streamlit)

The Streamlit web app provides:
- Next-day prediction (**UP / DOWN**)
- Model confidence score
- Model accuracy
- Interactive price chart with indicators
- Date-based X-axis
- Labeled Y-axis (Stock Price in ₹)

---

## 📈 Visualization

- Implemented using **Plotly**
- Displays:
  - Closing price
  - SMA (20)
  - EMA (20)
- Axes:
  - **X-axis:** Date
  - **Y-axis:** Stock Price (₹)

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Stock_price_predictor.git
cd Stock_price_predictor
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the web app

```bash
streamlit run app/app.py
```

The app will open automatically in your browser.

---

## 📦 Dependencies

* pandas
* numpy
* yfinance
* ta
* scikit-learn
* streamlit
* plotly

---

## 🧠 Key Learnings

* Time-series data handling
* Technical indicator engineering
* Avoiding data leakage
* ML model integration with web apps
* Robust path handling across environments

---

## 🚀 Future Enhancements

* Add multiple stock selection
* Include market index data (NIFTY 50)
* Add VWAP and advanced indicators
* Improve accuracy with hyperparameter tuning
* Deploy app on Streamlit Cloud
* Integrate live market data

---

## 👨‍💻 Author

**Sayan Pandit**
B.Tech CSE (Data Science)

---

## 📜 Disclaimer

This project is for **academic and learning purposes only**.
It does **not provide financial or investment advice**.

