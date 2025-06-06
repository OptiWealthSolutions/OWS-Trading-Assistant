import pandas as pd
import yfinance as yf
import numpy as np

def sma_crossing (tickers, fast_period=20, slow_period=50):
    results = []
    for ticker in tickers:
        data = yf.download(ticker, period="3mo", interval="1d")

        df = data[['Close']].copy()
        df['SMA_fast'] = df['Close'].rolling(window=fast_period).mean()
        df['SMA_slow'] = df['Close'].rolling(window=slow_period).mean()
        df['Signal'] = 0
        df['Signal'] = (df['SMA_fast'] > df['SMA_slow']).astype(int)
        df['Position'] = df['Signal'].diff()
        
        if df['Signal'].iloc[-1] == 1:
            signal = "BUY"
        elif df['Signal'].iloc[-1] == 0:
            signal = "SELL"
        else:
            signal = "HOLD"

        print(f"{ticker}: {signal}")

        results.append({
                'Ticker': ticker,
                'Signal': signal
            })
    return pd.DataFrame(results) 

report.to_csv("sma_report_signal.csv", index=False)

