import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_vix_data(ticker="^VIX"):
    data = yf.download(ticker, period='1mo', interval='1h')
    data.columns = data.columns.droplevel(1)
    print(data)
    last_close = data['Close'].iloc[-1]

    if last_close >= 20:
        print("⚠️  High volatility detected")
    else:
        print("✅  Normal volatility level")


def get_vix_signal(ticker="^VIX"):
    data = yf.download(ticker, period='1mo', interval='1h')
    data.columns = data.columns.droplevel(1)
    last_close = data['Close'].iloc[-1]

    if last_close > 20:
        return "SELL or HEDGE (high vol)"
    else:
        return "BUY or HOLD (normal vol)"
get_vix_data()

get_vix_signal()