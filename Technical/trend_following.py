import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt



# --------------- Fonction Trend Following ----------------
def trend_following(ticker, slow_sma ,fast_sma,):
    df = vbt.YFData(ticker, period='6mo',interval='4h').get()
    
    close = df['Close']
    high = df['High']
    low = df['Low']

    sma_fast = close.vbt.rolling(window=fast_sma).mean()
    sma_slow = close.vbt.rolling(window=slow_sma).mean()

    adx = vbt.indicators.ADX(high, low, close, window=14).adx

    rsi = vbt.indicators.RSI.run(close).rsi
    cross_signal = "WAIT"
    if sma_fast.vbt.crossed_above(sma_slow).iloc[-1]:
        last_cross = "BUY"
    elif sma_fast.vbt.crossed_below(sma_slow).iloc[-1]:
        last_cross = "SELL"
        
    adx_last_value = adx.iloc[-1]
    if adx_last_value >= 25:
        confiance = "forte"
    elif adx_last_value >= 15:
        confiance = "modérée"
    else:
        confiance = "faible"

    rsi_val = rsi.iloc[-1]
    rsi_signal = None
    if last_cross == "BUY" and rsi_val < 50:
        rsi_signal = "Tendance haussière, RSI < 50 : rebond technique possible"
    elif last_cross == "SELL" and rsi_val > 50:
        rsi_signal = "Tendance baissière, RSI > 50 : pullback probable"
        
    return cross_signal, confiance, rsi_signal

print(trend_following("EURUSD=X",50,20))