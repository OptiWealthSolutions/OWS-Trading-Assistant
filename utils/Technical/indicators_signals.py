import yfinance as yf
import vectorbt as vbt
import pandas as pd
import numpy as np
#from ta.trend import ADXIndicator

def signal_strategy(ticker):
    df = yf.download(ticker, period="1y", interval="1d")
    close = df['Close']

    # ===== INDICATEURS =====

    # SMA Crossover
    sma_fast = vbt.MA.run(close, window=20).ma
    sma_slow = vbt.MA.run(close, window=50).ma

# Utilise la méthode vectorbt native pour détecter le croisement haussier
    sma_signal = sma_fast.vbt.crossed_above(sma_slow)

    # ADX (utilisation directe du module ta)
    # adx_ind = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    # adx_signal = adx_ind.adx().reindex_like(sma_signal) >= 35

    # RSI
    rsi = vbt.RSI.run(close, window=14).rsi
    rsi_signal = rsi.reindex_like(sma_signal) > 70  # ou intégrer survente avec < 30
    # Combine entry signal (RSI > 70, SMA crossover, ADX > 35)
    entries = sma_signal & rsi_signal

    # Exit dès qu'un signal est invalide
    exits = ~(sma_signal & rsi_signal)

    # ===== BACKTEST =====

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0.001,  # 0.1% frais
        slippage=0.0005
    )

    return pf

portfolio = signal_strategy("EURUSD=X")
portfolio.stats()
