import yfinance as yf
import vectorbt as vbt
import pandas as pd
import numpy as np
#from ta.trend import ADXIndicator

def signal_strategy(ticker):
    df = yf.download(ticker, period="10y", interval="1d")
    close = df['Close']

    # ===== INDICATEURS =====

    # SMA Crossover
    sma_fast = vbt.MA.run(close, window=20).ma
    sma_slow = vbt.MA.run(close, window=50).ma

    # Croisement haussier du SMA rapide au-dessus du SMA lent
    sma_signal = sma_fast.vbt.crossed_above(sma_slow)

    # RSI
    rsi = vbt.RSI.run(close, window=14).rsi
    # Signal d'entrée quand RSI est en survente (< 30)
    rsi_signal = rsi.reindex_like(sma_signal) < 30

    # Combine entry signal (SMA crossover ET RSI en survente)
    entries = sma_signal

    # Exit dès qu'un des signaux d'entrée n'est plus valide
    exits = ~(sma_signal)

    # ===== STOP SIZING =====

    def stop_sizing(close, rr=2.0, atr_window=14, atr_mult=1.5):
        high = df['High']
        low = df['Low']
        atr = vbt.ATR.run(high, low, close, window=atr_window).atr
        sl = atr * atr_mult
        tp = sl * rr
        return sl.reindex_like(close), tp.reindex_like(close)

    sl, tp = stop_sizing(close)

    # ===== SIGNALS SL/TP =====
    entry_price = close.copy()
    entry_price[~entries] = np.nan
    entry_price = entry_price.ffill()

    sl_price = entry_price - sl
    tp_price = entry_price + tp

    # Conditions de sortie SL ou TP
    sl_exits = close <= sl_price
    tp_exits = close >= tp_price

    # Combine toutes les sorties possibles
    combined_exits = exits | sl_exits | tp_exits

    # ===== BACKTEST =====
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=combined_exits,
        init_cash=1000,
        fees=0.001
    )

    return pf

portfolio = signal_strategy("AAPL")
portfolio.stats()


import itertools

def optimize_strategy(ticker, rr_range, sma_fast_range, sma_slow_range, atr_mult_range):
    df = yf.download(ticker, period="10y", interval="1d")
    close = df['Close']
    high = df['High']
    low = df['Low']

    results = []

    for rr, sma_fast_w, sma_slow_w, atr_mult in itertools.product(rr_range, sma_fast_range, sma_slow_range, atr_mult_range):
        if sma_fast_w >= sma_slow_w:
            continue  # invalid crossover

        # Indicateurs
        sma_fast = vbt.MA.run(close, window=sma_fast_w).ma
        sma_slow = vbt.MA.run(close, window=sma_slow_w).ma
        sma_signal = sma_fast.vbt.crossed_above(sma_slow)

        entries = sma_signal
        exits = ~sma_signal

        # ATR pour SL/TP
        atr = vbt.ATR.run(high, low, close, window=14).atr
        sl = atr * atr_mult
        tp = sl * rr

        entry_price = close.copy()
        entry_price[~entries] = np.nan
        entry_price = entry_price.ffill()

        sl_price = entry_price - sl
        tp_price = entry_price + tp

        # Ensure sl_price and tp_price are Series with 1D shape
        if isinstance(sl_price, pd.DataFrame):
            sl_price = sl_price.iloc[:, 0]
        if isinstance(tp_price, pd.DataFrame):
            tp_price = tp_price.iloc[:, 0]
        if isinstance(sl_price, np.ndarray):
            sl_price = pd.Series(sl_price.flatten(), index=close.index)
        if isinstance(tp_price, np.ndarray):
            tp_price = pd.Series(tp_price.flatten(), index=close.index)

        # Align close with sl_price and tp_price before comparison
        # Combine tous en DataFrame pour forcer l'alignement
        df_compare = pd.DataFrame({
            "close": close,
            "sl_price": sl_price,
            "tp_price": tp_price
        }).dropna()

        sl_exits = df_compare["close"] <= df_compare["sl_price"]
        tp_exits = df_compare["close"] >= df_compare["tp_price"]

        # Crée une série booléenne alignée avec l'index d'origine
        sl_exits_full = pd.Series(False, index=close.index)
        tp_exits_full = pd.Series(False, index=close.index)
        sl_exits_full.loc[df_compare.index] = sl_exits
        tp_exits_full.loc[df_compare.index] = tp_exits

        combined_exits = exits | sl_exits_full | tp_exits_full

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=combined_exits,
            init_cash=1000,
            fees=0.001
        )

        stats = pf.stats()
        results.append((rr, sma_fast_w, sma_slow_w, atr_mult, stats['Total Return [%]']))

    results.sort(key=lambda x: x[-1], reverse=True)
    return results

best_params = optimize_strategy("AAPL", rr_range=[1, 2], sma_fast_range=[10, 20], sma_slow_range=[40, 50], atr_mult_range=[1.0, 1.5, 2.0])
print("Top strategies:", best_params[:3])