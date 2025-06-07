import yfinance as yf
import pandas as pd

def sma_crossing(ticker, fast_period=20, slow_period=50):
    try:
        data = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if data.empty:
            raise ValueError("Données introuvables pour le ticker.")
    except Exception as e:
        print(f"Erreur lors du téléchargement des données pour {ticker} : {e}")
        return pd.DataFrame({'Ticker': [ticker], 'Signal': ['ERROR']})

    df = data[['Close']].copy()
    df['SMA_fast'] = df['Close'].rolling(window=fast_period).mean()
    df['SMA_slow'] = df['Close'].rolling(window=slow_period).mean()
    df['Signal'] = (df['SMA_fast'] > df['SMA_slow']).astype(int)

    # Déterminer le signal final
    last_signal = df['Signal'].iloc[-1]
    if last_signal == 1:
        signal = "BUY"
    elif last_signal == 0:
        signal = "SELL"
    else:
        signal = "HOLD"

    print(f"{ticker}: {signal}")

    return pd.DataFrame({'Ticker': [ticker], 'Signal': [signal]})