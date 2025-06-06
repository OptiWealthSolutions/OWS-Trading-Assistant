import yfinance as yf
import pandas as pd
import numpy as np

def calculate_adx(ticker, period=14):
    # Télécharge les données boursières
    df = yf.download(ticker, period='5d', interval='1h')

    # Calcul des True Range (TR), +DM et -DM
    df['High'] = df['High'].shift(1)
    df['Low'] = df['Low'].shift(1)
    df['Close'] = df['Close'].shift(1)

    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close']), abs(df['Low'] - df['Close'])))
    df['+DM'] = df['High'] - df['High'].shift(1)
    df['-DM'] = df['Low'].shift(1) - df['Low']

    # Conditions pour filtrer les +DM et -DM
    df['+DM'] = np.where(df['+DM'] > 0, df['+DM'], 0)
    df['-DM'] = np.where(df['-DM'] > 0, df['-DM'], 0)

    # Calcul du smoothed +DM, -DM et TR sur la période donnée
    df['+DM_smooth'] = df['+DM'].rolling(window=period).sum()
    df['-DM_smooth'] = df['-DM'].rolling(window=period).sum()
    df['TR_smooth'] = df['TR'].rolling(window=period).sum()

    # Calcul de l'ADX
    df['+DI'] = (df['+DM_smooth'] / df['TR_smooth']) * 100
    df['-DI'] = (df['-DM_smooth'] / df['TR_smooth']) * 100
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].rolling(window=period).mean()

    # Déduction sur la force de la tendance selon la valeur de l'ADX
    last_adx = df['ADX'].iloc[-1]

    if last_adx < 20:
        interpretation = "Tendance faible ou inexistante"
    elif last_adx < 40:
        interpretation = "Tendance modérée"
    else:
        interpretation = "Tendance forte"

    return last_adx, interpretation

# Exemple d'appel
ticker = "EURUSD=X"
adx_value, interpretation = calculate_adx(ticker)
print(f"ADX pour {ticker}: {adx_value:.2f} — {interpretation}")