from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yfinance as yf

def prepare_dataset_signal(spread, zscore, pair1_close, gold_price, seuil=1):
    # Forcer Series 1D sans squeeze
    if isinstance(pair1_close, pd.DataFrame):
        pair1_close = pair1_close.iloc[:, 0]
    if isinstance(gold_price, pd.DataFrame):
        gold_price = gold_price.iloc[:, 0]


    # Align index
    pair1_close = pair1_close.reindex(spread.index)
    gold_price = gold_price.reindex(spread.index)
    zscore = zscore.reindex(spread.index)

    # Calcul RSI sur pair1_close
    rsi_pair1 = RSIIndicator(close=pair1_close, window=14).rsi()

    # Construire DataFrame features
    df = pd.DataFrame({
        'spread': spread,
        'z_score': zscore,
        'z_score_lag1': zscore.shift(1),
        'vol_spread': spread.rolling(30).std(),
        'rsi_pair1': rsi_pair1,
    })

    df.dropna(inplace=True)

    # Conversion explicite en float
    for col in ['z_score', 'z_score_lag1', 'vol_spread', 'rsi_pair1', 'adx']:
        df[col] = df[col].astype(float)

    # Target : -1 si z_score > seuil, +1 si z_score < -seuil, sinon 0
    df['target'] = 0
    df.loc[df['z_score'] > seuil, 'target'] = -1
    df.loc[df['z_score'] < -seuil, 'target'] = 1

    X = df[['z_score', 'z_score_lag1', 'vol_spread', 'rsi_pair1', 'adx']]
    y = df['target']

    return X, y

def main():
    # Télécharger les données
    pair1_data = yf.download("EURUSD=X", period="1y")
    pair1_close = pair1_data["Close"]
    pair2_close = yf.download("GBPUSD=X", period="1y")["Close"]
    gold_price = yf.download("GC=F", period="1y")["Close"]

    # Calculer le spread et le z-score
    spread = pair1_close - pair2_close
    rolling_mean = spread.rolling(30).mean()
    rolling_std = spread.rolling(30).std()
    zscore = (spread - rolling_mean) / rolling_std

    # Calcul réel de l'ADX sur pair1_close
    # adx_indicator = ADXIndicator(high=pair1_data['High'], low=pair1_data['Low'], close=pair1_data['Close'], window=14)
    # adx_series = adx_indicator.adx()
    # adx_series = adx_series.astype(float)

    # Préparer le dataset
    X, y = prepare_dataset_signal(spread, zscore, pair1_close, gold_price)

    # Convertir en numpy arrays pour scikit-learn
    X = X.to_numpy()
    y = y.to_numpy().ravel()  # 1D array

    # Pipeline de modèle
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )

    # Entraîner le modèle
    model.fit(X, y)

    # Prédire sur les données d'entraînement
    y_pred = model.predict(X)

    # Évaluer
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")

    # Visualiser la cible vs la prédiction
    plt.figure(figsize=(10,6))
    plt.plot(y, label='Target (réelle)')
    plt.plot(y_pred, label='Prédiction')
    plt.legend()
    plt.title("Régression polynomiale degré 2 - Target vs Prédiction")
    plt.show()

if __name__ == "__main__":
    main()