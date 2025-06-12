from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yfinance as yf
from fx_strategy_V2 import *
from sklearn.model_selection import train_test_split


def prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx, seuil=1):
    if isinstance(pair1_close, pd.DataFrame):
        pair1_close = pair1_close.iloc[:, 0]
    if isinstance(gold_price, pd.DataFrame):
        gold_price = gold_price.iloc[:, 0]

    pair1_close = pair1_close.reindex(spread.index)
    gold_price = gold_price.reindex(spread.index)
    zscore = zscore.reindex(spread.index)
    adx = adx.reindex(spread.index)

    rsi_pair1 = RSIIndicator(close=pair1_close, window=14).rsi()

    df = pd.DataFrame({
        'spread': spread,
        'z_score': zscore,
        'z_score_lag1': zscore.shift(1),
        'vol_spread': spread.rolling(30).std(),
        'rsi_pair1': rsi_pair1,
        'adx': adx
    })

    df.dropna(inplace=True)
    df = df.astype(float)

    df['target'] = 0
    df.loc[df['z_score'] > seuil, 'target'] = -1
    df.loc[df['z_score'] < -seuil, 'target'] = 1

    X = df[['z_score', 'z_score_lag1', 'vol_spread', 'rsi_pair1', 'adx']]
    y = df['target']

    return X, y

def main_strat():
    pair1 = "EURUSD=X"
    pair2 = "GBPUSD=X"
    commodity1 = "GC=F"
    commodity2 = "CL=F"

    df1, df2, df_commo1, df_commo2 = get_all_data(pair1, pair2, commodity1, commodity2)

    spread, _, _, _ = engle_granger_test(df1[f"{pair1}_Close"], df2[f"{pair2}_Close"])
    zscore = (spread - spread.mean()) / spread.std()

    pair1_close = df1[f"{pair1}_Close"]
    gold_price = df_commo1[f"{commodity1}_Close"] if not df_commo1.empty else pair1_close

    adx_series = calculate_adx(pair1)
    adx = adx_series.reindex(spread.index).fillna(method="bfill")

    X, y = prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx)

    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Pipeline avec standardisation + régression par descente de gradient
    model = make_pipeline(
        StandardScaler(),
        SGDRegressor(loss='squared_error', max_iter=1000, tol=1e-3, random_state=42)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Évaluation
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R2 score : {r2:.4f}")
    print(f"Mean Squared Error : {mse:.4f}")

    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Vrai signal', marker='o')
    plt.plot(y_pred, label='Prédiction ML', linestyle='--', marker='x')
    plt.title("Régression : Signal prédit vs réel")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main_strat()