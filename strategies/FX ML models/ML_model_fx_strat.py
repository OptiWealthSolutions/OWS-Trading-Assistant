# settings.py
tickers = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X"
]

# Example correlation matrix placeholder (empty dictionary)
correlation_matrix = {}

# Optional commodity mapping for currencies
commodity_mapping = {
    "USD": "CL=F",   # Crude Oil
    "EUR": "GC=F",   # Gold
    "GBP": "GC=F",
    "JPY": "SI=F",   # Silver
    "AUD": "XAU=X",  # Gold spot
    "CAD": "CL=F",
    "CHF": "XAG=X",  # Silver spot
    "NZD": "XAU=X"
}

# ML_model_fx_strat.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import settings


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


def run_model_for_pair(pair1, pair2):
    # Determine commodity for pair1 currency from mapping, fallback to pair1 close if none
    base_currency1 = pair1[:3]  # e.g. 'EUR' from 'EURUSD=X'
    commodity1 = settings.commodity_mapping.get(base_currency1, None)
    commodity2 = None  # Not used in current logic but kept for compatibility

    df1, df2, df_commo1, df_commo2 = get_all_data(pair1, pair2, commodity1, commodity2)

    if df1.empty or df2.empty:
        print(f"Data not available for pair {pair1} or {pair2}. Skipping.")
        return None

    spread, _, _, _ = engle_granger_test(df1[f"{pair1}_Close"], df2[f"{pair2}_Close"])
    zscore = (spread - spread.mean()) / spread.std()

    pair1_close = df1[f"{pair1}_Close"]
    gold_price = df_commo1[f"{commodity1}_Close"] if (commodity1 and not df_commo1.empty) else pair1_close

    adx_series = calculate_adx(pair1)
    adx = adx_series.reindex(spread.index).fillna(method="bfill")

    X, y = prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx)

    if X.empty or y.empty:
        print(f"Insufficient data after preparation for pair {pair1} and {pair2}. Skipping.")
        return None

    # Train/test split - use all data except last day for training, last day for prediction
    if len(X) < 2:
        print(f"Not enough data points for pair {pair1} and {pair2}. Skipping.")
        return None

    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    X_pred = X.iloc[-1:]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(multi_class='multinomial', max_iter=1000, solver='lbfgs', class_weight='balanced')
    )
    model.fit(X_train, y_train)

    y_pred_class = model.predict(X_pred)[0]
    y_proba = model.predict_proba(X_pred)[0]

    # Get index of predicted class in classes array to get probability
    classes = model.named_steps['logisticregression'].classes_
    class_index = np.where(classes == y_pred_class)[0][0]
    confidence = y_proba[class_index]

    # Interpret signal
    if y_pred_class == 1:
        signal = 'BUY'
    elif y_pred_class == -1:
        signal = 'SELL'
    else:
        signal = 'WAIT'

    return {
        'pair1': pair1,
        'pair2': pair2,
        'predicted_class': y_pred_class,
        'confidence': confidence,
        'signal': signal
    }


def test_all_pairs():
    results = []
    tickers = settings.tickers
    # Loop over consecutive pairs of tickers
    for i in range(len(tickers) - 1):
        pair1 = tickers[i]
        pair2 = tickers[i + 1]
        result = run_model_for_pair(pair1, pair2)
        if result is not None:
            results.append(result)

    df_results = pd.DataFrame(results)
    print("ML Model Prediction Results for all pairs:")
    print(df_results)
    return df_results


def main_strat():
    # Original main strat kept for backward compatibility if needed
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
    print("Distribution des classes dans y :")
    print(y.value_counts())
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    # Pipeline avec standardisation + régression logistique multinomiale
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(multi_class='multinomial', max_iter=1000, solver='lbfgs', class_weight='balanced')
    )
    model.fit(X_train, y_train)

    # Prédiction des classes (signal binaire)
    y_pred_class = model.predict(X_test)

    # Prédiction des probabilités par classe (niveau de confiance)
    y_proba = model.predict_proba(X_test)  # matrice (n_samples, n_classes)

    # Évaluation
    r2 = r2_score(y_test, y_pred_class)
    mse = mean_squared_error(y_test, y_pred_class)
    print(f"R2 score : {r2:.4f}")
    print(f"Mean Squared Error : {mse:.4f}")

    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Vrai signal', marker='o')
    plt.plot(y_pred_class, label='Prédiction ML', linestyle='--', marker='x')
    plt.title("Régression : Signal prédit vs réel")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Affichage des 10 premiers exemples avec proba
    classes = model.named_steps['logisticregression'].classes_

    for i in range(10):
        proba_str = ", ".join([f"{cls}: {prob:.2f}" for cls, prob in zip(classes, y_proba[i])])
        print(f"Index {i} - Vrai signal: {y_test.iloc[i]}, Prédiction: {y_pred_class[i]}, Probabilités: [{proba_str}]")
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Accuracy : {accuracy:.4f}")

    print("Classification report :")
    print(classification_report(y_test, y_pred_class))

    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred_class))

    return "done"


if __name__ == "__main__":
    test_all_pairs()