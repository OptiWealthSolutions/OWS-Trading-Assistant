# ML_model_fx_strat.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator

from fx_strategy_V2 import *
import settings


def prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx, seuil=1):
    """
    Prepare the feature matrix X and target vector y for the ML model based on input series.

    Parameters:
        spread (pd.Series): The spread between two currency pairs.
        zscore (pd.Series): Z-score normalized spread.
        pair1_close (pd.Series): Closing prices for the first currency pair.
        gold_price (pd.Series): Closing prices for the commodity (e.g., gold) related to the base currency.
        adx (pd.Series): Average Directional Index indicator values.
        seuil (float): Threshold for defining target signals based on z-score.

    Returns:
        X (pd.DataFrame): Feature matrix with indicators.
        y (pd.Series): Target vector with signals (-1: SELL, 0: WAIT, 1: BUY).
    """
    # Ensure pair1_close and gold_price are Series, not DataFrames
    if isinstance(pair1_close, pd.DataFrame):
        pair1_close = pair1_close.iloc[:, 0]
    if isinstance(gold_price, pd.DataFrame):
        gold_price = gold_price.iloc[:, 0]

    # Align indices of all series to the spread index
    pair1_close = pair1_close.reindex(spread.index)
    gold_price = gold_price.reindex(spread.index)
    zscore = zscore.reindex(spread.index)
    adx = adx.reindex(spread.index)

    # Compute RSI indicator for pair1 closing prices
    rsi_pair1 = RSIIndicator(close=pair1_close, window=14).rsi()

    # Construct DataFrame with features:
    # - spread: raw spread value
    # - z_score: current z-score of spread
    # - z_score_lag1: z-score lagged by 1 period (previous day)
    # - vol_spread: rolling standard deviation of spread over 30 periods (volatility proxy)
    # - rsi_pair1: RSI indicator of pair1
    # - adx: directional movement indicator
    df = pd.DataFrame({
        'spread': spread,
        'z_score': zscore,
        'z_score_lag1': zscore.shift(1),
        'vol_spread': spread.rolling(30).std(),
        'rsi_pair1': rsi_pair1,
        'adx': adx
    })

    # Drop rows with missing values and ensure float type
    df.dropna(inplace=True)
    df = df.astype(float)

    # Create target variable based on z-score threshold:
    # -1 indicates SELL signal, 1 indicates BUY signal, 0 means WAIT/neutral
    df['target'] = 0
    df.loc[df['z_score'] > seuil, 'target'] = -1
    df.loc[df['z_score'] < -seuil, 'target'] = 1

    # Define features and target for model training
    X = df[['z_score', 'z_score_lag1', 'vol_spread', 'rsi_pair1', 'adx']]
    y = df['target']

    return X, y


def run_model_for_pair(pair1, pair2):
    """
    Runs the ML model for a given pair of currency pairs and returns the predicted trading signal.

    Steps:
    - Load and prepare data for the two pairs and associated commodities.
    - Calculate spread and z-score.
    - Prepare features and target variable.
    - Train logistic regression model on all available data except last day.
    - Predict signal and confidence for the last day.

    Parameters:
        pair1 (str): First currency pair ticker.
        pair2 (str): Second currency pair ticker.

    Returns:
        dict or None: Dictionary with prediction details or None if data insufficient.
    """
    # Extract base currency from pair1 and get associated commodity
    base_currency1 = pair1[:3]  # e.g., 'EUR' from 'EURUSD=X'
    commodity1 = settings.commodity_mapping.get(base_currency1, None)
    commodity2 = None  # Placeholder, not used currently

    # Retrieve historical data for pairs and commodities
    df1, df2, df_commo1, df_commo2 = get_all_data(pair1, pair2, commodity1, commodity2)

    if df1.empty or df2.empty:
        print(f"Data not available for pair {pair1} or {pair2}. Skipping.")
        return None

    # Calculate spread and perform Engle-Granger cointegration test
    spread, _, _, _ = engle_granger_test(df1[f"{pair1}_Close"], df2[f"{pair2}_Close"])
    zscore = (spread - spread.mean()) / spread.std()

    # Use commodity price if available, else fallback to pair1 closing price
    pair1_close = df1[f"{pair1}_Close"]
    gold_price = df_commo1[f"{commodity1}_Close"] if (commodity1 and not df_commo1.empty) else pair1_close

    # Calculate ADX indicator for pair1 and align with spread index
    adx_series = calculate_adx(pair1)
    adx = adx_series.reindex(spread.index).fillna(method="bfill")

    # Prepare features and target for modeling
    X, y = prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx)

    if X.empty or y.empty:
        print(f"Insufficient data after preparation for pair {pair1} and {pair2}. Skipping.")
        return None

    # Require at least two data points to train and predict
    if len(X) < 2:
        print(f"Not enough data points for pair {pair1} and {pair2}. Skipping.")
        return None

    # Use all data except last day for training, last day for prediction
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    X_pred = X.iloc[-1:]

    # Define logistic regression pipeline with standard scaling and balanced classes
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(multi_class='multinomial', max_iter=10000, solver='lbfgs', class_weight='balanced')
    )
    model.fit(X_train, y_train)

    # Predict class and probabilities for last day
    y_pred_class = model.predict(X_pred)[0]
    y_proba = model.predict_proba(X_pred)[0]

    # Extract confidence of predicted class
    classes = model.named_steps['logisticregression'].classes_
    class_index = np.where(classes == y_pred_class)[0][0]
    confidence = y_proba[class_index]

    # Map predicted class to trading signal
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


def save_results_to_pdf(df_results, filename="ml_signals_report.pdf"):
    """
    Save the ML model prediction results to a PDF file as a styled table.

    The table highlights rows based on the trading signal:
    - SELL signals in light red with dark red text
    - BUY signals in light green with dark green text
    - WAIT signals with white background and black text

    Parameters:
        df_results (pd.DataFrame): DataFrame containing predictions and signals.
        filename (str): Output PDF filename.
    """
    fig, ax = plt.subplots(figsize=(12, len(df_results)*0.5 + 1))
    ax.axis('off')  # Hide axes for cleaner table display

    # Define row background colors based on signal type
    row_colors = []
    for sig in df_results['signal']:
        if sig == 'SELL':
            row_colors.append("#ff4d4d89")  # Light red background
        elif sig == 'BUY':
            row_colors.append("#4CAF4F76")  # Light green background
        else:
            row_colors.append('white')      # White background for WAIT

    # Create matplotlib table with color-coded rows
    table = ax.table(cellText=df_results.values,
                     colLabels=df_results.columns,
                     cellColours=[[color]*len(df_results.columns) for color in row_colors],
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Adjust text color for better readability on colored backgrounds
    for i, sig in enumerate(df_results['signal']):
        for j in range(len(df_results.columns)):
            cell = table[i+1, j]  # +1 because row 0 is header
            if sig == 'SELL':
                cell.get_text().set_color('darkred')
            elif sig == 'BUY':
                cell.get_text().set_color('darkgreen')
            else:
                cell.get_text().set_color('black')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Report saved as {filename}")
    plt.close()


def test_all_pairs():
    """
    Test the ML model on all consecutive pairs from the tickers list.

    For each consecutive pair of currency pairs, runs the model and collects results.
    Saves the aggregated results to a PDF report file.

    Returns:
        pd.DataFrame: DataFrame containing prediction results for all pairs.
    """
    results = []
    tickers = settings.tickers

    # Iterate over consecutive pairs of tickers (e.g., ticker[i], ticker[i+1])
    for i in range(len(tickers) - 1):
        pair1 = tickers[i]
        pair2 = tickers[i + 1]
        result = run_model_for_pair(pair1, pair2)
        if result is not None:
            results.append(result)

    df_results = pd.DataFrame(results)
    print("ML Model Prediction Results for all pairs:")
    print(df_results)

    # Save results table to PDF report
    save_results_to_pdf(df_results)

    return df_results

if __name__ == "__main__":
    test_all_pairs()