import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import ta
from ta.momentum import RSIIndicator

# ==================== PAIR TRADING MODEL ====================

def data_loader(ticker_1, ticker_2, duration):
    data_1 = yf.download(ticker_1, period=duration, interval="1d")[["Close"]]
    data_2 = yf.download(ticker_2, period=duration, interval="1d")[["Close"]]
    data_1.rename(columns={"Close": f"{ticker_1}_Close"}, inplace=True)
    data_2.rename(columns={"Close": f"{ticker_2}_Close"}, inplace=True)
    df = pd.concat([data_1, data_2], axis=1).dropna()
    return df

def test_adf(series: pd.Series):
    result = adfuller(series)
    return result[1]

def engle_granger_test(y_serie: pd.Series, x_serie: pd.Series):
    df_temp = pd.concat([y_serie, x_serie], axis=1).dropna()
    y_serie_clean = df_temp.iloc[:, 0]
    x_serie_clean = df_temp.iloc[:, 1]
    x = sm.add_constant(x_serie_clean)
    model = sm.OLS(y_serie_clean, x).fit()
    residuals = model.resid
    p_value = test_adf(residuals)
    r_squared = model.rsquared
    params = model.params
    return residuals, p_value, r_squared, params

def pairs_trading_summary(ticker1: str, ticker2: str, duration: str = "1y", save_path=None):
    df = data_loader(ticker1, ticker2, duration)
    spread, p_value, r_squared, params = engle_granger_test(df[f"{ticker1}_Close"], df[f"{ticker2}_Close"])
    cointegration = p_value < 0.05
    interpretation = f"Analyse de la paire {ticker1} / {ticker2} sur {duration}:\n"
    interpretation += f"- Co-intégration détectée : {'Oui' if cointegration else 'Non'} (p-value={p_value:.4f})\n"
    interpretation += f"- R² de la régression : {r_squared:.4f}\n"
    interpretation += f"- Coefficients : alpha (intercept) = {params[0]:.4f}, beta = {params[1]:.4f}\n"
    #créer une matrice avec 10 ans de z_score pour regression 
    zscore = (spread - spread.mean()) / spread.std()
    zscore_final = zscore.iloc[-1]
    interpretation += f"- Niveau actuel du z-score : {zscore_final:.2f} (distance à la moyenne du spread)\n"
    
    if abs(zscore_final) > 2:
        interpretation += "- Le spread est significativement éloigné de sa moyenne (potentiel signal de trading).\n"
    else:
        interpretation += "- Le spread est proche de sa moyenne (pas de signal fort).\n"
    return zscore


# ==================== RISK MANAGEMENT ====================

def atr_index(ticker):
    """
    Calcule l'Average True Range (ATR) et un coefficient k basé sur la distribution du ratio distance/ATR.
    """
    df = yf.download(ticker, period="6mo", interval='4H')
    
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df.dropna(inplace=True)  # Supprime les NaN après calcul ATR
    # Calcul de k
    distance = abs(df['Close'].shift(1) - df['Low'])
    ratio = (distance / df['ATR'].iloc[-1]).dropna()

    k_95 = np.percentile(ratio, 95)
    atr_res = df['ATR'].iloc[-1]
    return atr_res, round(k_95,1)

def sl_sizing(atr, k, entry_price, direction):
    """
    Calcule le niveau de stop loss en fonction de l'ATR, d'un coefficient k, du prix d'entrée et de la direction du trade.
    Retourne le prix du stop et la taille du stop en pips.
    """
    if direction == "SELL":
        stop_price = entry_price + atr * k
        sl_pips_size = round(abs(entry_price - stop_price) * 10000, 3)
        return round(stop_price, 6), sl_pips_size
    elif direction == "BUY":
        stop_price = entry_price - atr * k
        sl_pips_size = round(abs(entry_price - stop_price) * 10000, 3)
        return round(stop_price, 6), sl_pips_size
    else:
        raise ValueError("INVALID DIRECTION")
    
# ==================== INDICATEURS TECHNIQUES ====================

def calculate_adx(ticker, period=14):
    """
    Calcule l'Average Directional Index (ADX) pour évaluer la force de la tendance.
    Retourne la dernière valeur d'ADX et une interprétation textuelle.
    """
    df = yf.download(ticker, period='5d', interval='1h')

    df['High'] = df['High'].shift(1)
    df['Low'] = df['Low'].shift(1)
    df['Close'] = df['Close'].shift(1)

    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close']), abs(df['Low'] - df['Close'])))
    df['+DM'] = df['High'] - df['High'].shift(1)
    df['-DM'] = df['Low'].shift(1) - df['Low']

    df['+DM'] = np.where(df['+DM'] > 0, df['+DM'], 0)
    df['-DM'] = np.where(df['-DM'] > 0, df['-DM'], 0)

    df['+DM_smooth'] = df['+DM'].rolling(window=period).sum()
    df['-DM_smooth'] = df['-DM'].rolling(window=period).sum()
    df['TR_smooth'] = df['TR'].rolling(window=period).sum()

    df['+DI'] = (df['+DM_smooth'] / df['TR_smooth']) * 100
    df['-DI'] = (df['-DM_smooth'] / df['TR_smooth']) * 100
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].rolling(window=period).mean()

    last_adx = df['ADX'].iloc[-1]

    if last_adx < 20:
        interpretation = "Tendance faible ou inexistante"
    elif last_adx < 40:
        interpretation = "Tendance modérée"
    else:
        interpretation = "Tendance forte"

    return last_adx, interpretation


# ==================== COMMODITÉS & ML ====================

eurusd = yf.download("EURUSD=X", period="15y")['Close']
gold = yf.download("GC=F", period="15y")['Close']
oil = yf.download("CL=F", period="15y")['Close']

eurusd = pd.DataFrame(eurusd)
eurusd.columns = ['eurusd']

gold_return = gold.pct_change()
oil_return = oil.pct_change()

df = pd.concat([eurusd, gold_return, oil_return], axis=1)
df.columns = ['eurusd', 'gold_return', 'oil_return']

df['log_return'] = np.log(df['eurusd'] / df['eurusd'].shift(1))

df_model = df[['gold_return', 'oil_return', 'log_return']].dropna().copy()

df_model['gold_return_lag1'] = df_model['gold_return'].shift(1)
df_model['oil_return_lag1'] = df_model['oil_return'].shift(1)
df_model['gold_return_sq'] = df_model['gold_return'] ** 2
df_model['oil_return_sq'] = df_model['oil_return'] ** 2
df_model['interaction'] = df_model['gold_return'] * df_model['oil_return']
df_model = df_model.dropna()

df_model['target'] = (df_model['log_return'] > 0).astype(int)

X = df_model[['gold_return', 'oil_return', 'gold_return_lag1', 'oil_return_lag1','gold_return_sq', 'oil_return_sq', 'interaction']].values
y = df_model[['target']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = np.hstack((X_scaled, np.ones((X.shape[0], 1))))

theta = np.zeros((X.shape[1], 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model_logistic(X, theta):
    return sigmoid(X.dot(theta))

def cost_function_logistic(X, y, theta):
    m = len(y)
    h = model_logistic(X, theta)
    epsilon = 1e-5
    cost = -(1/m) * (y.T.dot(np.log(h + epsilon)) + (1 - y).T.dot(np.log(1 - h + epsilon)))
    return cost.flatten()[0]

def grad_logistic(X, y, theta):
    m = len(y)
    h = model_logistic(X, theta)
    return (1/m) * X.T.dot(h - y)

def gradient_descent_logistic(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        theta = theta - learning_rate * grad_logistic(X, y, theta)
        cost_history[i] = cost_function_logistic(X, y, theta)
    return theta, cost_history

learning_rate = 1
n_iterations = 1000

theta_final, cost_history = gradient_descent_logistic(X, y, theta, learning_rate, n_iterations)

pred_prob = model_logistic(X, theta_final)
pred_class = (pred_prob >= 0.5).astype(int)

accuracy = (pred_class == y).mean()
print(f"Accuracy du modèle : {accuracy:.4f}")





def prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx_series, seuil=1):
    """
    Construit les features pour un modèle ML et génère un signal multi-classe :
    -1 = SELL (écart positif extrême)
     0 = NEUTRAL
     1 = BUY (écart négatif extrême)
    """
    # S'assurer que inputs sont des Series 1D
    if isinstance(pair1_close, pd.DataFrame):
        pair1_close = pair1_close.squeeze()
    if isinstance(gold_price, pd.DataFrame):
        gold_price = gold_price.squeeze()
    if isinstance(adx_series, pd.DataFrame):
        adx_series = adx_series.squeeze()

    df = pd.DataFrame({
        'spread': spread,
        'z_score': zscore,
        'z_score_lag1': zscore.shift(1),
        'vol_spread': spread.rolling(30).std(),
        'rsi_pair1': RSIIndicator(close=pair1_close, window=14).rsi(),
        'corr_gold': pair1_close.rolling(30).corr(gold_price),
        'adx': adx_series
    })

    df.dropna(inplace=True)

    df['target'] = 0
    df.loc[df['z_score'] > seuil, 'target'] = -1
    df.loc[df['z_score'] < -seuil, 'target'] = 1

    X = df[['z_score', 'z_score_lag1', 'vol_spread', 'rsi_pair1', 'corr_gold', 'adx']]
    y = df['target']

    return X, y