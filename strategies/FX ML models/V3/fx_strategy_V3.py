import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import ta
from ta.momentum import RSIIndicator
from fredapi import Fred
import warnings

def get_all_data(pair_1,pair_2,commodity_1=None,commodity_2=None,period = "5y"):
    data_1 = yf.download(pair_1, period=period, interval="1d", progress=False, auto_adjust=False)
    data_2 = yf.download(pair_2, period=period, interval="1d", progress=False, auto_adjust=False)
    data_1.rename(columns={"Close": f"{pair_1}_Close"}, inplace=True)
    data_2.rename(columns={"Close": f"{pair_2}_Close"}, inplace=True)
    df = pd.concat([data_1, data_2], axis=1).dropna()
    
    if commodity_1 is not None:
        data_commodity1 = yf.download(commodity_1, period=period, progress=False, auto_adjust=False)
        data_commodity1.rename(columns={"Close": f"{commodity_1}_Close"}, inplace=True)
    if commodity_2 is not None:
        data_commodity2 = yf.download(commodity_2, period=period, progress=False, auto_adjust=False)
        data_commodity2.rename(columns={"Close": f"{commodity_2}_Close"},inplace=True)

    data_commodity1 = data_commodity1 if commodity_1 is not None else pd.DataFrame()
    data_commodity2 = data_commodity2 if commodity_2 is not None else pd.DataFrame()
    
    return data_1, data_2, data_commodity1, data_commodity2

# ==================== PAIR TRADING MODEL ====================
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

def pairs_trading_summary(ticker1,ticker2,data_frame_pair_1, data_frame_pair_2,duration: str = "1y"):
    df = pd.concat([data_frame_pair_1, data_frame_pair_2], axis=1).dropna()
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
 #probleme il faut retourner une data frame avec tout les z_score de la meme taille que les autres dataframe pour pouvoir faire une regression polynomiale pour trouver la target   
    return zscore

# ==================== INDICATEURS TECHNIQUES ====================

def calculate_adx(ticker, period=14):
    df, _, _, _ = get_all_data(ticker, ticker)
    df.rename(columns={f"{ticker}_Close": "Close"}, inplace=True)

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
    
    return df['ADX']

# ==================== CORRELATION DETERMINATION ====================

currency_commodity_map = {
    "AUD": ["GC=F", "HG=F"],  # Gold, Copper
    "CAD": ["CL=F", "NG=F"],  # Oil, Gas
    "NZD": ["ZC=F", "ZW=F"],  # Agricultural products
    "NOK": ["BZ=F"],          # Brent Oil
    "USD": ["GC=F", "CL=F"],  # Indirectly correlated with many
    "BRL": ["SB=F", "KC=F"],  # Sugar, Coffee
    "MXN": ["CL=F", "ZS=F"],  # Oil, Soybean
    "ZAR": ["GC=F", "PL=F"],  # Gold, Platinum
}

def get_macro_data_fred(start='2000-01-01', end=None):
    from dotenv import load_dotenv
    import os
    load_dotenv()
    key = os.getenv("FRED_API_KEY")
    if not key:
        print("FRED API key is missing. Returning empty macro DataFrame.")
        return pd.DataFrame()
    fred = Fred(api_key=key)
    cpi = fred.get_series('CPIAUCSL', observation_start=start, observation_end=end)
    unemployment = fred.get_series('UNRATE', observation_start=start, observation_end=end)
    gdp = fred.get_series('GDP', observation_start=start, observation_end=end)

    df_macro = pd.concat([cpi, unemployment, gdp], axis=1)
    df_macro.columns = ['CPI', 'Unemployment', 'GDP']
    df_macro.index = pd.to_datetime(df_macro.index)
    return df_macro

def get_commo_corr(ticker, period="10y", interval="1d"):
    base1 = ticker[:3].upper()
    base2 = ticker[3:6].upper()
    commodities = list(set(currency_commodity_map.get(base1, []) + currency_commodity_map.get(base2, [])))

    if not commodities:
        print(f"Aucune commodity trouvée pour {base1} ou {base2}")
        return pd.DataFrame()

    # Données forex (plus robuste et explicite)
    forex = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if forex.empty or 'Close' not in forex.columns:
        raise ValueError(f"Impossible de récupérer les données de clôture pour le ticker {ticker}")

    forex = forex[['Close']].copy()
    forex.columns = ['forex']
    forex.index = forex.index.tz_localize(None)
    data = forex.copy()

    # Données commodities
    for commo in commodities:
        commo_data = yf.download(commo, period=period, interval=interval, progress=False, auto_adjust=False)
        if 'Close' not in commo_data.columns:
            print(f"Pas de 'Close' pour {commo}")
            continue
        commo_close = commo_data['Close'].copy()
        commo_close.name = commo
        commo_close.index = commo_close.index.tz_localize(None)
        data = data.join(commo_close, how="inner")

    if data.empty or data.shape[1] < 2:
        print("Pas assez de données croisées pour corrélation.")
        return pd.DataFrame()

    # Corrélation glissante
    window = 30
    correlations = pd.DataFrame()
    for commo in commodities:
        if commo not in data.columns:
            continue
        forex_returns = data['forex'].pct_change()
        commo_returns = data[commo].pct_change()
        correlations[commo] = forex_returns.rolling(window).corr(commo_returns)

    return correlations

# ==================== MAKE DATASET FOR ML ====================

def get_interest_rate_difference(pair: str) -> float:
    """
    Retourne une estimation simplifiée de l'écart de taux d'intérêt entre les deux devises de la paire.
    À adapter avec des sources dynamiques (ex: FRED, BCE API).
    """
    # Dictionnaire statique (à automatiser plus tard)
    rates = {
        "USD": 4.50,
        "EUR": 2.15,
        "GBP": 4.25,
        "JPY": 0.50,
        "CHF": 0.25,
        "CAD": 2.75,
        "AUD": 3.85,
        "NZD": 3.25,
    }

    base = pair[:3].upper()
    quote = pair[3:6].upper()

    if base in rates and quote in rates:
        return rates[base] - rates[quote]
    else:
        return 0.0

def prepare_dataset_signal(spread, zscore, pair1_close, commo_price_1, adx_series, pair1, seuil=1):
    """
    Construit les features pour un modèle ML et génère un signal multi-classe :
    -1 = SELL (écart positif extrême)
     0 = NEUTRAL
     1 = BUY (écart négatif extrême)
    """
    # S'assurer que inputs sont des Series 1D
    if isinstance(pair1_close, pd.DataFrame):
        pair1_close = pair1_close.squeeze()
    if isinstance(commo_price_1, pd.DataFrame):
        commo_price_1 = commo_price_1.squeeze()
    if isinstance(adx_series, pd.DataFrame):
        adx_series = adx_series.squeeze()

    rate_diff = get_interest_rate_difference(pair1)

    df = pd.DataFrame({
        'spread': spread,
        'z_score': zscore,
        'z_score_lag1': zscore.shift(1),
        'vol_spread': spread.rolling(30).std(),
        'rsi_pair1': RSIIndicator(close=pair1_close, window=14).rsi(),
        'rolling_corr_commo': pair1_close.rolling(30).corr(commo_price_1),
        'adx': adx_series,
        'rate_diff': rate_diff
    })

    df.dropna(inplace=True)

    df['target'] = 0
    df.loc[df['z_score'] > seuil, 'target'] = -1
    df.loc[df['z_score'] < -seuil, 'target'] = 1

    X = df[['z_score', 'z_score_lag1', 'vol_spread', 'rsi_pair1', 'rolling_corr_commo', 'adx', 'rate_diff']]
    y = df['target']

    return X, y

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")