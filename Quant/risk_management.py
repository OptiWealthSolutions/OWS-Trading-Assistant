import yfinance as yf
import pandas as pd
import numpy as np

# --------------- Fonction % max ----------------
def gestion_risque_adaptative(capital, vol, var, max_risk_pct=0.02, min_risk_pct=0.0):
    poids_vol = 0.5
    poids_var = 0.5
    score_risque = vol * poids_vol + var * poids_var 
    risque_pct = max(min_risk_pct, max_risk_pct * (1 - score_risque))
    risque_pct = float(round(risque_pct * 100, 2))
    risk_amount = round(capital * (risque_pct/100), 2)
    return risque_pct,risk_amount

# --------------- Fonction indice de vol ----------------
def vol_index (ticker):
    fx_vol_data_brut = yf.download(ticker, period="6mo", interval="4h")
    fx_df_vol = pd.DataFrame(fx_vol_data_brut)
    fx_df_vol['Log Returns'] = np.log(fx_df_vol['Close'] / fx_df_vol['Close'].shift(1))
    fx_df_vol['Volatility_20D'] = fx_df_vol['Log Returns'].rolling(window=20).std(ddof=0) * 100
    fx_df_vol.dropna(inplace=True)
    current_vol = fx_df_vol['Volatility_20D'].iloc[-1]
    print(round(current_vol,4)*100 ,"%")
    return round(current_vol,4)

# --------------- Fonction indice de std ----------------
def std_index (ticker):
    fx_std_data = yf.download(ticker, period="6mo", interval="4h")
    fx_df_std = pd.DataFrame(fx_std_data)
    fx_df_std['Log Returns'] = np.log(fx_df_std['Close'] / fx_df_std['Close'].shift(1))
    fx_df_std['STD'] = fx_df_std['Log Returns'].std()
    current_std = fx_df_std['STD'].iloc[-1]
    print(round(current_std,4)*100 ,"%")
    return round(current_std,4)

capital = 1000
ticker = "EURJPY=X"
std = std_index(ticker)
vol = vol_index(ticker)

print(gestion_risque_adaptative(capital,std,vol))
