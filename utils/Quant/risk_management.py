import yfinance as yf
import pandas as pd
import numpy as np

# --------------- Fonction % max ----------------
def gestion_risque_adaptative(capital, ticker, max_risk_pct=0.02, min_risk_pct=0.0):
    # Calcul std
    fx_std_data = yf.download(ticker, period="6mo", interval="4h")
    fx_df_std = pd.DataFrame(fx_std_data)
    fx_df_std['Log Returns'] = np.log(fx_df_std['Close'] / fx_df_std['Close'].shift(1))
    fx_df_std['STD'] = fx_df_std['Log Returns'].std()
    current_std = fx_df_std['STD'].iloc[-1]
    # Calcul vol
    fx_vol_data_brut = yf.download(ticker, period="6mo", interval="4h")
    fx_df_vol = pd.DataFrame(fx_vol_data_brut)
    fx_df_vol['Log Returns'] = np.log(fx_df_vol['Close'] / fx_df_vol['Close'].shift(1))
    fx_df_vol['Volatility_20D'] = fx_df_vol['Log Returns'].rolling(window=20).std(ddof=0) * 100
    fx_df_vol.dropna(inplace=True)
    current_vol = fx_df_vol['Volatility_20D'].iloc[-1]
    poids_vol = 0.5
    poids_var = 0.5
    score_risque = current_std * poids_var + current_vol * poids_vol 
    risque_pct = max(min_risk_pct, max_risk_pct * (1 - score_risque))
    risque_pct = float(round(risque_pct * 100, 2))
    risk_amount = round(capital * (risque_pct/100), 2)
    final_df = pd.DataFrame([{
        'Vol %': round(current_vol,4)*100,
        'Std %': round(current_std,4)*100,
        'Risk €': risk_amount,
        'Risk %': risque_pct
    }])
    return final_df


gestion_risque_adaptative(1000,"EURUSD=X")