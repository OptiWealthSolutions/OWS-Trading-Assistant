import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from ML_model_fx_strat_V3 import prepare_dataset_signal, get_all_data, engle_granger_test, calculate_adx
from ML_model_fx_strat_V3 import gestion_risque_adaptative
from ML_model_fx_strat_V3 import get_macro_data_fred
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from settings import tickers, commodity_mapping
import yfinance as yf  # add import if missing

st.set_page_config(layout="wide")
st.title("📈 Backtest ML Forex Strategy (Prévisionnel 5 ans)")


# Inputs
pair1 = st.selectbox("Choisissez la paire principale :", tickers)
capital = st.number_input("Capital initial (€)", min_value=100, max_value=10000, value=1000, step=100)

# Durée du backtest : sélection dynamique
durations = {
    "1 an": 365,
    "2 ans": 365 * 2,
    "3 ans": 365 * 3,
    "5 ans": 365 * 5,
}
selected_duration_label = st.selectbox("Durée du backtest :", list(durations.keys()))
selected_duration_days = durations[selected_duration_label]

if st.button("Lancer le backtest"):

    i = tickers.index(pair1)
    pair2 = tickers[i + 1] if i + 1 < len(tickers) else tickers[i - 1]
    base_currency1 = pair1[:3]
    commodity1 = commodity_mapping.get(base_currency1, None)

    df1, df2, df_commo1, _ = get_all_data(pair1, pair2, commodity1, None)

    if df1.empty or df2.empty:
        st.error("Erreur : données manquantes.")
        st.stop()

    # Filtrage par plage de dates selon la durée sélectionnée
    end_date = df1.index.max()
    start_date = end_date - pd.Timedelta(days=selected_duration_days)

    df1 = df1.loc[start_date:end_date]
    df2 = df2.loc[start_date:end_date]
    if commodity1 is not None and not df_commo1.empty:
        df_commo1 = df_commo1.loc[start_date:end_date]

    spread, _, _, _ = engle_granger_test(df1[f"{pair1}_Close"], df2[f"{pair2}_Close"])
    zscore = (spread - spread.mean()) / spread.std()
    pair1_close = df1[f"{pair1}_Close"]
    gold_price = df_commo1[f"{commodity1}_Close"] if commodity1 is not None and not df_commo1.empty else pair1_close
    adx = calculate_adx(pair1).reindex(spread.index).bfill()
    
    macro_data = get_macro_data_fred()
    
    X, y = prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx, macro_data=macro_data)

    smote = SMOTE()
    # Nettoyage des NaN et infinis
    mask = X.notnull().all(axis=1) & ~np.isinf(X).any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]

    # Affichage diagnostic dans Streamlit
    st.write("Distribution des classes après nettoyage :")
    st.write(y_clean.value_counts())

    if len(y_clean.unique()) < 2:
        st.error("Pas assez de classes différentes après nettoyage pour appliquer SMOTE.")
        st.stop()

    try:
        X_resampled, y_resampled = smote.fit_resample(X_clean, y_clean)
    except Exception as e:
        st.error(f"Erreur lors de SMOTE : {e}")
        st.stop()

    model = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', max_iter=1000))
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    proba_buy = y_proba[:, 2]
    proba_sell = y_proba[:, 0]

    # Pré-chargement des données volatiles pour gestion_risque_adaptative
    fx_std_data = yf.download(pair1, period="6mo", interval="4h", progress=False)
    fx_vol_data_brut = yf.download(pair1, period="6mo", interval="4h", progress=False)

    fx_df_std = pd.DataFrame(fx_std_data)
    fx_df_std['Log Returns'] = np.log(fx_df_std['Close'] / fx_df_std['Close'].shift(1))
    fx_df_std['STD'] = fx_df_std['Log Returns'].std()

    fx_df_vol = pd.DataFrame(fx_vol_data_brut)
    fx_df_vol['Log Returns'] = np.log(fx_df_vol['Close'] / fx_df_vol['Close'].shift(1))
    fx_df_vol['Volatility_20D'] = fx_df_vol['Log Returns'].rolling(window=20).std(ddof=0) * 100
    fx_df_vol.dropna(inplace=True)

    # Simuler capital
    df = df1.reindex(X.index).copy()
    df["Signal"] = y_pred
    df["Price"] = df[f"{pair1}_Close"]
    df["Position"] = 0
    df["Capital"] = capital
    df["Capital"] = df["Capital"].astype(float)  
    # Paramètres
    rr = 2  # Risk/Reward ratio
    pip_value = 0.0001  # Valeur pip pour la plupart des paires forex (sauf JPY)
    capital_current = capital
    position_open = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    position_direction = 0  # 1=buy, -1=sell

    df["Capital"] = capital
    df["Position"] = 0

    for i in range(1, len(df)):
        signal = int(df["Signal"].iloc[i])
        price = df["Price"].iloc[i]
        
        # Gestion du risque et taille position
        risk_info = gestion_risque_adaptative(capital_current, pair1, fx_df_std, fx_df_vol)
        risk_amount = risk_info['Risk €'].iloc[0]
        
        if signal == 2 and not position_open:
            # Ouverture position BUY
            position_open = True
            position_direction = 1
            entry_price = price
            stop_loss = entry_price - (risk_amount / capital_current) / pip_value * 0.0001  # stop loss en prix
            take_profit = entry_price + rr * (entry_price - stop_loss)
            df.at[df.index[i], "Position"] = 1

        elif signal == 0 and not position_open:
            # Ouverture position SELL
            position_open = True
            position_direction = -1
            entry_price = price
            stop_loss = entry_price + (risk_amount / capital_current) / pip_value * 0.0001  # stop loss en prix
            take_profit = entry_price - rr * (stop_loss - entry_price)
            df.at[df.index[i], "Position"] = -1

        if position_open:
            # Vérification si SL ou TP atteint
            if position_direction == 1:
                if price <= stop_loss:  # stop loss touché
                    loss = -risk_amount
                    capital_current += loss
                    position_open = False
                    df.at[df.index[i], "Capital"] = capital_current
                    df.at[df.index[i], "Position"] = 0
                elif price >= take_profit:  # take profit touché
                    gain = risk_amount * rr
                    capital_current += gain
                    position_open = False
                    df.at[df.index[i], "Capital"] = capital_current
                    df.at[df.index[i], "Position"] = 0
                else:
                    df.at[df.index[i], "Capital"] = capital_current
                    df.at[df.index[i], "Position"] = 1

            elif position_direction == -1:
                if price >= stop_loss:
                    loss = -risk_amount
                    capital_current += loss
                    position_open = False
                    df.at[df.index[i], "Capital"] = capital_current
                    df.at[df.index[i], "Position"] = 0
                elif price <= take_profit:
                    gain = risk_amount * rr
                    capital_current += gain
                    position_open = False
                    df.at[df.index[i], "Capital"] = capital_current
                    df.at[df.index[i], "Position"] = 0
                else:
                    df.at[df.index[i], "Capital"] = capital_current
                    df.at[df.index[i], "Position"] = -1
        else:
            df.at[df.index[i], "Capital"] = capital_current
            df.at[df.index[i], "Position"] = 0

    returns = df["Capital"].pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    pnl_eur = df["Capital"].iloc[-1] - capital
    pnl_pct = (df["Capital"].iloc[-1] / capital - 1) * 100

    # Affichage
    st.subheader("📉 Évolution du capital simulé")
    st.line_chart(df["Capital"])

    st.markdown("### 📈 Résultats du backtest")
    st.metric("PnL (€)", f"{pnl_eur:.2f} €")
    st.metric("PnL (%)", f"{pnl_pct:.2f} %")
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    st.subheader("📊 Détails des signaux")
    st.dataframe(df[["Price", "Signal", "Position", "Capital"]])