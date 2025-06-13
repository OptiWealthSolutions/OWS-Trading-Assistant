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

st.set_page_config(layout="wide")
st.title("📈 Backtest ML Forex Strategy (Prévisionnel 5 ans)")

# Inputs
pair1 = st.selectbox("Choisissez la paire principale :", tickers)
capital = st.number_input("Capital initial (€)", min_value=100, max_value=10000, value=1000, step=100)

if st.button("Lancer le backtest"):

    i = tickers.index(pair1)
    pair2 = tickers[i + 1] if i + 1 < len(tickers) else tickers[i - 1]
    base_currency1 = pair1[:3]
    commodity1 = commodity_mapping.get(base_currency1, None)

    df1, df2, df_commo1, _ = get_all_data(pair1, pair2, commodity1, None)

    if df1.empty or df2.empty:
        st.error("Erreur : données manquantes.")
        st.stop()

    spread, _, _, _ = engle_granger_test(df1[f"{pair1}_Close"], df2[f"{pair2}_Close"])
    zscore = (spread - spread.mean()) / spread.std()
    pair1_close = df1[f"{pair1}_Close"]
    gold_price = df_commo1[f"{commodity1}_Close"] if commodity1 is not None and not df_commo1.empty else pair1_close
    adx = calculate_adx(pair1).reindex(spread.index).bfill()
    
    macro_data = get_macro_data_fred()
    
    X, y = prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx, macro_data=macro_data)

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    model = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', max_iter=1000))
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    proba_buy = y_proba[:, 2]
    proba_sell = y_proba[:, 0]

    # Simuler capital
    df = df1.reindex(X.index).copy()
    df["Signal"] = y_pred
    df["Price"] = df[f"{pair1}_Close"]
    df["Position"] = 0
    df["Capital"] = capital
    rr = 2

    for i in range(1, len(df)):
        signal = int(df["Signal"].iloc[i])
        if signal == 2:  # BUY
            risk_info = gestion_risque_adaptative(df.iloc[i-1]["Capital"], pair1)
            risk = risk_info['Risk €'].iloc[0]
            reward = risk * rr
            df.iloc[i, df.columns.get_loc("Capital")] = df.iloc[i-1]["Capital"] + reward
            df.iloc[i, df.columns.get_loc("Position")] = 1
        elif signal == 0:  # SELL
            risk_info = gestion_risque_adaptative(df.iloc[i-1]["Capital"], pair1)
            risk = risk_info['Risk €'].iloc[0]
            reward = risk * rr
            df.iloc[i, df.columns.get_loc("Capital")] = df.iloc[i-1]["Capital"] + reward
            df.iloc[i, df.columns.get_loc("Position")] = -1
        else:
            df.iloc[i, df.columns.get_loc("Capital")] = df.iloc[i-1]["Capital"]
            df.iloc[i, df.columns.get_loc("Position")] = 0

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