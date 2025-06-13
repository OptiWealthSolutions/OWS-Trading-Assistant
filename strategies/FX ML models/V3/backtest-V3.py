import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from settings import tickers, commodity_mapping
from ML_model_fx_strat_V3 import prepare_dataset_signal, get_all_data, engle_granger_test, calculate_adx, gestion_risque_adaptative, get_macro_data_fred

st.set_page_config(layout="wide")
st.title("📈 Backtest ML Forex Strategy (Prévisionnel multi-paires & multi-durations)")

def compute_atr(ticker):
    df = yf.download(ticker, period="6mo", interval='4h', progress=False)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df.dropna(inplace=True)
    return df['ATR'].iloc[-1]

# UI inputs
selected_pairs = st.multiselect("Choisissez une ou plusieurs paires :", tickers, default=[tickers[0]])
durations = {"1 an": 365, "2 ans": 365*2, "3 ans": 365*3, "5 ans": 365*5}
selected_durations = st.multiselect("Choisissez durée(s) du backtest :", list(durations.keys()), default=["1 an"])
capital = st.number_input("Capital initial (€)", min_value=100, max_value=10000, value=1000, step=100)
threshold = 0.6  # confidence threshold for signals

if st.button("Lancer le backtest"):

    for pair1 in selected_pairs:
        i = tickers.index(pair1)
        pair2 = tickers[i+1] if i+1 < len(tickers) else tickers[i-1]
        base_currency1 = pair1[:3]
        commodity1 = commodity_mapping.get(base_currency1, None)

        df1, df2, df_commo1, _ = get_all_data(pair1, pair2, commodity1, None)
        if df1.empty or df2.empty:
            st.error(f"Données manquantes pour {pair1}")
            continue

        for dur_label in selected_durations:
            days = durations[dur_label]
            end_date = df1.index.max()
            start_date = end_date - pd.Timedelta(days=days)

            df1_filt = df1.loc[start_date:end_date]
            df2_filt = df2.loc[start_date:end_date]
            df_commo1_filt = df_commo1.loc[start_date:end_date] if commodity1 and not df_commo1.empty else pd.DataFrame()

            spread, _, _, _ = engle_granger_test(df1_filt[f"{pair1}_Close"], df2_filt[f"{pair2}_Close"])
            zscore = (spread - spread.mean()) / spread.std()
            pair1_close = df1_filt[f"{pair1}_Close"]
            gold_price = df_commo1_filt[f"{commodity1}_Close"] if commodity1 and not df_commo1_filt.empty else pair1_close
            adx = calculate_adx(pair1)
            adx = adx.loc[adx.index.isin(spread.index)].reindex(spread.index).bfill()
            macro_data = get_macro_data_fred()

            X, y = prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx, macro_data=macro_data)
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            y = y.loc[X.index]  # réaligner y

            mask = X.notnull().all(axis=1) & ~np.isinf(X).any(axis=1)
            X_clean, y_clean = X[mask], y[mask]

            st.write(f"Distribution des classes pour {pair1} - {dur_label} :", y_clean.value_counts())
            if len(y_clean.unique()) < 2:
                st.warning(f"Pas assez de classes différentes pour {pair1} sur {dur_label}")
                continue

            smote = SMOTE()
            try:
                X_res, y_res = smote.fit_resample(X_clean, y_clean)
            except Exception as e:
                st.error(f"SMOTE erreur sur {pair1} {dur_label} : {e}")
                continue

            model = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', max_iter=1000))
            model.fit(X_res, y_res)
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)

            proba_buy = y_proba[:, 2]
            proba_sell = y_proba[:, 0]

            df = df1_filt.reindex(X.index).copy()
            df["Signal"] = 1  # default WAIT
            for idx in range(len(df)):
                if proba_buy[idx] >= threshold:
                    df.at[df.index[idx], "Signal"] = 2
                elif proba_sell[idx] >= threshold:
                    df.at[df.index[idx], "Signal"] = 0

            df["Price"] = df[f"{pair1}_Close"]
            df["Position"] = 0
            df["Capital"] = capital
            df["Capital"] = df["Capital"].astype(float)

            rr = 4
            pip_value = 0.0001
            commission = 2.5
            spread_pips = 1.5
            slippage_pips = 0.5

            capital_current = capital
            position_open = False
            entry_price = 0
            stop_loss = 0
            take_profit = 0
            position_direction = 0
            entry_time = None

            atr = compute_atr(pair1)

            trade_log = []
            trade_open = False
            trade_entry_price = 0
            trade_entry_time = None
            trade_direction = 0
            trade_risk = 0

            for i in range(1, len(df)):
                signal = int(df["Signal"].iloc[i])
                price = df["Price"].iloc[i]

                risk_info = gestion_risque_adaptative(capital_current, pair1)
                risk_amount = risk_info['Risk €'].iloc[0]

                if signal == 2 and not position_open:
                    position_open = True
                    position_direction = 1
                    entry_price = price + (spread_pips + slippage_pips) * pip_value
                    entry_time = df.index[i]
                    stop_loss = entry_price - atr * 1.5
                    take_profit = entry_price + rr * (entry_price - stop_loss)
                    df.at[df.index[i], "Position"] = 1
                    capital_current -= commission
                    trade_open = True
                    trade_entry_price = entry_price
                    trade_entry_time = entry_time
                    trade_direction = 1
                    trade_risk = risk_amount

                elif signal == 0 and not position_open:
                    position_open = True
                    position_direction = -1
                    entry_price = price - (spread_pips + slippage_pips) * pip_value
                    entry_time = df.index[i]
                    stop_loss = entry_price + atr * 1.5
                    take_profit = entry_price - rr * (stop_loss - entry_price)
                    df.at[df.index[i], "Position"] = -1
                    capital_current -= commission
                    trade_open = True
                    trade_entry_price = entry_price
                    trade_entry_time = entry_time
                    trade_direction = -1
                    trade_risk = risk_amount

                if position_open:
                    if position_direction == 1:
                        if price <= stop_loss:
                            capital_current -= commission
                            loss = -risk_amount
                            capital_current += loss
                            position_open = False
                            entry_time = None
                            df.at[df.index[i], "Capital"] = capital_current
                            df.at[df.index[i], "Position"] = 0
                            trade_log.append({
                                'Entry Time': trade_entry_time,
                                'Exit Time': df.index[i],
                                'Direction': 'BUY',
                                'Entry Price': trade_entry_price,
                                'Exit Price': price,
                                'Result (€)': loss - commission,
                                'Duration (h)': (df.index[i] - trade_entry_time).total_seconds() / 3600,
                                'Risk (€)': trade_risk,
                                'Profit Factor': loss / trade_risk if trade_risk != 0 else np.nan
                            })
                            trade_open = False
                        elif price >= take_profit:
                            capital_current -= commission
                            gain = risk_amount * rr
                            capital_current += gain
                            position_open = False
                            entry_time = None
                            df.at[df.index[i], "Capital"] = capital_current
                            df.at[df.index[i], "Position"] = 0
                            trade_log.append({
                                'Entry Time': trade_entry_time,
                                'Exit Time': df.index[i],
                                'Direction': 'BUY',
                                'Entry Price': trade_entry_price,
                                'Exit Price': price,
                                'Result (€)': gain - commission,
                                'Duration (h)': (df.index[i] - trade_entry_time).total_seconds() / 3600,
                                'Risk (€)': trade_risk,
                                'Profit Factor': gain / trade_risk if trade_risk != 0 else np.nan
                            })
                            trade_open = False
                        else:
                            df.at[df.index[i], "Capital"] = capital_current
                            df.at[df.index[i], "Position"] = 1

                    elif position_direction == -1:
                        if price >= stop_loss:
                            capital_current -= commission
                            loss = -risk_amount
                            capital_current += loss
                            position_open = False
                            entry_time = None
                            df.at[df.index[i], "Capital"] = capital_current
                            df.at[df.index[i], "Position"] = 0
                            trade_log.append({
                                'Entry Time': trade_entry_time,
                                'Exit Time': df.index[i],
                                'Direction': 'SELL',
                                'Entry Price': trade_entry_price,
                                'Exit Price': price,
                                'Result (€)': loss - commission,
                                'Duration (h)': (df.index[i] - trade_entry_time).total_seconds() / 3600,
                                'Risk (€)': trade_risk,
                                'Profit Factor': loss / trade_risk if trade_risk != 0 else np.nan
                            })
                            trade_open = False
                        elif price <= take_profit:
                            capital_current -= commission
                            gain = risk_amount * rr
                            capital_current += gain
                            position_open = False
                            entry_time = None
                            df.at[df.index[i], "Capital"] = capital_current
                            df.at[df.index[i], "Position"] = 0
                            trade_log.append({
                                'Entry Time': trade_entry_time,
                                'Exit Time': df.index[i],
                                'Direction': 'SELL',
                                'Entry Price': trade_entry_price,
                                'Exit Price': price,
                                'Result (€)': gain - commission,
                                'Duration (h)': (df.index[i] - trade_entry_time).total_seconds() / 3600,
                                'Risk (€)': trade_risk,
                                'Profit Factor': gain / trade_risk if trade_risk != 0 else np.nan
                            })
                            trade_open = False
                        else:
                            df.at[df.index[i], "Capital"] = capital_current
                            df.at[df.index[i], "Position"] = -1

                    # Close trade after 2 days if no TP or SL hit
                    if trade_open and (df.index[i] - trade_entry_time) > pd.Timedelta(days=2):
                        pnl = (price - trade_entry_price) if trade_direction == 1 else (trade_entry_price - price)
                        pnl *= (risk_amount / (abs(entry_price - stop_loss) if entry_price != stop_loss else 1))
                        capital_current -= commission
                        capital_current += pnl
                        position_open = False
                        entry_time = None
                        trade_log.append({
                            'Entry Time': trade_entry_time,
                            'Exit Time': df.index[i],
                            'Direction': 'BUY' if trade_direction == 1 else 'SELL',
                            'Entry Price': trade_entry_price,
                            'Exit Price': price,
                            'Result (€)': pnl - commission,
                            'Duration (h)': (df.index[i] - trade_entry_time).total_seconds() / 3600,
                            'Risk (€)': trade_risk,
                            'Profit Factor': pnl / trade_risk if trade_risk != 0 else np.nan,
                            'Reason': 'Time Exit'
                        })
                        trade_open = False
                        df.at[df.index[i], "Capital"] = capital_current
                        df.at[df.index[i], "Position"] = 0
                        continue
                else:
                    df.at[df.index[i], "Capital"] = capital_current
                    df.at[df.index[i], "Position"] = 0

            trades_df = pd.DataFrame(trade_log)
            nb_trades = len(trades_df)
            winrate = (trades_df['Result (€)'] > 0).mean() * 100 if nb_trades > 0 else 0
            avg_gain = trades_df.loc[trades_df['Result (€)'] > 0, 'Result (€)'].mean() if nb_trades > 0 else 0
            avg_loss = trades_df.loc[trades_df['Result (€)'] <= 0, 'Result (€)'].mean() if nb_trades > 0 else 0
            max_drawdown = max(1 - df['Capital'] / df['Capital'].cummax()) * 100
            expectancy = (winrate/100)*avg_gain + ((100 - winrate)/100)*avg_loss if nb_trades > 0 else 0
            returns = df["Capital"].pct_change().dropna()
            sharpe = np.sqrt(252)*returns.mean()/returns.std() if returns.std() > 0 else 0
            pnl_eur = df["Capital"].iloc[-1] - capital
            pnl_pct = (df["Capital"].iloc[-1] / capital - 1)*100

            st.header(f"Backtest {pair1} - Durée : {dur_label}")
            st.line_chart(df["Capital"])
            st.markdown(f"**PnL (€) :** {pnl_eur:.2f} €")
            st.markdown(f"**PnL (%) :** {pnl_pct:.2f} %")
            st.markdown(f"**Sharpe Ratio :** {sharpe:.2f}")
            st.markdown(f"**Nombre de trades :** {nb_trades}")
            st.markdown(f"**Winrate (%) :** {winrate:.2f} %")
            st.markdown(f"**Max Drawdown (%) :** {max_drawdown:.2f} %")
            st.markdown(f"**Espérance de gain par trade :** {expectancy:.2f} €")

            st.subheader("Journal de trading détaillé")
            st.dataframe(trades_df)

            st.markdown("Distribution des signaux :")
            st.write(pd.Series(df["Signal"]).value_counts())