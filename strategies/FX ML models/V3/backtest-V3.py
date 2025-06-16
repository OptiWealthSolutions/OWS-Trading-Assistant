import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from settings import tickers, commodity_mapping
from ML_model_fx_strat_V3 import prepare_dataset_signal, get_all_data, engle_granger_test, calculate_adx, gestion_risque_adaptative, get_macro_data_fred
from xgboost import XGBClassifier

# --------- Optuna tuning block (injection) ---------
import optuna
from sklearn.model_selection import cross_val_score

@st.cache_resource
def tune_xgb_hyperparams(X, y):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0)
        }
        model = make_pipeline(StandardScaler(), XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, **params))
        score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40)
    return study.best_params

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
threshold = st.slider("Seuil de confiance du modèle (0 = agressif, 1 = conservateur)", 0.5, 0.95, 0.7, step=0.05)

if st.button("Lancer le backtest"):
    # --- Multi-paires, capital partagé, backtest global ---
    for dur_label in selected_durations:
        days = durations[dur_label]
        # 1. Charger les données pour chaque paire, préparer signaux, aligner les index
        pair_data = {}
        min_end_date = None
        max_start_date = None
        for pair1 in selected_pairs:
            i = tickers.index(pair1)
            pair2 = tickers[i+1] if i+1 < len(tickers) else tickers[i-1]
            base_currency1 = pair1[:3]
            commodity1 = commodity_mapping.get(base_currency1, None)
            df1, df2, df_commo1, _ = get_all_data(pair1, pair2, commodity1, None)
            if df1.empty or df2.empty:
                st.error(f"Données manquantes pour {pair1}")
                continue
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
            y = y.loc[X.index]
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
            st.write(f"Optimisation des hyperparamètres pour {pair1} ...")
            best_params = tune_xgb_hyperparams(X_res, y_res)
            model = make_pipeline(StandardScaler(), XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, **best_params))
            model.fit(X_res, y_res)
            y_proba = model.predict_proba(X)
            proba_buy = y_proba[:, 2]
            proba_sell = y_proba[:, 0]
            df = df1_filt.reindex(X.index).copy()
            df["Hour"] = df.index.hour
            atr = compute_atr(pair1)
            min_atr = 0.001
            valid_volatility = atr > min_atr
            valid_hours = ~df["Hour"].between(22, 6, inclusive="neither")
            mask_time_vol = valid_volatility & valid_hours
            X = X[mask_time_vol]
            y = y[mask_time_vol]
            proba_buy = proba_buy[mask_time_vol]
            proba_sell = proba_sell[mask_time_vol]
            df = df.loc[mask_time_vol]
            df["Signal"] = 1
            for idx in range(len(df)):
                if proba_buy[idx] >= threshold:
                    df.at[df.index[idx], "Signal"] = 2
                elif proba_sell[idx] >= threshold:
                    df.at[df.index[idx], "Signal"] = 0
            df["Price"] = df[f"{pair1}_Close"]
            df = df[["Price", "Signal"]]
            pair_data[pair1] = df
            # Pour alignement index commun
            if min_end_date is None or df.index.max() < min_end_date:
                min_end_date = df.index.max()
            if max_start_date is None or df.index.min() > max_start_date:
                max_start_date = df.index.min()
        # 2. Calculer l'index commun
        if not pair_data:
            st.warning("Aucune donnée de paire disponible pour ce backtest.")
            continue
        # Intersect all index
        common_index = None
        for df in pair_data.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        # Limiter à la période commune
        common_index = common_index[(common_index >= max_start_date) & (common_index <= min_end_date)]
        # 3. Initialiser portefeuille global
        rr = 2
        pip_value = 0.0001
        commission = 0
        spread_pips = 1
        slippage_pips = 0.5
        capital_current = capital
        open_positions = {}  # {pair: {...}}
        portfolio_log = []
        # 4. Boucle sur chaque timestamp commun
        for t in common_index[1:]:
            signals = {pair: int(pair_data[pair].at[t, "Signal"]) for pair in pair_data.keys() if t in pair_data[pair].index}
            prices = {pair: float(pair_data[pair].at[t, "Price"]) for pair in pair_data.keys() if t in pair_data[pair].index}
            # 4a. Fermer les positions si TP/SL/time
            closed_pairs = []
            for pair, pos in open_positions.items():
                price = prices.get(pair, None)
                if price is None:
                    continue
                atr = compute_atr(pair)
                risk_info = gestion_risque_adaptative(capital_current, pair)
                risk_amount = risk_info['Risk €'].iloc[0]
                # BUY
                if pos['direction'] == 1:
                    if price <= pos['stop_loss']:
                        capital_current -= commission
                        loss = -pos['risk']
                        capital_current += loss
                        portfolio_log.append({
                            'Time': t, 'Pair': pair, 'Type': 'SL', 'Result (€)': loss - commission, 'Capital': capital_current
                        })
                        closed_pairs.append(pair)
                    elif price >= pos['take_profit']:
                        capital_current -= commission
                        gain = pos['risk'] * rr
                        capital_current += gain
                        portfolio_log.append({
                            'Time': t, 'Pair': pair, 'Type': 'TP', 'Result (€)': gain - commission, 'Capital': capital_current
                        })
                        closed_pairs.append(pair)
                    elif (t - pos['entry_time']) > pd.Timedelta(days=2):
                        pnl = (price - pos['entry_price']) * (pos['risk'] / (abs(pos['entry_price'] - pos['stop_loss']) if pos['entry_price'] != pos['stop_loss'] else 1))
                        capital_current -= commission
                        capital_current += pnl
                        portfolio_log.append({
                            'Time': t, 'Pair': pair, 'Type': 'TimeExit', 'Result (€)': pnl - commission, 'Capital': capital_current
                        })
                        closed_pairs.append(pair)
                # SELL
                elif pos['direction'] == -1:
                    if price >= pos['stop_loss']:
                        capital_current -= commission
                        loss = -pos['risk']
                        capital_current += loss
                        portfolio_log.append({
                            'Time': t, 'Pair': pair, 'Type': 'SL', 'Result (€)': loss - commission, 'Capital': capital_current
                        })
                        closed_pairs.append(pair)
                    elif price <= pos['take_profit']:
                        capital_current -= commission
                        gain = pos['risk'] * rr
                        capital_current += gain
                        portfolio_log.append({
                            'Time': t, 'Pair': pair, 'Type': 'TP', 'Result (€)': gain - commission, 'Capital': capital_current
                        })
                        closed_pairs.append(pair)
                    elif (t - pos['entry_time']) > pd.Timedelta(days=2):
                        pnl = (pos['entry_price'] - price) * (pos['risk'] / (abs(pos['entry_price'] - pos['stop_loss']) if pos['entry_price'] != pos['stop_loss'] else 1))
                        capital_current -= commission
                        capital_current += pnl
                        portfolio_log.append({
                            'Time': t, 'Pair': pair, 'Type': 'TimeExit', 'Result (€)': pnl - commission, 'Capital': capital_current
                        })
                        closed_pairs.append(pair)
            for pair in closed_pairs:
                del open_positions[pair]
            # 4b. Ouvrir les nouvelles positions si signal et non déjà ouvert
            for pair in pair_data.keys():
                if pair in open_positions:
                    continue
                signal = signals.get(pair, 1)
                price = prices.get(pair, None)
                if price is None:
                    continue
                atr = compute_atr(pair)
                risk_info = gestion_risque_adaptative(capital_current, pair)
                risk_amount = risk_info['Risk €'].iloc[0]
                if signal == 2:
                    entry_price = price + (spread_pips + slippage_pips) * pip_value
                    stop_loss = entry_price - atr * 1.5
                    take_profit = entry_price + rr * (entry_price - stop_loss)
                    open_positions[pair] = {
                        'direction': 1, 'entry_price': entry_price, 'entry_time': t,
                        'stop_loss': stop_loss, 'take_profit': take_profit, 'risk': risk_amount
                    }
                    capital_current -= commission
                    portfolio_log.append({
                        'Time': t, 'Pair': pair, 'Type': 'Entry BUY', 'Result (€)': 0, 'Capital': capital_current
                    })
                elif signal == 0:
                    entry_price = price - (spread_pips + slippage_pips) * pip_value
                    stop_loss = entry_price + atr * 1.5
                    take_profit = entry_price - rr * (stop_loss - entry_price)
                    open_positions[pair] = {
                        'direction': -1, 'entry_price': entry_price, 'entry_time': t,
                        'stop_loss': stop_loss, 'take_profit': take_profit, 'risk': risk_amount
                    }
                    capital_current -= commission
                    portfolio_log.append({
                        'Time': t, 'Pair': pair, 'Type': 'Entry SELL', 'Result (€)': 0, 'Capital': capital_current
                    })
            # 4c. Log du portefeuille à chaque timestamp
            portfolio_log.append({
                'Time': t, 'Pair': 'Portfolio', 'Type': 'Snapshot', 'Result (€)': 0, 'Capital': capital_current
            })
        # 5. DataFrame résultat portefeuille
        portefeuille = pd.DataFrame(portfolio_log)
        portefeuille = portefeuille.sort_values("Time").reset_index(drop=True)
        portefeuille = portefeuille[portefeuille['Pair'] == 'Portfolio']
        portefeuille = portefeuille.drop_duplicates(subset=["Time"], keep='last')
        portefeuille = portefeuille.set_index("Time")
        # 6. Calcul des métriques
        returns = portefeuille["Capital"].pct_change().dropna()
        sharpe = np.sqrt(252)*returns.mean()/returns.std() if returns.std() > 0 else 0
        pnl_eur = portefeuille["Capital"].iloc[-1] - capital
        pnl_pct = (portefeuille["Capital"].iloc[-1] / capital - 1)*100
        max_drawdown = max(1 - portefeuille['Capital'] / portefeuille['Capital'].cummax()) * 100
        st.header(f"Backtest Multi-paires (capital partagé) - Durée : {dur_label}")
        st.line_chart(portefeuille["Capital"])
        col1, col2, col3 = st.columns(3)
        col1.metric("PnL (€)", f"{pnl_eur:.2f} €")
        col2.metric("PnL (%)", f"{pnl_pct:.2f} %")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Drawdown Max (%)", f"{max_drawdown:.2f} %")
        st.subheader("Courbe du portefeuille global (capital partagé)")
        st.dataframe(portefeuille)