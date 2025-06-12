import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fx_strategy_V2 import get_all_data, engle_granger_test, calculate_adx, prepare_dataset_signal

def simulate_trades_with_sl_tp(y_test, y_pred, price_series, capital=900, tp_pips=40, sl_pips=20, lot_size=10000):
    """
    Simule des trades en utilisant les signaux ML, TP et SL fixes.
    """
    pip_value = 0.0001  # EURUSD
    balance = capital
    capital_curve = [balance]
    prices = price_series.loc[y_test.index]

    for idx, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
        entry_price = prices.iloc[idx]

        if pred_label == 0:
            capital_curve.append(balance)
            continue

        # Détermination direction
        is_long = pred_label == 1
        tp_price = entry_price + pip_value * tp_pips if is_long else entry_price - pip_value * tp_pips
        sl_price = entry_price - pip_value * sl_pips if is_long else entry_price + pip_value * sl_pips

        # Hypothèse : soit le TP soit le SL est touché (on choisit selon vérité pour simplifier)
        hit_tp = (pred_label == true_label)
        exit_price = tp_price if hit_tp else sl_price
        pnl = (exit_price - entry_price) * lot_size if is_long else (entry_price - exit_price) * lot_size

        balance += pnl
        capital_curve.append(balance)

    return capital_curve


def simple_ml_backtest(pair1="EURUSD=X", pair2="GBPUSD=X", commodity1="GC=F", commodity2="CL=F", seuil=1, capital=900):
    # Chargement des données
    df1, df2, df_commo1, df_commo2 = get_all_data(pair1, pair2, commodity1, commodity2)

    spread, _, _, _ = engle_granger_test(df1[f"{pair1}_Close"], df2[f"{pair2}_Close"])
    zscore = (spread - spread.mean()) / spread.std()
    pair1_close = df1[f"{pair1}_Close"]
    gold_price = df_commo1[f"{commodity1}_Close"] if not df_commo1.empty else pair1_close
    adx_series = calculate_adx(pair1)
    adx = adx_series.reindex(spread.index).fillna(method="bfill")

    # Dataset ML
    X, y = prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx, seuil=seuil)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

    # Modèle simple
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, class_weight='balanced')
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy simple ML backtest : {acc:.4f}")

    # Simuler un backtest plus réaliste avec TP/SL
    price_series = pair1_close
    capital_curve = simulate_trades_with_sl_tp(y_test, y_pred, price_series, capital=capital)
    #print(f"Capital final (SL/TP fixed) : {capital_curve[-1]:.2f} €")

    # Visualisation
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(capital_curve, label="Capital avec SL/TP")
    plt.title("Évolution du capital (avec SL/TP)")
    plt.xlabel("Trades")
    plt.ylabel("Capital (€)")
    plt.grid(True)
    plt.legend()
    plt.show()

    return acc, y_test, y_pred

if __name__ == "__main__":
    simple_ml_backtest()


# --- Simulation de trades avec SL/TP fixes ---

if __name__ == "__main__":
    # Lancement propre du backtest ML avec SL/TP
    simple_ml_backtest()