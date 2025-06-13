from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import ADXIndicator, SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yfinance as yf
from fx_strategy_V3 import *
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import settings
from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import shap
from sklearn.model_selection import TimeSeriesSplit
import shap

def prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx, macro_data=None, seuil=1):
    if isinstance(pair1_close, pd.DataFrame):
        pair1_close = pair1_close.iloc[:, 0]
    if isinstance(gold_price, pd.DataFrame):
        gold_price = gold_price.iloc[:, 0]

    pair1_close = pair1_close.reindex(spread.index)
    gold_price = gold_price.reindex(spread.index)
    zscore = zscore.reindex(spread.index)
    adx = adx.reindex(spread.index)

    rsi_pair1 = RSIIndicator(close=pair1_close, window=14).rsi()
    sma_20 = SMAIndicator(close=pair1_close, window=20).sma_indicator()
    ema_20 = EMAIndicator(close=pair1_close, window=20).ema_indicator()
    macd = MACD(close=pair1_close).macd_diff()
    bb_bands = BollingerBands(close=pair1_close, window=20)
    bb_bbh = bb_bands.bollinger_hband()
    bb_bbl = bb_bands.bollinger_lband()
    roc = ROCIndicator(close=pair1_close, window=12).roc()
    atr = AverageTrueRange(high=pair1_close, low=pair1_close, close=pair1_close).average_true_range()

    rate_diff = get_interest_rate_difference(pair1_close.name if hasattr(pair1_close, 'name') else "")

    df = pd.DataFrame({
        'spread': spread,
        'z_score': zscore,
        'z_score_lag1': zscore.shift(1),
        'vol_spread': spread.rolling(30).std(),
        'rsi_pair1': rsi_pair1,
        'adx': adx,
        'sma_20': sma_20,
        'ema_20': ema_20,
        'macd': macd,
        'bb_high': bb_bbh,
        'bb_low': bb_bbl,
        'roc': roc,
        'atr': atr,
        'rate_diff': get_interest_rate_difference(pair1_close.name if hasattr(pair1_close, 'name') else "")
    })

    if macro_data is not None:
        # Merge macro data on index (dates), align on dates
        macro_data_reindexed = macro_data.reindex(df.index).ffill().bfill()
        df = pd.concat([df, macro_data_reindexed], axis=1)

    df.dropna(inplace=True)
    df = df.astype(float)

    # Nouvelle étiquette : -1 → 0 (SELL), 0 → 1 (WAIT), 1 → 2 (BUY)
    df['target'] = 1  # WAIT par défaut
    df.loc[df['z_score'] > seuil, 'target'] = 0  # SELL
    df.loc[df['z_score'] < -seuil, 'target'] = 2  # BUY

    feature_cols = ['z_score', 'z_score_lag1', 'vol_spread', 'rsi_pair1', 'adx',
            'sma_20', 'ema_20', 'macd', 'bb_high', 'bb_low', 'roc', 'atr', 'rate_diff']
    if macro_data is not None:
        feature_cols += list(macro_data.columns)

    X = df[feature_cols]
    y = df['target']

    return X, y


def save_results_to_pdf(df_results, filename="ml_signals_report.pdf"):
    fig, ax = plt.subplots(figsize=(12, len(df_results)*0.5 + 1))
    ax.axis('off')

    # Couleurs lignes selon signal
    row_colors = []
    for sig in df_results['signal']:
        if sig == 'SELL':
            row_colors.append("#ff4d4d89")  # rouge clair
        elif sig == 'BUY':
            row_colors.append("#4CAF4F76")  # vert
        else:
            row_colors.append('white')    # blanc fond neutre

    # Création du tableau matplotlib
    table = ax.table(cellText=df_results.values,
                     colLabels=df_results.columns,
                     cellColours=[[color]*len(df_results.columns) for color in row_colors],
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Ajuster la couleur du texte (noir par défaut, sauf rouge/vert sur signal)
    for i, sig in enumerate(df_results['signal']):
        for j in range(len(df_results.columns)):
            cell = table[i+1, j]  # +1 car ligne 0 = header
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
    results = []
    tickers = settings.tickers
    # Get macro data once
    macro_data = get_macro_data_fred()
    # Loop over consecutive pairs of tickers
    for i in range(len(tickers) - 1):
        pair1 = tickers[i]
        pair2 = tickers[i + 1]
        # Determine commodity for pair1 currency from mapping, fallback to pair1 close if none
        base_currency1 = pair1[:3]
        commodity1 = settings.commodity_mapping.get(base_currency1, None)
        commodity2 = None

        df1, df2, df_commo1, df_commo2 = get_all_data(pair1, pair2, commodity1, commodity2)

        if df1.empty or df2.empty:
            print(f"Data not available for pair {pair1} or {pair2}. Skipping.")
            continue

        spread, _, _, _ = engle_granger_test(df1[f"{pair1}_Close"], df2[f"{pair2}_Close"])
        zscore = (spread - spread.mean()) / spread.std()

        pair1_close = df1[f"{pair1}_Close"]
        gold_price = df_commo1[f"{commodity1}_Close"] if (commodity1 and not df_commo1.empty) else pair1_close

        adx_series = calculate_adx(pair1)
        adx = adx_series.reindex(spread.index).bfill()

        X, y = prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx, macro_data=macro_data)

        if X.empty or y.empty:
            print(f"Insufficient data after preparation for pair {pair1} and {pair2}. Skipping.")
            continue

        if len(X) < 2:
            print(f"Not enough data points for pair {pair1} and {pair2}. Skipping.")
            continue

        # Apply SMOTE to balance classes
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(multi_class='multinomial', max_iter=10000, solver='lbfgs', class_weight='balanced')
        )

        # Use TimeSeriesSplit for evaluation
        tscv = TimeSeriesSplit(n_splits=5)
        for fold, (train_index, test_index) in enumerate(tscv.split(X_resampled)):
            X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
            y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred_class = model.predict(X_test)
            print(f"Pair {pair1}-{pair2} Fold {fold + 1} - Accuracy: {accuracy_score(y_test, y_pred_class):.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred_class))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred_class))
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
            plt.title(f"Confusion Matrix for {pair1}-{pair2} Fold {fold + 1}")
            plt.show()

            # Probabilités (softmax) sur X_test
            probas = model.predict_proba(X_test)
            plt.plot(probas[:, 0], label="Prob SELL")
            plt.plot(probas[:, 1], label="Prob WAIT")
            plt.plot(probas[:, 2], label="Prob BUY")
            plt.legend()
            plt.title(f"Probabilités des classes dans le temps for {pair1}-{pair2} Fold {fold + 1}")
            plt.show()

        # Final prediction on last day
        y_pred_class_final = model.predict(X.iloc[-1:].values)[0]
        y_proba_final = model.predict_proba(X.iloc[-1:].values)[0]
        classes = model.named_steps['logisticregression'].classes_
        class_index = np.where(classes == y_pred_class_final)[0][0]
        confidence = y_proba_final[class_index]

        if y_pred_class_final == 2:
            signal = 'BUY'
        elif y_pred_class_final == 0:
            signal = 'SELL'
        else:
            signal = 'WAIT'

        results.append({
            'pair1': pair1,
            'pair2': pair2,
            'predicted_class': y_pred_class_final,
            'confidence': confidence,
            'signal': signal
        })

    df_results = pd.DataFrame(results)
    print("ML Model Prediction Results for all pairs:")
    print(df_results)

    save_results_to_pdf(df_results)

    return df_results


def test_single_pair(pair1, pair2):

    print(f"\n🔍 Test du modèle sur la paire : {pair1} / {pair2}")
    base_currency1 = pair1[:3]
    commodity1 = settings.commodity_mapping.get(base_currency1, None)

    df1, df2, df_commo1, _ = get_all_data(pair1, pair2, commodity1, None)
    if df1.empty or df2.empty:
        print("❌ Données indisponibles.")
        return

    macro_data = get_macro_data_fred()
    spread, _, _, _ = engle_granger_test(df1[f"{pair1}_Close"], df2[f"{pair2}_Close"])
    zscore = (spread - spread.mean()) / spread.std()
    pair1_close = df1[f"{pair1}_Close"]
    gold_price = df_commo1[f"{commodity1}_Close"] if (commodity1 and not df_commo1.empty) else pair1_close
    adx = calculate_adx(pair1).reindex(spread.index).bfill()

    X, y = prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx, macro_data=macro_data)
    if X.empty or len(X) < 10:
        print("❌ Données insuffisantes pour entraîner le modèle.")
        return

    print(f"✔️ Dataset prêt. Nombre de points : {len(X)}")
    print("Distribution des classes cibles :\n", y.value_counts())

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(multi_class='multinomial', max_iter=10000, solver='lbfgs', class_weight='balanced')
    )

    # Séparation train/test simple (20% test)
    X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

    model.fit(X_train, y_train)  # <-- Important : entraînement du modèle

    y_pred = model.predict(X_test)

    # Résumé prédiction finale sur test set
    results = []
    for idx in range(len(y_test)):
        pred_class = y_pred[idx]
        true_class = y_test.iloc[idx]
        confidence = np.max(model.predict_proba(X_test.iloc[[idx]]))
        signal = 'WAIT'
        if pred_class == 0:
            signal = 'SELL'
        elif pred_class == 2:
            signal = 'BUY'
        results.append({
            'index': y_test.index[idx],
            'true_class': true_class,
            'predicted_class': pred_class,
            'confidence': confidence,
            'signal': signal
        })
    df_results = pd.DataFrame(results)

    print("\n📊 Rapport de classification :")
    print(classification_report(y_test, y_pred))

    print("\n🧮 Matrice de confusion :")
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Matrice de confusion (SELL/WAIT/BUY)")
    plt.show()

    # Affiche la distribution des erreurs
    errors = y_test != y_pred
    plt.figure(figsize=(8,4))
    plt.hist(errors.astype(int), bins=3)
    plt.xticks([0,1])
    plt.title("Distribution des erreurs (0=correct, 1=erreur)")
    plt.show()

    # Courbes de probabilités prédites
    probas = model.predict_proba(X_test)
    plt.figure(figsize=(12, 5))
    plt.plot(probas[:, 0], label="Proba SELL")
    plt.plot(probas[:, 1], label="Proba WAIT")
    plt.plot(probas[:, 2], label="Proba BUY")
    plt.title("Probabilités des classes sur le test set")
    plt.xlabel("Index (temps)")
    plt.ylabel("Probabilité")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n📈 Explications SHAP :")
    explainer = shap.Explainer(model.named_steps['logisticregression'], X_train, feature_names=X.columns)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)
    # Sauvegarde résultats dans PDF
    save_results_to_pdf(df_results)

    # Validation croisée temporelle avec TimeSeriesSplit
    print("\n🔄 Validation croisée temporelle avec TimeSeriesSplit")
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []
    fold = 1
    for train_index, test_index in tscv.split(X_resampled):
        X_train_cv, X_test_cv = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train_cv, y_test_cv = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_test_cv)
        acc = accuracy_score(y_test_cv, y_pred_cv)
        accuracies.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        fold += 1
    print(f"Moyenne accuracy CV : {np.mean(accuracies):.4f}")


def test_all_pairs_pdf_only():
    results = []
    tickers = settings.tickers
    macro_data = get_macro_data_fred()

    for i in range(len(tickers) - 1):
        pair1 = tickers[i]
        pair2 = tickers[i + 1]
        base_currency1 = pair1[:3]
        commodity1 = settings.commodity_mapping.get(base_currency1, None)
        commodity2 = None

        df1, df2, df_commo1, df_commo2 = get_all_data(pair1, pair2, commodity1, commodity2)

        if df1.empty or df2.empty:
            continue

        spread, _, _, _ = engle_granger_test(df1[f"{pair1}_Close"], df2[f"{pair2}_Close"])
        zscore = (spread - spread.mean()) / spread.std()

        pair1_close = df1[f"{pair1}_Close"]
        gold_price = df_commo1[f"{commodity1}_Close"] if (commodity1 and not df_commo1.empty) else pair1_close

        adx_series = calculate_adx(pair1)
        adx = adx_series.reindex(spread.index).bfill()

        X, y = prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx, macro_data=macro_data)

        if X.empty or y.empty or len(X) < 2:
            continue

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(multi_class='multinomial', max_iter=10000, solver='lbfgs', class_weight='balanced')
        )
        model.fit(X_resampled, y_resampled)

        X_pred = X.iloc[-1:].values
        y_pred_class = model.predict(X_pred)[0]
        y_proba = model.predict_proba(X_pred)[0]
        classes = model.named_steps['logisticregression'].classes_
        class_index = np.where(classes == y_pred_class)[0][0]
        confidence = y_proba[class_index]

        if y_pred_class == 2:
            signal = 'BUY'
        elif y_pred_class == 0:
            signal = 'SELL'
        else:
            signal = 'WAIT'

        results.append({
            'pair1': pair1,
            'pair2': pair2,
            'predicted_class': y_pred_class,
            'confidence': confidence,
            'signal': signal
        })

    df_results = pd.DataFrame(results)
    save_results_to_pdf(df_results)
    return df_results

if __name__ == "__main__":
    test_all_pairs_pdf_only()
    pass