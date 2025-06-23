# --------------- APPEL DE TOUS LES MODULES ET FONCITONS EXTERNES AUX FICHIER ----------------

# Le modèle analyse la relation entre deux paires de devises fortement corrélées (pair trading) et détecte les déviations anormales de leur écart de prix via un z-score.

# Il combine cette information avec des indicateurs techniques (RSI, ADX), des données fondamentales (écart de taux d’intérêt) et la corrélation avec certaines matières premières pour prédire un signal de marché (BUY, SELL, WAIT).

# Le modèle est entraîné avec des techniques de machine learning supervisé (régression logistique multinomiale) pour apprendre à reconnaître les configurations historiques gagnantes.

# Chaque signal est accompagné d’un niveau de confiance probabiliste, et une gestion du risque adaptative ajuste automatiquement la taille de position selon la volatilité et le capital disponible.

# Le résultat est un tableau PDF clair listant les signaux du jour pour chaque paire de devises, la probabilité de succès estimée, et la taille de position recommandée.
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
from fx_strategy_V3 import get_interest_rate_difference
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import warnings

# Configuration de la police Menlo
menlo_path = '/System/Library/Fonts/Menlo.ttc'  # Chemin système macOS pour la police Menlo
menlo_prop = fm.FontProperties(fname=menlo_path)
plt.rcParams['font.family'] = menlo_prop.get_name()

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")
# --------------- PREPARE DATA SET FONCTION ----------------

def prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx, macro_data=None, seuil=1):
    # Vérification et préparation des données
    if isinstance(pair1_close, pd.DataFrame):
        pair1_close = pair1_close.iloc[:, 0]
    if isinstance(gold_price, pd.DataFrame):
        gold_price = gold_price.iloc[:, 0]
    
    # Alignement des indices
    pair1_close = pair1_close.reindex(spread.index)
    gold_price = gold_price.reindex(spread.index)
    zscore = zscore.reindex(spread.index)
    adx = adx.reindex(spread.index)
    
    # Calcul des indicateurs techniques
    rsi_pair1 = RSIIndicator(close=pair1_close, window=14).rsi()
    sma_20 = SMAIndicator(close=pair1_close, window=20).sma_indicator()
    ema_20 = EMAIndicator(close=pair1_close, window=20).ema_indicator()
    macd = MACD(close=pair1_close).macd_diff()
    bb_bands = BollingerBands(close=pair1_close, window=20)
    bb_bbh = bb_bands.bollinger_hband()
    bb_bbl = bb_bands.bollinger_lband()
    roc = ROCIndicator(close=pair1_close, window=12).roc()
    atr = AverageTrueRange(high=pair1_close, low=pair1_close, close=pair1_close).average_true_range()
    
    # Récupération de l'écart de taux d'intérêt
    rate_diff = get_interest_rate_difference(pair1_close.name if hasattr(pair1_close, 'name') else "")

    # Création du DataFrame avec toutes les features
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
        'rate_diff': rate_diff
    })

    # Ajout des données macro si disponibles
    if macro_data is not None:
        macro_data_reindexed = macro_data.reindex(df.index).ffill().bfill()
        df = pd.concat([df, macro_data_reindexed], axis=1)

    # Nettoyage des données
    df = df.dropna()
    df = df.astype(float)

    # Définition des targets
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

# --------------- SAUVERGARDE EN PDF ----------------

def save_results_to_pdf(df_results, filename="ml_signals_report_V3.pdf"):
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

    # Ajuster la couleur du texte et la police Menlo
    for i, sig in enumerate(df_results['signal']):
        for j in range(len(df_results.columns)):
            cell = table[i+1, j]  # +1 car ligne 0 = header
            cell.get_text().set_fontproperties(menlo_prop)  # force Menlo font
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

# --------------- TETS DE LA STRATEGIE SUR TOUTES LES PAIRES DE "settings.py" ----------------

# --------------- TEST SUR TOUTS LAES PAIRES AVEC GENERATION DE PDF SEULEMENT (MAIN CALL)----------------

def test_all_pairs_pdf_only(capital=900):
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

        smote = SMOTE(random_state=42, sampling_strategy='auto')
        X_resampled, y_resampled = smote.fit_resample(X, y)

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('randomforestclassifier', RandomForestClassifier())
        ])
        model.fit(X_resampled, y_resampled)

        X_pred = X.iloc[[-1]]  # reste un DataFrame avec les noms de colonnes
        y_pred_class = model.predict(X_pred)[0]
        y_proba = model.predict_proba(X_pred)[0]
        classes = model.named_steps['randomforestclassifier'].classes_
        class_index = np.where(classes == y_pred_class)[0][0]
        confidence = y_proba[class_index]

        if y_pred_class == 2:
            signal = 'BUY'
        elif y_pred_class == 0:
            signal = 'SELL'
        else:
            signal = 'WAIT'

        # Gestion du risque adaptative selon volatilité et capital
        risk_df = gestion_risque_adaptative(capital, pair1)
        risk_percent = risk_df['Risk %'].iloc[0]
        risk_amount = risk_df['Risk €'].iloc[0]

        results.append({
            'pair1': pair1,
            'pair2': pair2,
            'predicted_class': y_pred_class,
            'confidence': round(confidence, 4),
            'signal': signal,
            'risk_percent': risk_percent,
            'risk_amount': risk_amount
        })

    df_results = pd.DataFrame(results)
    # Ensure risk columns are present and in the DataFrame
    if not df_results.empty:
        # Optional: order columns for PDF clarity
        cols = ['pair1', 'pair2', 'predicted_class', 'confidence', 'signal', 'risk_percent', 'risk_amount']
        df_results = df_results[cols]
    save_results_to_pdf(df_results)
    return df_results

# --------------- GESTION DE RISQUE ADAPTATIVE ----------------

def gestion_risque_adaptative(capital, ticker, fx_df_std=None, fx_df_vol=None, max_risk=0.02, min_risk=0):
    if fx_df_std is None or fx_df_vol is None:
        # téléchargement seulement si nécessaire
        fx_std_data = yf.download(ticker, period="6mo", interval="4h", progress=False)
        fx_df_std = pd.DataFrame(fx_std_data)
        fx_df_std['Log Returns'] = np.log(fx_df_std['Close'] / fx_df_std['Close'].shift(1))
        fx_df_std['STD'] = fx_df_std['Log Returns'].std()

        fx_vol_data_brut = yf.download(ticker, period="6mo", interval="4h", progress=False)
        fx_df_vol = pd.DataFrame(fx_vol_data_brut)
        fx_df_vol['Log Returns'] = np.log(fx_df_vol['Close'] / fx_df_vol['Close'].shift(1))
        fx_df_vol['Volatility_20D'] = fx_df_vol['Log Returns'].rolling(window=20).std(ddof=0) * 100
        fx_df_vol.dropna(inplace=True)

    current_std = fx_df_std['STD'].iloc[-1]
    current_vol = fx_df_vol['Volatility_20D'].iloc[-1]

    poids_vol = 0.5
    poids_std = 0.5
    score_risque = current_std * poids_std + current_vol * poids_vol 
    risque_pct = max(min_risk, max_risk * (1 - score_risque))
    risque_pct = float(round(risque_pct * 100, 2))
    risk_amount = round(capital * (risque_pct / 100), 2)

    final_df = pd.DataFrame([{
        'Vol %': round(current_vol, 4) * 100,
        'Std %': round(current_std, 4) * 100,
        'Risk €': risk_amount,
        'Risk %': risque_pct
    }])

    return final_df

# --------------- ZONE D'APPEL DES FONCTION GÉNÉRAL ----------------

if __name__ == "__main__":
    test_all_pairs_pdf_only()

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")