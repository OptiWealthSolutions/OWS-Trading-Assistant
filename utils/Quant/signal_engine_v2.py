from statsmodels.tsa.stattools import adfuller
import pandas as pd
from data_loader import data_loader     
import statsmodels.api as sm
import matplotlib.pyplot as plt
from cointegration import engle_granger_test

# Tickers utilisés
ticker1 = "EURUSD=X"
ticker2 = "EURJPY=X"

# Fonction pour calculer le z-score du spread
def z_score(df: pd.DataFrame, ticker1: str, ticker2: str) -> pd.Series:
    y = df[f"{ticker1}_Close"]
    x = df[f"{ticker2}_Close"]
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    spread = y - model.predict(x)
    zscore = (spread - spread.mean()) / spread.std()
    return zscore    

# Fonction pour générer les signaux d'arbitrage
def signal_engine(df_name: pd.DataFrame, zscore_series: pd.Series, alpha: float):
    df = df_name.copy()  # On garde toutes les colonnes

    # S'assurer que les index coïncident
    zscore_series = zscore_series.reindex(df.index)

    # Ajouter la colonne Z-Score
    df["Z-Score"] = zscore_series

    # Initialiser la colonne Signal à 0
    df["Signal"] = 0

    # Définir les signaux
    df.loc[df["Z-Score"] > alpha, "Signal"] = 1    # Signal de vente
    df.loc[df["Z-Score"] < -alpha, "Signal"] = -1  # Signal d'achat

    return df

# Chargement des données
df = data_loader(ticker1, ticker2, "1y")

# Calcul du z-score et génération des sign-aux
zscore_series = z_score(df, ticker1, ticker2)
zscore_values = z_score(df, "EURUSD=X", "EURJPY=X")
signals = signal_engine(df, zscore_values, alpha=1.0)
# Affichage final
signals.tail()

#visualisation graphique :
import matplotlib.pyplot as plt

def plot_zscore_signals(signals_df):
    plt.figure(figsize=(14, 6))

    # Tracer la courbe du z-score
    plt.plot(signals_df.index, signals_df["Z-Score"], label="Z-Score", color="blue")

    # Lignes horizontales de seuil
    plt.axhline(1.0, color='red', linestyle='--', label='Seuil +1')
    plt.axhline(-1.0, color='green', linestyle='--', label='Seuil -1')
    plt.axhline(0.0, color='black', linestyle='-')

    # Signaux d'achat
    buy_signals = signals_df[signals_df["Signal"] == -1]
    plt.scatter(buy_signals.index, buy_signals["Z-Score"], label='BUY', color='green', marker='^', s=100)

    # Signaux de vente
    sell_signals = signals_df[signals_df["Signal"] == 1]
    plt.scatter(sell_signals.index, sell_signals["Z-Score"], label='SELL', color='red', marker='v', s=100)

    # Personnalisation
    plt.title("Z-Score avec signaux de trading")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Appel de la fonction avec tes données
plot_zscore_signals(signals)
