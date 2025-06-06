import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from settings import currency_commodity_map


def plot_currency_vs_commodities(ticker="AUDUSD", period="6mo", interval="1d"):
    base = ticker[:3]  # Exemple : "AUD" depuis "AUDUSD"
    commodities = currency_commodity_map.get(base.upper(), [])
    
    if not commodities:
        print(f"Aucune commodité connue pour {base}")
        return

    # Télécharger données
    forex = yf.download(ticker, period=period, interval=interval)["Close"].rename(ticker)
    data = pd.DataFrame(forex)

    for commo in commodities:
        commo_data = yf.download(commo, period=period, interval=interval)["Close"].rename(commo)
        data = data.join(commo_data, how="inner")
    
    data.dropna(inplace=True)
    
    # Affichage
    for commo in commodities:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        ax1.plot(data[ticker], label=ticker, color="blue")
        ax2.plot(data[commo], label=commo, color="orange", alpha=0.6)

        plt.title(f"{ticker} vs {commo}")
        ax1.set_ylabel(f"{ticker}", color="blue")
        ax2.set_ylabel(f"{commo}", color="orange")
        plt.legend()
        plt.grid()
        plt.show()

        # Corrélation glissante
        rolling_corr = data[ticker].pct_change().rolling(20).corr(data[commo].pct_change())
        rolling_corr.plot(title=f"Rolling Correlation ({ticker} vs {commo}) - 20 périodes", figsize=(10,3), color="green")
        plt.grid()
        plt.show()