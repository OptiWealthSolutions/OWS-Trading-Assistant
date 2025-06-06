import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from settings import *

# Import des fonctions depuis les fichiers spécifiques dans utils
from utils.Quant import gestion_risque_adaptative, atr_index, sl_sizing, get_vol_index
from utils.Macro.seasonality import seasonality
from utils.Technical.plot_tools import plot_currency_vs_commodities

# --------------- Main call fonction----------------

def main_call():
    print("=== Commodities Correlation ===")
    # plot_currency_vs_commodities affiche directement le graphique, pas besoin d'assigner
    print(plot_currency_vs_commodities(tickers_default))
    
    print("\n=== Seasonality ===")
    seasonality(tickers_default)
    
    print("\n=== Volatility Index ===")
    get_vol_index(tickers_default)
    
    print("\n=== Stop Loss Sizing ===")
    atr, k = atr_index(tickers_default, 20, "6mo")
    print(f"ATR: {atr}, k: {k}")
    sl = sl_sizing(atr, k, entry_price_ticker_default, "SELL")
    print(f"Stop Loss (pips): {sl}")
    
    print("\n=== Risk Management ===")
    risk_df = gestion_risque_adaptative(1000, tickers_default)
    risk_df
    
    return "done"

main_call()