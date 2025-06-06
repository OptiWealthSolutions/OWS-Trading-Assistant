import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from settings import *
from utils import *


# --------------- Main call fonction----------------

def main_call():
    print("=== Commodities Correlation ===")
    # plot_currency_vs_commodities affiche directement le graphique, pas besoin d'assigner
    plot_currency_vs_commodities(tickers_default)
    
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