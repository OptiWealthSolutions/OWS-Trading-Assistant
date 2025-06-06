import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import *
from settings import tickers_default
def main_call():
    print("=== Commodities Correlation ===")
    correlations = plot_currency_vs_commodities("EURUSD=X")
    

    print("\n=== Risk Management ===")
    gestion_risque_adaptative(1000,tickers_default)

    print("\n=== Stop Loss Sizing ===")
    atr_index(tickers_default)

    print("\n=== Volatility Index PDF ===")
    get_vol_index()

    print("\n=== Seasonality ===")
    seasonality()

if __name__ == "__main__":
    main_call()