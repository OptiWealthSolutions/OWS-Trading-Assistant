import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from settings import *
from utils import *
from settings import tickers_default
def main_call():
    print("=== Commodities Correlation ===")
    correlations = plot_currency_vs_commodities("EURUSD=X")
    

    print("\n=== Risk Management ===")
    print(gestion_risque_adaptative(1000,tickers_default))

    print("\n=== Stop Loss Sizing ===")
    print(atr_index(tickers_default,20,"6mo"))
    atr,k = atr_index(tickers_default,20,"6mo")
    print(sl_sizing(atr,k,entry_price_ticker_default,"SELL"))

    print("\n=== Volatility Index PDF ===")
    get_vol_index(tickers_default)

    print("\n=== Seasonality ===")
    seasonality(tickers_default)

if __name__ == "__main__":
    main_call()