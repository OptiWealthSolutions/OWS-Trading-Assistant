import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from settings import *
from utils.Quant import *
from utils.Macro import *
from utils.Technical import *
from utils.Macro import *

# --------------- Main call fonction----------------

def main_call():
    print("=== Commodities Correlation ===")
    correlation = plot_currency_vs_commodities(tickers_default)
    print(correlation)

    print("\n=== Risk Management ===")
    print(gestion_risque_adaptative(current_capital,tickers_default))

    print("\n=== Stop Loss Sizing ===")
    print(atr_index(tickers_default,20,"6mo"))
    atr,k = atr_index(tickers_default,20,"6mo")
    print(sl_sizing(atr,k,entry_price_ticker_default,"SELL"))

    print("\n=== Volatility Index PDF ===")
    get_vol_index(tickers_default)

    print("\n=== Seasonality ===")
    seasonality(tickers_default)

    print("\n=== Generating PDF Report ===")
    pdf_report_ticker()

main_call()
