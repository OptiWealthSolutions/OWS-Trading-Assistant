import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import pandas as pd
from settings import tickers_default, entry_price_ticker_default, current_capital
from fpdf import FPDF
# Quant
from utils.Quant.risk_management import gestion_risque_adaptative
from utils.Quant.SL_sizing import atr_index
from utils.Quant.SL_sizing import sl_sizing
from utils.Quant.vol_index import get_vol_index
# Macro
from utils.Technical.seasonality import seasonality
# Technical (corrige le nom du fichier ici si nécessaire !)
from utils.Macro.commodities_graph import plot_currency_vs_commodities
# PDF generator (si c’est dans main.py tu n’as pas besoin de l’importer)


# --------------- Main call fonction----------------

def main_call():
    print("=== Commodities Correlation ===")
    correlation = plot_currency_vs_commodities(tickers_default)
    print(correlation)

    print("\n=== Volatility Index PDF ===")
    get_vol_index(tickers_default)

    print("\n=== Seasonality ===")
    seasonality(tickers_default)
    
    print("\n=== Stop Loss Sizing ===")
    print(atr_index(tickers_default))
    atr,k = atr_index(tickers_default)
    
    print(sl_sizing(atr,k,entry_price_ticker_default,"SELL"))
    print("\n=== Risk Management ===")
    df = gestion_risque_adaptative(current_capital,tickers_default)
    return df

main_call()