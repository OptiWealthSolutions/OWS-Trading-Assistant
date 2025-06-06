import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import pandas as pd
from settings import *
from fpdf import FPDF
# Quant
from utils.Quant.risk_management import gestion_risque_adaptative
from utils.Quant.SL_sizing import atr_index
from utils.Quant.SL_sizing import sl_sizing
from utils.Quant.vol_index import get_vol_index
from strategies.pairs_trade_sys import pairs_trading_summary
# Macro
from utils.Macro.seasonality import seasonality
from utils.Technical.indicators_signals import sma_crossing
from utils.Technical.trend_following import calculate_adx
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
    
    print("\n=== Technics Signals ===")
    sma_crossing(tickers_default)
    calculate_adx(tickers_default)
    
    print("\n=== Pairs Trading Summary ===")
    # Exemple avec une paire forex corrélée
    ticker1 = tickers_default
    ticker2 = "GBPUSD=X"  # ou une autre paire corrélée
    summary = pairs_trading_summary(ticker1, ticker2)
    print(summary)
    
    print("\n=== Stop Loss Sizing ===")
    print(atr_index(tickers_default))
    atr,k = atr_index(tickers_default)
    print(sl_sizing(atr,k,entry_price_ticker_default,"BUY"))
    
    print("\n=== Risk Management ===")
    df = gestion_risque_adaptative(current_capital,tickers_default)


    return df


main_call()