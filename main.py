import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import pandas as pd
from settings import *
from fpdf import FPDF
# Quant
from utils.Quant.risk_assessement.risk_management import gestion_risque_adaptative
from utils.Quant.risk_assessement.stop_sizing import atr_index, sl_sizing
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
    for ticker1, ticker2 in forex_pairs:
        if tickers_default in (ticker1, ticker2):
            other_ticker = ticker2 if ticker1 == tickers_default else ticker1
            summary = pairs_trading_summary(tickers_default, other_ticker)
            print(summary)
            print("=" * 80)
    
    print("\n=== Stop Loss Sizing ===")
    print(atr_index(tickers_default))
    atr,k = atr_index(tickers_default)
    print(sl_sizing(atr,k,entry_price_ticker_default,"BUY"))
    
    print("\n=== Risk Management ===")
    df = gestion_risque_adaptative(current_capital,tickers_default)


    return df


main_call()