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
# PDF generator (si c’est dans main.py tu n’as pas besoin de l’importer
from settings import *


# --------------- Main call fonction----------------

def main_call():
    print("=== Commodities Correlation ===")
    print(plot_currency_vs_commodities(tickers_default))
    

    print("\n=== Volatility Index ===")
    get_vol_index(tickers_default)

    print("\n=== Seasonality ===")
    seasonality(tickers_default)
    
    print("\n=== Technics Signals ===")
    sma_crossing(tickers_default)
    calculate_adx(tickers_default)
    
    print("\n=== Pairs Trading Summary ===")
    for ticker1, ticker2 in forex_pairs_correlated :
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



def generate_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Rapport de Marché - {tickers_default}", ln=True)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Corrélation avec les Matières Premières :", ln=True)
    correlation = plot_currency_vs_commodities(tickers_default)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 10, str(correlation).encode("ascii", "ignore").decode("ascii"))

    pdf.cell(0, 10, "Indice de Volatilité :", ln=True)
    vol = get_vol_index(tickers_default)
    pdf.multi_cell(0, 10, str(vol).encode("ascii", "ignore").decode("ascii"))

    pdf.cell(0, 10, "Saisonnalité :", ln=True)
    seas = seasonality(tickers_default)
    pdf.multi_cell(0, 10, str(seas).encode("ascii", "ignore").decode("ascii"))

    pdf.cell(0, 10, "Signaux Techniques :", ln=True)
    signal1 = sma_crossing(tickers_default)
    signal2 = calculate_adx(tickers_default)
    pdf.multi_cell(0, 10, f"SMA: {signal1}, ADX: {signal2}".encode("ascii", "ignore").decode("ascii"))

    pdf.cell(0, 10, "Pairs Trading :", ln=True)
    for ticker1, ticker2 in forex_pairs_correlated:
        if tickers_default in (ticker1, ticker2):
            other_ticker = ticker2 if ticker1 == tickers_default else ticker1
            summary = pairs_trading_summary(tickers_default, other_ticker)
            pdf.multi_cell(0, 10, summary.encode("ascii", "ignore").decode("ascii"))
            pdf.cell(0, 5, "-"*50, ln=True)

    pdf.cell(0, 10, "Stop Loss Sizing :", ln=True)
    atr, k = atr_index(tickers_default)
    sl = sl_sizing(atr, k, entry_price_ticker_default, "BUY")
    pdf.multi_cell(0, 10, f"ATR: {atr}, K: {k}, Stop Loss: {sl}".encode("ascii", "ignore").decode("ascii"))

    pdf.cell(0, 10, "Gestion du Risque :", ln=True)
    df_risk = gestion_risque_adaptative(current_capital, tickers_default)
    pdf.multi_cell(0, 10, df_risk.to_string().encode("ascii", "ignore").decode("ascii"))

    pdf.output(f"rapport_marche_{tickers_default}.pdf")

main_call()
# Pour générer le rapport PDF, décommentez la ligne suivante :
#generate_report()