import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import pandas as pd
from settings import *
from fpdf import FPDF
import smtplib
import matplotlib.pyplot as plt
from email.message import EmailMessage
# Quant
from utils.Quant.risk_management import gestion_risque_adaptative
from utils.Quant.SL_sizing import atr_index
from utils.Quant.SL_sizing import sl_sizing
from utils.Quant.vol_index import get_vol_index
from utils.Quant.pairs_trade_sys import pairs_trading_summary, data_loader
# Macro
from utils.Technical.seasonality import seasonality
from utils.Technical.sma_crossing import sma_crossing
from utils.Technical.trend_following import calculate_adx
# Technical (corrige le nom du fichier ici si nécessaire !)
from utils.Macro.commodities_graph import plot_currency_vs_commodities

# PDF generator (si c’est dans main.py tu n’as pas besoin de l’importer)


# --------------- Main call fonction----------------


def main_call(ticker):
    html_content = """
    <html>
    <head>
        <title>Rapport Trading</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h2 { color: #2F4F4F; }
            table { border-collapse: collapse; width: 80%; margin-bottom: 40px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; margin-bottom: 40px; }
        </style>
    </head>
    <body>
    """

    # === Commodities Correlation ===
    html_content += "<h2>Commodities Correlation</h2>"
    correlation = plot_currency_vs_commodities(tickers_default)
    plt.savefig('commodities.png')
    plt.close()
    html_content += f"<p>{correlation}</p>"
    html_content += '<img src="commodities.png" alt="Commodities Correlation">'

    # === Volatility Index ===
    html_content += "<h2>Volatility Index</h2>"
    get_vol_index(tickers_default)
    plt.savefig('vol_index.png')
    plt.close()
    html_content += '<img src="vol_index.png" alt="Volatility Index">'

    # === Seasonality ===
    html_content += "<h2>Seasonality</h2>"
    seasonality(tickers_default)
    plt.savefig('seasonality.png')
    plt.close()
    html_content += '<img src="seasonality.png" alt="Seasonality">'

    # === Pairs Trading ===
    html_content += "<h2>Pairs Trading</h2>"
    forex_pairs = [
        ("EURUSD=X", "GBPUSD=X"),
        ("AUDUSD=X", "NZDUSD=X"),
        ("EURUSD=X", "USDCHF=X"),
        ("USDCAD=X", "CL=F")
    ]
    selected_pairs = [pair for pair in forex_pairs if ticker in pair]
    for pair in selected_pairs:
        other = pair[1] if pair[0] == ticker else pair[0]
        result = pairs_trading_summary(ticker, other, save_path=f"spread_{ticker}_{other}.png")
        html_content += f"<h3>Analyse {ticker} / {other}</h3>"
        html_content += f"<pre>{result}</pre>"
        html_content += f'<img src="spread_{ticker}_{other}.png" alt="Spread Graph {ticker}-{other}">'

    # === Technical Signals ===
    html_content += "<h2>Technical Signals</h2>"
    html_content += "<p>SMA Crossing and ADX calculated (details non affichés)</p>"

    # === Stop Loss Sizing ===
    html_content += "<h2>Stop Loss Sizing</h2>"
    atr_result = atr_index(tickers_default)
    html_content += f"<p>ATR: {atr_result[0]:.6f}, K: {atr_result[1]}</p>"
    sl_result = sl_sizing(atr_result[0], atr_result[1], entry_price_ticker_default, "SELL")
    html_content += f"<p>Stop Loss Sizing: {sl_result}</p>"

    # === Risk Management ===
    html_content += "<h2>Risk Management</h2>"
    df_risk = gestion_risque_adaptative(current_capital, tickers_default)
    html_content += df_risk.to_html()

    html_content += "</body></html>"

    report_file = "rapport_trading.html"
    with open(report_file, "w") as f:
        f.write(html_content)

    print(f"Rapport HTML généré : {os.path.abspath(report_file)}")

    return df_risk


# Exemple d’appel dans ton script principal
main_call(tickers_default)