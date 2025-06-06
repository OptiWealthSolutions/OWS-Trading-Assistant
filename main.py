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
    # Initialisation du contenu HTML
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
    html_content += f"<p>{correlation}</p>"

    # === Volatility Index ===
    html_content += "<h2>Volatility Index</h2>"
    get_vol_index(tickers_default)
    html_content += "<p>(Indicateurs de volatilité calculés)</p>"

    # === Seasonality ===
    html_content += "<h2>Seasonality</h2>"
    seasonality(tickers_default)
    html_content += "<p>(Saisonnalité analysée)</p>"

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
        result = pairs_trading_summary(ticker, other)
        html_content += f"<h3>Analyse {ticker} / {other}</h3>"
        html_content += f"<pre>{result['interpretation']}</pre>"
        # Save spread graph image generated inside pairs_trading_summary
        # Here you can adjust to save plots with unique names per pair if needed
        # For example:
        # filename = f"spread_{ticker}_{other}.png"
        # plt.savefig(filename)
        # html_content += f'<img src="{filename}" alt="Spread Graph">'
        # But since pairs_trading_summary shows plot, consider modifying it to save plot instead

    # === Technical Signals ===
    html_content += "<h2>Technical Signals</h2>"
    # You can add summaries of sma_crossing and calculate_adx here
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

    # === Spread Graph (example for first pair) ===
    # Generate spread plot image and embed
    if selected_pairs:
        first_pair = selected_pairs[0]
        ticker1 = ticker
        ticker2 = first_pair[1] if first_pair[0] == ticker else first_pair[0]
        df = data_loader(ticker1, ticker2, "1y")
        spread = df[f"{ticker1}_Close"] - df[f"{ticker2}_Close"]  # simple spread example
        plt.figure(figsize=(12, 6))
        plt.plot(spread.tail(90))
        plt.axhline(spread.mean(), color='black', linestyle='--', label='Moyenne du spread')
        plt.axhline(spread.std(), color='red', linestyle='--', label='STD du spread')
        plt.title(f'Spread entre {ticker1} et {ticker2} (90 derniers jours)')
        plt.legend()
        plt.grid()
        image_path = "spread_example.png"
        plt.savefig(image_path)
        plt.close()
        html_content += f'<h2>Spread Graph for {ticker1} / {ticker2}</h2>'
        html_content += f'<img src="{image_path}" alt="Spread Graph">'

    html_content += "</body></html>"

    # Écriture du fichier HTML
    report_file = "rapport_trading.html"
    with open(report_file, "w") as f:
        f.write(html_content)

    print(f"Rapport HTML généré : {os.path.abspath(report_file)}")

    return df_risk  # Optionnel, si tu veux récupérer le DataFrame


# Exemple d’appel dans ton script principal
main_call("EURUSD=X")