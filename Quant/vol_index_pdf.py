import pandas as pd
import yfinance as yf
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# Assure-toi que le dossier existe
output_dir = "vol pdf reports"
os.makedirs(output_dir, exist_ok=True)

# Chemin complet du fichier PDF
ticker = "EURUSD=X"
output_path = os.path.join(output_dir, f"forex_volatility_report_{ticker}.pdf")

with PdfPages(output_path) as pdf:
    # Téléchargement des données
    fx_vol_data_brut = yf.download(ticker, period="6mo", interval="4h")
    fx_df_vol = pd.DataFrame(fx_vol_data_brut)

    # Calcul des rendements log et de la volatilité
    fx_df_vol['Log Returns'] = np.log(fx_df_vol['Close'] / fx_df_vol['Close'].shift(1))
    fx_df_vol['Volatility_20D'] = fx_df_vol['Log Returns'].rolling(window=20).std(ddof=0) * 100
    fx_df_vol.dropna(inplace=True)


    
    current_vol = fx_df_vol['Volatility_20D'].iloc[-1]
    mean_vol = fx_df_vol['Volatility_20D'].mean()

    if current_vol > mean_vol * 1.2:
        comment = f"La volatilité actuelle ({current_vol:.2f}%) est significativement SUPÉRIEURE à la moyenne des 20 dernières périodes ({mean_vol:.2f}%)."
    elif current_vol < mean_vol * 0.8:
        comment = f"La volatilité actuelle ({current_vol:.2f}%) est significativement INFÉRIEURE à la moyenne ({mean_vol:.2f}%)."
    else:
        comment = f"La volatilité actuelle ({current_vol:.2f}%) est proche de la moyenne ({mean_vol:.2f}%)."

    # Création d'une page PDF de text
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(fx_df_vol.index, fx_df_vol['Volatility_20D'], label='Volatility (20d)')
    ax.set_title(f'Volatility (20d) - {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (%)')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    
    fig2 = plt.figure(figsize=(11, 6))
    fig2.text(0.1, 0.5, "Analyse de la volatilité", fontsize=16, weight='bold')
    fig2.text(0.1, 0.3, comment, fontsize=12)
    fig2.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)