import pandas as pd
import yfinance as yf
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ticker = "EURUSD=X"

forex_tickers = [
    "EURUSD=X",  # Euro / US Dollar
    "USDJPY=X",  # US Dollar / Japanese Yen
    "GBPUSD=X",  # British Pound / US Dollar
    "USDCHF=X",  # US Dollar / Swiss Franc
    "AUDUSD=X",  # Australian Dollar / US Dollar
    "NZDUSD=X",  # New Zealand Dollar / US Dollar
    "USDCAD=X",  # US Dollar / Canadian Dollar

    "EURGBP=X",  # Euro / British Pound
    "EURJPY=X",  # Euro / Japanese Yen
    "GBPJPY=X",  # British Pound / Japanese Yen
    "AUDJPY=X",  # Australian Dollar / Japanese Yen
    "NZDJPY=X",  # New Zealand Dollar / Japanese Yen

    "EURAUD=X",  # Euro / Australian Dollar
    "GBPAUD=X",  # British Pound / Australian Dollar
    "EURCHF=X",  # Euro / Swiss Franc
]


import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

forex_tickers = [
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X",
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "NZDJPY=X", "EURAUD=X", "GBPAUD=X", "EURCHF=X"
]
with PdfPages("forex_volatility_report.pdf") as pdf:
    for ticker in forex_tickers:
        fx_vol_data_brut = yf.download(ticker, period="6mo", interval="1d")
        fx_df_vol = pd.DataFrame(fx_vol_data_brut)
        
        fx_df_vol['Log Returns'] = np.log(fx_df_vol['Close'] / fx_df_vol['Close'].shift(1))
        fx_df_vol['Volatility'] = fx_df_vol['Log Returns'].rolling(window=20).std(ddof=0) * 100
        fx_df_vol.dropna(inplace=True)

        # Crée une figure pour chaque ticker
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(fx_df_vol.index, fx_df_vol['Volatility'], label='Volatility (20d)')
        ax.set_title(f'Volatility (20d) - {ticker}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility (%)')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        pdf.savefig(fig)  # sauvegarde explicite de la figure
        plt.close(fig) 

