import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --------------- Fonction seasonnality ----------------

def seasonality(ticker):
    data = yf.download(ticker, period='10y', interval='1d')
    df = data[['Close']].dropna()
    df['Return'] = df['Close'].pct_change()
    df['Month'] = df.index.month  
    df['Year'] = df.index.year

    monthly_seasonality = df.groupby('Month')['Return'].mean() * 100
    
    plt.figure(figsize=(10, 5))
    monthly_seasonality.plot(kind='line', color='red')
    monthly_seasonality.plot(kind='bar', color='slategrey')
    plt.title(f"Saisonnalité moyenne mensuelle de {ticker} (10 ans)")
    plt.xlabel("Mois")
    plt.ylabel("Rendement moyen (%)")
    plt.grid(color='grey', linestyle='-', linewidth=0.5)
    plt.xticks(ticks=range(0,12), labels=[
        'Janv', 'Fév', 'Mars', 'Avr', 'Mai', 'Juin',
        'Juil', 'Août', 'Sept', 'Oct', 'Nov', 'Déc'
    ], rotation=45)
    plt.tight_layout()
    
    return plt.show()

seasonality("EURUSD=X")