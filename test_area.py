import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import ta

currency_commodity_map = {
    "AUD": ["GC=F", "HG=F"],  # Gold, Copper
    "CAD": ["CL=F", "NG=F"],  # Oil, Gas
    "NZD": ["ZC=F", "ZW=F"],  # Agricultural products
    "NOK": ["BZ=F"],          # Brent Oil
    "USD": ["GC=F", "CL=F"],  # Indirectly correlated with many
    "BRL": ["SB=F", "KC=F"],  # Sugar, Coffee
    "MXN": ["CL=F", "ZS=F"],  # Oil, Soybean
    "ZAR": ["GC=F", "PL=F"],  # Gold, Platinum
}

def plot_currency_vs_commodities(ticker, period="6mo", interval="1h"):
    base1 = ticker[:3].upper()
    base2 = ticker[3:6].upper()
    commodities = list(set(currency_commodity_map.get(base1, []) + currency_commodity_map.get(base2, [])))

    if not commodities:
        print(f"No known commodity for {base1} or {base2}")
        return

    # Download data
    forex = yf.download(ticker, period=period, interval=interval)
    forex = forex['Close']
    data = pd.DataFrame(forex)

    for commo in commodities:
        commo_data = yf.download(commo, period=period, interval=interval)
        commo_data = commo_data['Close']
        data = data.join(commo_data, how="inner")

    data.dropna(inplace=True)

    # Prepare to store correlations
    correlations = pd.DataFrame()

    # Display
    for commo in commodities:
        # Compute and print correlation
        forex_returns = data[ticker].pct_change()
        commo_returns = data[commo].pct_change()
        correlation = forex_returns.corr(commo_returns)
        correlations = correlation
       

    return correlations

plot_currency_vs_commodities("EURUSD=X")