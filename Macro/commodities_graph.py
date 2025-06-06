import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import sys

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
    correlations = {}

    # Display
    for commo in commodities:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        ax1.plot(data[ticker], label=ticker, color="blue")
        ax2.plot(data[commo], label=commo, color="orange", alpha=0.6)

        plt.title(f"{ticker} vs {commo}")
        ax1.set_ylabel(f"{ticker}", color="blue")
        ax2.set_ylabel(f"{commo}", color="orange")
        plt.legend()
        plt.grid()
        plt.show()

        # Compute and print correlation
        forex_returns = data[ticker].pct_change()
        commo_returns = data[commo].pct_change()
        correlation = forex_returns.corr(commo_returns)
        correlations[commo] = correlation
       

    return  print(f"Correlation between {ticker} and {commo}: {correlation:.4f}")

plot_currency_vs_commodities("EURUSD=X")