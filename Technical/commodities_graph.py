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
    base = ticker[:3]  # Example: "AUD" from "AUDUSD"
    commodities = currency_commodity_map.get(base.upper(), [])
    
    if not commodities:
        print(f"No known commodity for {base}")
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
        
        
plot_currency_vs_commodities("AUDUSD=X")