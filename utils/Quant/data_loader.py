import yfinance as yf
import pandas as pd

def data_loader(ticker_1, ticker_2, duration):
    # Téléchargement des données
    data_1 = yf.download(ticker_1, period=duration, interval="1d")[["Close"]]
    data_2 = yf.download(ticker_2, period=duration, interval="1d")[["Close"]]
    data_1.rename(columns={"Close": f"{ticker_1}_Close"}, inplace=True)
    data_2.rename(columns={"Close": f"{ticker_2}_Close"}, inplace=True)
    # axis = 1 permet de concatener en prenant la date comme index et donc  de mettre les df horizontalement
    main_df = pd.concat([data_1, data_2], axis=1)

    main_df[f"{ticker_1}_Return"] = main_df[f"{ticker_1}_Close"].pct_change()
    main_df[f"{ticker_2}_Return"] = main_df[f"{ticker_2}_Close"].pct_change()

    return main_df

data = data_loader("AAPL",'MSFT', "3mo")
data

