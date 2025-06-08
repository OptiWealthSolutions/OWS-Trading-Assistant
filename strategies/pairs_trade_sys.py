import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Liste des paires forex fortement corrélées
forex_pairs_correlated = [
    ("AUDJPY=X", "NZDJPY=X"),
    ("AUDUSD=X", "NZDUSD=X"),
    ("GBPJPY=X", "EURJPY=X"),
    ("GBPUSD=X", "EURUSD=X"),
    ("USDCHF=X", "USDJPY=X"),
    ("EURUSD=X", "GBPUSD=X")
]

def data_loader(ticker_1, ticker_2, duration):
    data_1 = yf.download(ticker_1, period=duration, interval="1d")[["Close"]]
    data_2 = yf.download(ticker_2, period=duration, interval="1d")[["Close"]]
    data_1.rename(columns={"Close": f"{ticker_1}_Close"}, inplace=True)
    data_2.rename(columns={"Close": f"{ticker_2}_Close"}, inplace=True)
    df = pd.concat([data_1, data_2], axis=1).dropna()
    return df

def test_adf(series: pd.Series):
    result = adfuller(series)
    return result[1]

def engle_granger_test(y_serie: pd.Series, x_serie: pd.Series):
    df_temp = pd.concat([y_serie, x_serie], axis=1).dropna()
    y_serie_clean = df_temp.iloc[:, 0]
    x_serie_clean = df_temp.iloc[:, 1]
    x = sm.add_constant(x_serie_clean)
    model = sm.OLS(y_serie_clean, x).fit()
    residuals = model.resid
    p_value = test_adf(residuals)
    r_squared = model.rsquared
    params = model.params
    return residuals, p_value, r_squared, params

def pairs_trading_summary(ticker1: str, ticker2: str, duration: str = "1y", save_path=None):
    df = data_loader(ticker1, ticker2, duration)
    spread, p_value, r_squared, params = engle_granger_test(df[f"{ticker1}_Close"], df[f"{ticker2}_Close"])
    
    cointegration = p_value < 0.05
    interpretation = f"Analyse de la paire {ticker1} / {ticker2} sur {duration}:\n"
    interpretation += f"- Co-intégration détectée : {'Oui' if cointegration else 'Non'} (p-value={p_value:.4f})\n"
    interpretation += f"- R² de la régression : {r_squared:.4f}\n"
    interpretation += f"- Coefficients : alpha (intercept) = {params[0]:.4f}, beta = {params[1]:.4f}\n"
    
    zscore = (spread - spread.mean()) / spread.std()
    zscore_final = zscore.iloc[-1]
    interpretation += f"- Niveau actuel du z-score : {zscore_final:.2f} (distance à la moyenne du spread)\n"
    
    if abs(zscore_final) > 2:
        interpretation += "- Le spread est significativement éloigné de sa moyenne (potentiel signal de trading).\n"
    else:
        interpretation += "- Le spread est proche de sa moyenne (pas de signal fort).\n"
    
    plt.figure(figsize=(12,6))
    plt.plot(spread.tail(90))
    plt.axhline(spread.mean(), color='black', linestyle='--', label='Moyenne du spread')
    plt.axhline(spread.std(), color='red', linestyle='--', label='STD du spread')
    plt.title(f'Spread entre {ticker1} et {ticker2} (90 derniers jours)')
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return interpretation

if __name__ == "__main__":
    ticker_default = "EURUSD=X"

    for ticker1, ticker2 in forex_pairs:
        if ticker_default in (ticker1, ticker2):
            other_ticker = ticker2 if ticker1 == ticker_default else ticker1
            result = pairs_trading_summary(ticker_default, other_ticker)
            print(result)
            print("=" * 80)