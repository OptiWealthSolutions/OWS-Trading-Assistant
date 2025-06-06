import yfinance as yf
import pandas as pd
import numpy as np
#from risk_management import gestion_risque_adaptative


# --------------- Fonction average true range index ----------------
def atr_index(ticker):
    df = yf.download(ticker, period="6mo", interval='4H')
    
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df.dropna(inplace=True)  # Supprime les NaN après calcul ATR
    # Calcul de k
    distance = abs(df['Close'].shift(1) - df['Low'])
    ratio = (distance / df['ATR'].iloc[-1]).dropna()

    k_95 = np.percentile(ratio, 95)
    atr_res = df['ATR'].iloc[-1]
    return atr_res, round(k_95,1)

# --------------- Fonction taille de stop en pips ----------------
def sl_sizing (atr,k, entry_price,direction):
    if direction == "SELL":
        stop_price = entry_price + atr*k
        sl_pips_size = round(abs(entry_price-stop_price)*10000,3)
        return print(round(stop_price,6),sl_pips_size,'pips')
    elif direction =="BUY":
        stop_price = entry_price - atr*k
        sl_pips_size = round(abs(entry_price-stop_price)*10000,3)
        return print(round(stop_price,6),sl_pips_size,'pips')
    else : 
        return "INVALID DIRECTION"

entry_price_test = 1.1303
sl = sl_sizing(atr,k,entry_price_test,"")


# --------------- Fonction pip value for differents pairs ----------------
# def pip_value(pair,lot_size=1,price=None,account_currency="EUR"):
#     base, quote = pair[:3], pair[4:]
#     pip = 0.0001 if "JPY" not in pair else 0.01

#     if account_currency == quote:
#         return lot_size * pip 
#     elif account_currency == base:
#         return lot_size * pip / price 
#     else:

#         conv_pair = f"{quote}{account_currency}=X"
#         rate = yf.download(conv_pair, period="1d", interval="1h")['Close'].iloc[-1]
#         return lot_size * pip * rate
# pip = pip_value("EURUSD",price=1)
# print(pip)

# --------------- Fonction position sizing ----------------
# def position_sizing(pair, entry_price, direction, capital, account_currency="EUR"):
#     atr, k = atr_index(pair + "=X", window=20, duration="6mo")
#     stop_price = entry_price + atr * k if direction == "SELL" else entry_price - atr * k
#     result_risk = gestion_risque_adaptative(capital, pair + "=X")
#     risk_amount = result_risk["Risk"].iloc[0]
#     pip_size = 0.01 if "JPY" in pair else 0.0001
#     sl_pips = abs(entry_price - stop_price) / pip_size
#     pip_val = pip_value(pair, lot_size=1, price=entry_price, account_currency=account_currency)
#     lots = risk_amount / (sl_pips * pip_val)
#     print(f"Taille de position calculée : {lots:.2f} lots (risk: {risk_amount}€, SL: {sl_pips:.2f} pips, pip_val: {pip_val})")
#     return round(lots, 2)

# print(position_sizing("EURUSD",1.3,"SELL",1000))

# def position_vol_sizing(vol,capital,risk):
#     risk_amount = capital * risk
#     position_size = risk_amount/vol
#     return round(position_size,3)