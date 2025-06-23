import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression

# Téléchargement des données de l'or (Gold) et de l'USD/JPY
df_gold = yf.download('GC=F', start='2010-01-01', end='2025-10-01')[['Close']].rename(columns={'Close': 'Gold'})
df_usdjpy = yf.download("USDJPY=X", start='2010-01-01', end='2025-10-01')[['Close']].rename(columns={'Close': 'USDJPY'})

df = df_gold.merge(df_usdjpy)
print(df)