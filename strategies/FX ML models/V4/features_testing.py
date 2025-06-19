import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import *
import yfinance as yf
#features testing for commodities correlation : 
data_commo =  yf.Download('GC=F', start='2010-01-01', end='2025-10-01')['Close']
df_commo = pd.Dataframe(data_commo)
df_commo.rename(columns={'Close': 'Gold'}, inplace=True)

data_pairs = yf.download("EURUSD=X", start='2010-01-01', end='2025-10-01')['Close']
df_pairs = pd.DataFrame(data_pairs)

df_commo = df_commo.dropna(inplace=True)
df_pairs = df_pairs.dropna(inplace=True)

# Merge the two DataFrames on the index
df_merged = pd.merge(df_commo, df_pairs, left_index=True, right_index=True, how='inner')
df_merged.rename(columns={'Close_x': 'Gold', 'Close_y': 'EURUSD'}, inplace=True)

# Calculate the correlation
correlation = df_merged['Gold'].corr(df_merged['EURUSD'])
print(f"Correlation between Gold and EURUSD: {correlation}")