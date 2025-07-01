import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.preprocessing import StandardScaler

def load_raw_data(file_path):
    """Charge les données brutes des prix"""
    return pd.read_csv(file_path, parse_dates=['date'], index_col='date')

def calculate_technical_indicators(df):
    """Calcule les indicateurs techniques pour le DataFrame"""
    # Ajouter tous les indicateurs techniques
    df = add_all_ta_features(
        df, 
        open="open", 
        high="high", 
        low="low", 
        close="close", 
        volume="volume",
        fillna=True
    )
    return df

def calculate_spread(df1, df2):
    """Calcule l'écart entre deux paires de devises"""
    spread = df1['close'] - df2['close']
    return spread

def calculate_z_score(spread):
    """Calcule le z-score de l'écart"""
    mean = spread.rolling(window=20).mean()
    std = spread.rolling(window=20).std()
    z_score = (spread - mean) / std
    return z_score

def create_feature_matrix(df, spread, z_score):
    """Crée la matrice de features complète"""
    features = pd.DataFrame({
        'spread': spread,
        'z_score': z_score,
        'rsi': df['momentum_rsi'],
        'adx': df['trend_adx'],
        'macd': df['trend_macd'],
        'bb_upper': df['volatility_bbh'],
        'bb_lower': df['volatility_bbl'],
        'volume': df['volume']
    })
    
    # Normalisation des features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
