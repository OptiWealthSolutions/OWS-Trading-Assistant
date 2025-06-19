import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import AverageTrueRange
from fx_strategy_V4 import get_interest_rate_difference
from sklearn import *
import warnings
warnings.filterwarnings("ignore")



def prepare_dataset_signal(spread, zscore, pair1_close, gold_price, adx, macro_data=None, seuil=1):
    if isinstance(pair1_close, pd.DataFrame):
        pair1_close = pair1_close.iloc[:, 0]
    if isinstance(gold_price, pd.DataFrame):
        gold_price = gold_price.iloc[:, 0]
        
    #liste des indicateurs calculés dans "fx_strategy_V3;py"
    pair1_close = pair1_close.reindex(spread.index)
    gold_price = gold_price.reindex(spread.index)
    zscore = zscore.reindex(spread.index)
    adx = adx.reindex(spread.index)
    
    #liste des indicateurs techniques utilisés dans la regression logistique mutlinomiale
    rsi_pair1 = RSIIndicator(close=pair1_close, window=14).rsi()
    sma_20 = SMAIndicator(close=pair1_close, window=20).sma_indicator()
    ema_20 = EMAIndicator(close=pair1_close, window=20).ema_indicator()
    macd = MACD(close=pair1_close).macd_diff()
    bb_bands = BollingerBands(close=pair1_close, window=20)
    bb_bbh = bb_bands.bollinger_hband()
    bb_bbl = bb_bands.bollinger_lband()
    roc = ROCIndicator(close=pair1_close, window=12).roc()
    atr = AverageTrueRange(high=pair1_close, low=pair1_close, close=pair1_close).average_true_range()
    #indicateur macro utilisé dans la regression multinomiale
    rate_diff = get_interest_rate_difference(pair1_close.name if hasattr(pair1_close, 'name') else "")

    df = pd.DataFrame({
        'spread': spread,
        'z_score': zscore,
        'z_score_lag1': zscore.shift(1),
        'vol_spread': spread.rolling(30).std(),
        'rsi_pair1': rsi_pair1,
        'adx': adx,
        'sma_20': sma_20,
        'ema_20': ema_20,
        'macd': macd,
        'bb_high': bb_bbh,
        'bb_low': bb_bbl,
        'roc': roc,
        'atr': atr,
        'rate_diff': get_interest_rate_difference(pair1_close.name if hasattr(pair1_close, 'name') else "")
    })

    if macro_data is not None:
        # Merge macro data on index (dates), align on dates
        macro_data_reindexed = macro_data.reindex(df.index).ffill().bfill()
        df = pd.concat([df, macro_data_reindexed], axis=1)

    df.dropna(inplace=True)
    df = df.astype(float)

    # Nouvelle étiquette : -1 → 0 (SELL), 0 → 1 (WAIT), 1 → 2 (BUY)
    df['target'] = 1  # WAIT par défaut
    df.loc[df['z_score'] > seuil, 'target'] = 0  # SELL
    df.loc[df['z_score'] < -seuil, 'target'] = 2  # BUY

    feature_cols = ['z_score', 'z_score_lag1', 'vol_spread', 'rsi_pair1', 'adx',
            'sma_20', 'ema_20', 'macd', 'bb_high', 'bb_low', 'roc', 'atr', 'rate_diff']
    if macro_data is not None:
        feature_cols += list(macro_data.columns)

    X = df[feature_cols]
    y = df['target']

    return X, y