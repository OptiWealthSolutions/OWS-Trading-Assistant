import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import pandas as pd
from settings import *
from fpdf import FPDF
# Quant
from utils.Quant.risk_assessement.risk_management import gestion_risque_adaptative
from utils.Quant.risk_assessement.stop_sizing import atr_index, sl_sizing
from utils.Quant.vol_index import get_vol_index
from strategies.pairs_trade_sys import pairs_trading_summary
# Macro
from utils.Macro.seasonality import seasonality
from utils.Technical.indicators_signals import sma_crossing
from utils.Technical.trend_following import calculate_adx
# Technical (corrige le nom du fichier ici si nécessaire !)
from utils.Macro.commodities_graph import plot_currency_vs_commodities
# PDF generator (si c’est dans main.py tu n’as pas besoin de l’importer
from settings import *

