# settings.py

tickers = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X",
    "EURJPY=X",
    "EURGBP=X",
    "EURCHF=X",
    "GBPJPY=X",
    "AUDJPY=X",
    "CHFJPY=X",
    "USDMXN=X",
    "USDZAR=X"
]

correlation_matrix = {
    ("EURUSD=X", "GBPUSD=X"): 0.85,
    ("EURUSD=X", "USDJPY=X"): -0.40,
    ("GBPUSD=X", "USDJPY=X"): -0.35,
    ("AUDUSD=X", "NZDUSD=X"): 0.90,
    ("EURUSD=X", "EURJPY=X"): 0.75,
    ("USDCHF=X", "USDJPY=X"): 0.80,
    ("EURUSD=X", "USDCAD=X"): -0.50,
    ("GBPUSD=X", "EURGBP=X"): -0.88,
    ("USDCHF=X", "EURCHF=X"): 0.95,
    # Complète selon tes analyses
}

commodity_mapping = {
    "USD": "CL=F",   # Crude Oil
    "EUR": "GC=F",   # Gold
    "GBP": "GC=F",
    "JPY": "SI=F",   # Silver
    "AUD": "GC=F",  # Gold spot
    "CAD": "CL=F",
    "CHF": "XAG=X",  # Silver spot
    "NZD": "GC=F",
    "MXN": "CL=F",   # Par défaut pétrole
    "ZAR": "GC=F"   # Par défaut or
}