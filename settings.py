
# --------------- Tickers' lists ----------------
tickers = [
    "EURUSD=X",  # Euro / US Dollar
    "USDJPY=X",  # US Dollar / Japanese Yen
    "GBPUSD=X",  # British Pound / US Dollar
    "USDCHF=X",  # US Dollar / Swiss Franc
    "AUDUSD=X",  # Australian Dollar / US Dollar
    "NZDUSD=X",  # New Zealand Dollar / US Dollar
    "USDCAD=X",  # US Dollar / Canadian Dollar

    "EURGBP=X",  # Euro / British Pound
    "EURJPY=X",  # Euro / Japanese Yen
    "GBPJPY=X",  # British Pound / Japanese Yen
    "AUDJPY=X",  # Australian Dollar / Japanese Yen
    "NZDJPY=X",  # New Zealand Dollar / Japanese Yen

    "EURAUD=X",  # Euro / Australian Dollar
    "GBPAUD=X",  # British Pound / Australian Dollar
    "EURCHF=X",  # Euro / Swiss Franc
]

stocks_tickers = []

tickers_default = "EURUSD=X"
entry_price_ticker_default = 1.14468

# mapping devise/commodité
currency_commodity_map = {
    "AUD": ["GC=F", "HG=F"],  # Or, Cuivre
    "CAD": ["CL=F", "NG=F"],  # Pétrole, gaz
    "NZD": ["ZC=F", "ZW=F"],  # Produits agricoles
    "NOK": ["BZ=F"],          # Pétrole Brent
    "USD": ["GC=F", "CL=F"],  # Indirectement corrélé à beaucoup
    "BRL": ["SB=F", "KC=F"],  # Sucre, café
    "MXN": ["CL=F", "ZS=F"],  # Pétrole, soja
    "ZAR": ["GC=F", "PL=F"],  # Or, platine
}


forex_pairs_correlated = [
    ("AUDJPY=X", "NZDJPY=X"),
    ("AUDUSD=X", "NZDUSD=X"),
    ("GBPJPY=X", "EURJPY=X"),
    ("GBPUSD=X", "EURUSD=X"),
    ("USDCHF=X", "USDJPY=X"),
    ("EURUSD=X", "GBPUSD=X")
]

# --------------- Testing values ----------------
current_capital = 909
max_risk = 0.02
min_risk = 0

# --------------- Constants ----------------



# --------------- API KEYS ----------------


# --------------- Importants links ----------------



# --------------- to do ----------------