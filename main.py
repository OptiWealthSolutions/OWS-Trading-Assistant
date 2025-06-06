from Macro.commodities_graph import plot_currency_vs_commodities
#from Quant.most_correlated_pairs import compute_most_correlated_pairs
from Quant.risk_management import gestion_risque_adaptative,vol_index,std_index 
from Quant.SL_sizing import atr_index
#from Quant.value_at_risk import calculate_var
from Quant.vol_index_pdf import get_vol_index
from Technical.seasonality import seasonality
from fpdf import FPDF



    
def main_call(ticker):
    plot_currency_vs_commodities(ticker)
        
    atr_index(ticker)
    seasonality(ticker)
    get_vol_index(ticker)
    
main_call("EURUSD=X")