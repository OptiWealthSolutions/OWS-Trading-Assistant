import pandas as pd
import requests
from datetime import datetime
from zipfile import ZipFile
from io import BytesIO
import matplotlib.pyplot as plt

def download_cot_data():
    url = "https://www.cftc.gov/files/dea/history/fut_fin_txt_2025.zip"
    response = requests.get(url)
    with ZipFile(BytesIO(response.content)) as z:
        file_list = z.namelist()
        txt_files = [f for f in file_list if f.endswith('.txt')]
        if not txt_files:
            raise Exception("Aucun fichier texte trouvé dans le ZIP")
        cot_file = txt_files[0]

        with z.open(cot_file) as f:
            df = pd.read_fwf(f, skiprows=1, encoding='latin1')
            df.columns = df.columns.str.strip()  # Nettoyage noms de colonnes
            return df

def filter_currency_data(cot_data, currency):
    possible_colnames = [col for col in cot_data.columns if 'market' in col.lower()]
    if not possible_colnames:
        raise KeyError("Aucune colonne contenant 'market' trouvée.")
    market_col = possible_colnames[0]
    print(f"Colonne utilisée pour filtrer les devises : {market_col}")

    filtered = cot_data[cot_data[market_col].str.contains(currency, case=False, na=False)]
    if filtered.empty:
        print(f"Aucune donnée trouvée pour la devise '{currency}'.")
    return filtered

def compute_position_changes(currency_data):
    currency_data = currency_data.copy()
    currency_data = currency_data.sort_values(by='Report_Date_as_YYYY-MM-DD')

    # Calcul des variations hebdomadaires
    currency_data["Asset_Mgr_Long_Change"] = currency_data["Asset_Mgr_Positions_Long_All"].diff()
    currency_data["Asset_Mgr_Short_Change"] = currency_data["Asset_Mgr_Positions_Short_All"].diff()
    currency_data["Lev_Money_Long_Change"] = currency_data["Lev_Money_Positions_Long_All"].diff()
    currency_data["Lev_Money_Short_Change"] = currency_data["Lev_Money_Positions_Short_All"].diff()

    return currency_data

if __name__ == "__main__":
    cot_data = download_cot_data()
    currency_data = filter_currency_data(cot_data, "JPY")  # Change "JPY" si tu veux une autre devise
    changes_df = compute_position_changes(currency_data)

    print("\nDernières variations hebdomadaires des positions :\n")
    print(changes_df[[
        "Report_Date_as_YYYY-MM-DD",
        "Asset_Mgr_Long_Change",
        "Asset_Mgr_Short_Change",
        "Lev_Money_Long_Change",
        "Lev_Money_Short_Change"
    ]].dropna().tail(10))