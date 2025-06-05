import requests
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
from io import BytesIO

# Étape 1 : Télécharger et extraire les données COT depuis la CFTC
def download_cot_data():
    """
    Télécharge et extrait les données COT (Futures Only) depuis le site officiel de la CFTC.
    Retourne un DataFrame pandas.
    """
    url = "https://www.cftc.gov/files/dea/history/fut_fin_txt_2023.zip"  # Modifier pour l'année en cours si besoin

    try:
        print("Téléchargement des données...")
        response = requests.get(url)
        if response.status_code == 200:
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                # Trouve automatiquement le fichier .txt dans l'archive
                file_name = [f for f in z.namelist() if f.endswith('.txt')][0]
                print(f"Fichier extrait : {file_name}")
                with z.open(file_name) as f:
                    # Lecture dans DataFrame
                    cot_data = pd.read_csv(f, delimiter=',', encoding='utf-8')
                    return cot_data
        else:
            print(f"Erreur {response.status_code} lors du téléchargement.")
            return None
    except Exception as e:
        print(f"Erreur : {e}")
        return None

# Étape 2 : Filtrer les données pour une devise spécifique
def filter_currency_data(cot_data, currency):
    """
    Filtre les données COT pour une devise spécifique.
    """
    filtered = cot_data[cot_data['Market_and_Exchange_Names'].str.contains(currency, case=False, na=False)]
    if filtered.empty:
        print(f"Aucune donnée trouvée pour la devise '{currency}'.")
    return filtered

# Étape 3 : Tracer les données COT
def plot_cot_data(currency_data, currency):
    """
    Affiche un graphique des positions COT pour une devise spécifique.
    Compatible avec le format Disaggregated COT.
    """
    print("Colonnes disponibles :", currency_data.columns.tolist())

    # Identifier la colonne de date
    date_col = next((col for col in currency_data.columns if "date" in col.lower()), None)
    if date_col is None:
        raise KeyError("Aucune colonne de date trouvée.")
    currency_data[date_col] = pd.to_datetime(currency_data[date_col], errors='coerce')

    # Colonnes utilisées (Asset Managers et Leverage Money)
    asset_mgr_long = "Asset_Mgr_Positions_Long_All"
    asset_mgr_short = "Asset_Mgr_Positions_Short_All"
    lev_money_long = "Lev_Money_Positions_Long_All"
    lev_money_short = "Lev_Money_Positions_Short_All"

    # Vérification existence colonnes
    for col in [asset_mgr_long, asset_mgr_short, lev_money_long, lev_money_short]:
        if col not in currency_data.columns:
            raise KeyError(f"Colonne manquante : {col}")

    # Tracé
    plt.figure(figsize=(14, 6))
    plt.plot(currency_data[date_col], currency_data[asset_mgr_long], label='Asset Manager Long', color='blue')
    plt.plot(currency_data[date_col], currency_data[asset_mgr_short], label='Asset Manager Short', color='red')
    plt.plot(currency_data[date_col], currency_data[lev_money_long], label='Leverage Money Long', linestyle='--', color='green')
    plt.plot(currency_data[date_col], currency_data[lev_money_short], label='Leverage Money Short', linestyle='--', color='orange')

    plt.title(f"Disaggregated COT - {currency}")
    plt.xlabel("Date")
    plt.ylabel("Nombre de contrats")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# Étape 4 : Point d'entrée du script
if __name__ == "__main__":
    cot_data = download_cot_data()
    if cot_data is not None:
        currency = "USD"  # Exemples : "EUR", "JPY", "CAD"
        currency_data = filter_currency_data(cot_data, currency)
        if not currency_data.empty:
            plot_cot_data(currency_data, currency)