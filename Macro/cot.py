import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

# Étape 1 : Télécharger les données depuis le site de la CFTC
def download_cot_data():
    """
    Télécharge les données COT (Futures Only) depuis le site officiel de la CFTC.
    Retourne un DataFrame pandas.
    """
    url = "https://www.cftc.gov/files/dea/history/fut_fin_txt_2023.zip"  # Exemple pour l'année 2023
    zip_file = "cot_data.zip"
    extracted_file = "FinFutYY.txt"  # Nom mis à jour pour refléter le fichier extrait

    try:
        # Télécharger le fichier ZIP
        print("Téléchargement des données...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(zip_file, 'wb') as f:
                f.write(response.content)
            print("Téléchargement terminé.")

            # Extraire le fichier
            print("Extraction des données...")
            os.system(f"unzip -o {zip_file}")
            print("Extraction terminée.")

            # Charger les données dans un DataFrame
            cot_data = pd.read_csv(extracted_file, delimiter=',', encoding='utf-8')
            return cot_data
        else:
            print("Erreur lors du téléchargement des données.")
            return None
    except Exception as e:
        print(f"Erreur : {e}")
        return None

# Étape 2 : Filtrer les données pour une paire spécifique
def filter_currency_data(cot_data, currency):
    """
    Filtre les données COT pour une devise spécifique.
    """
    currency_data = cot_data[cot_data['Market_and_Exchange_Names'].str.contains(currency, case=False)]
    if currency_data.empty:
        print(f"Aucune donnée trouvée pour {currency}.")
    return currency_data

# Étape 3 : Visualiser les données
def plot_cot_data(currency_data, currency):
    """
    Trace le positionnement des Non-Commercials et Commercials.
    """
    plt.figure(figsize=(12, 6))

    # Tracer les positions des Non-Commercials
    plt.plot(currency_data['Report_Date_as_MM_DD_YYYY'], 
             currency_data['Noncomm_Positions_Long_All'], 
             label='Non-Commercial Long', color='blue')
    plt.plot(currency_data['Report_Date_as_MM_DD_YYYY'], 
             currency_data['Noncomm_Positions_Short_All'], 
             label='Non-Commercial Short', color='red')

    # Tracer les positions des Commercials
    plt.plot(currency_data['Report_Date_as_MM_DD_YYYY'], 
             currency_data['Comm_Positions_Long_All'], 
             label='Commercial Long', linestyle='--', color='green')
    plt.plot(currency_data['Report_Date_as_MM_DD_YYYY'], 
             currency_data['Comm_Positions_Short_All'], 
             label='Commercial Short', linestyle='--', color='orange')

    # Ajouter des légendes et des titres
    plt.title(f"Commitments of Traders (COT) - {currency}")
    plt.xlabel("Date")
    plt.ylabel("Positions")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Afficher le graphique
    plt.show()

# Étape 4 : Exécution principale
if __name__ == "__main__":
    # Téléchargement et chargement des données
    cot_data = download_cot_data()

    if cot_data is not None:
        # Filtrer les données pour une paire spécifique (ex : USD)
        currency = "USD"  # Remplacez par la paire souhaitée
        currency_data = filter_currency_data(cot_data, currency)

        if not currency_data.empty:
            # Visualiser les données
            plot_cot_data(currency_data, currency)
