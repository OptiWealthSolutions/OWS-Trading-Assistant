import requests
import pandas as pd

# Remplacez ces informations par vos propres identifiants IG
API_KEY = "votre_clé_api"
ACCESS_TOKEN = "votre_token_d'accès"
BASE_URL = "https://api.ig.com/gateway/deal"

# Fonction pour obtenir le sentiment des traders retail
def get_retail_sentiment(pair='EURUSD'):
    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'X-IG-API-KEY': API_KEY,
        'Content-Type': 'application/json'
    }
    
    # URL de l'API pour obtenir les informations de positionnement retail sur une paire donnée
    url = f"{BASE_URL}/public/positions/{pair}"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        sentiment_data = data['retailSentiment']
        
        print(f"Sentiment des traders retail pour {pair}:")
        print(f"Position nette acheteuse : {sentiment_data['longs']}")
        print(f"Position nette vendeuse : {sentiment_data['shorts']}")
    else:
        print("Erreur lors de la récupération des données: ", response.status_code)

# Appel de la fonction pour obtenir le positionnement retail sur EUR/USD
get_retail_sentiment('EUR/USD')