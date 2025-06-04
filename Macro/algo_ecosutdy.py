import world_bank_data as wb
import pandas as pd
import matplotlib.pyplot as plt

# 1. Récupérer les données économiques via World Bank Data
def get_economic_data(country_code):
    # Indicateurs spécifiques (PIB, inflation, chômage, etc.)
    indicators = {
        'GDP (current US$)': 'NY.GDP.MKTP.CD',           # PIB
        'Inflation Rate (%)': 'FP.CPI.TOTL.ZG',          # Taux d'inflation
        'Unemployment Rate (%)': 'SL.UEM.TOTL.ZS',       # Taux de chômage
        'Interest Rate (%)': 'FR.INR.LEND',              # Taux d'intérêt
        'GDP Growth (%)': 'NY.GDP.MKTP.KD.ZG'            # Croissance du PIB
    }
    
    # Extraire les données les plus récentes
    data = {indicator: wb.get_series(indicators[indicator], country=country_code, mrv=1).iloc[0]
            for indicator in indicators}
    
    return data

# 2. Comparer deux économies
def compare_economies(country1, country_code1, country2, country_code2):
    data1 = get_economic_data(country_code1)
    data2 = get_economic_data(country_code2)
    
    df = pd.DataFrame([data1, data2], index=[country1, country2])
    
    # 3. Créer une image du tableau
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
    
    # Sauvegarder en format .jpg
    plt.savefig(f'{country1}_{country2}_comparison.jpg', bbox_inches='tight')

    return df

# 4. Tester la comparaison pour USDJPY (United States vs Japan)
compare_economies('United States', 'USA', 'Japan', 'JPN')