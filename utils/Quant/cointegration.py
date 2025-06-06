from statsmodels.tsa.stattools import adfuller
import pandas as pd
from data_loader import data_loader
import statsmodels.api as sm
import matplotlib.pyplot as plt

ticker1 = "EURUSD=X"
ticker2 = "EURJPY=X"

def test_adf(series: pd.Series):
    result = adfuller(series)
    print(f"Statistique ADF : {result[0]}")
    print(f"p-value         : {result[1]}")
    print("Valeurs critiques :")
    for key, value in result[4].items():
        print(f"   {key} : {value}")
    if result[1] < 0.05:
        print("=> La série est stationnaire (rejette H0)")
    else:
        print("=> La série n'est PAS stationnaire (ne rejette pas H0)")
    return result[1]

def engle_granger_test(y_serie: pd.Series, x_serie: pd.Series):
    # Concaténer les deux séries pour supprimer les NaN communs
    df_temp = pd.concat([y_serie, x_serie], axis=1).dropna()
    y_serie_clean = df_temp.iloc[:, 0]
    x_serie_clean = df_temp.iloc[:, 1]

    x = sm.add_constant(x_serie_clean)
    model = sm.OLS(y_serie_clean, x).fit()
    residuals = model.resid
    hedge_ratio = model.params[1]
    intercept = model.params[0]

    print(model.summary())

    # Test ADF sur les résidus
    p_value = test_adf(residuals)

    if p_value < 0.05:
        print("Coïntégration détectée (résidus stationnaires)")
    else:
        print("Pas de coïntégration (résidus non stationnaires)")

    # Visualiser le spread
    spread = residuals
    plt.figure(figsize=(12, 6))
    #la graphique nous donne que les 90 denriers jours de données
    plt.plot(spread.tail(90))
    plt.axhline(spread.mean(), color='black', linestyle='--', label='Moyenne du spread')
    plt.axhline(spread.std(), color='red', linestyle='--', label='STD du spread')
    plt.title('Spread entre les deux actifs')
    plt.legend()
    plt.grid()
    plt.show()

    return  spread
#les données sont puisées sur 10ans 
df = data_loader(ticker1, ticker2, "10y")
spread= engle_granger_test(df[f"{ticker1}_Close"], df[f"{ticker2}_Close"])