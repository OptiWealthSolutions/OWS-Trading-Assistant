import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# mise en place des features et d'n processus afin de savoir si une features est utile ou non
# étape 1 : test de correlation de la features avec la target
# étape 2 : test de la solidité de la features avec la target
# étape 3 : test de la solidité de la features avec les autres features
# étape 4 : verification de l'importance de chaque features selon le modele de machine learning
#TEST A EFFECTUER :
        # real_rate_diff
        # oi_change
        # entropie_prix
        # skew_returns_14
        # macro_regime
        # sentiment_boj
        # coint_stability
        # P-val test Engle-Granger glissant

def calculate_real_rate_diff(pair):
    """Calcule l'écart de taux d'intérêt réels"""
    rates = {
        "USD": 4.50,
        "EUR": 2.15,
        "GBP": 4.25,
        "JPY": 0.50,
        "CHF": 0.25,
        "CAD": 2.75,
        "AUD": 3.85,
        "NZD": 3.25,
    }
    base = pair[:3].upper()
    quote = pair[3:6].upper()
    return rates[base] - rates[quote]

def calculate_oi_change(data, window=14):
    """Calcule le changement de l'Open Interest"""
    data['oi_change'] = data['Open'].rolling(window=window).std()
    return data['oi_change']

def calculate_entropy(data, window=14):
    """Calcule l'entropie des prix"""
    entropy_values = []
    for i in range(len(data)):
        if i < window:
            entropy_values.append(np.nan)
        else:
            prices = data['Close'].iloc[i-window:i]
            hist, _ = np.histogram(prices, bins='auto', density=True)
            entropy_values.append(entropy(hist))
    return pd.Series(entropy_values, index=data.index)

def calculate_skew(data, window=14):
    """Calcule le skew des rendements"""
    returns = data['Close'].pct_change()
    return returns.rolling(window=window).skew()

def calculate_macro_regime(data, pair):
    """Calcule le régime macroéconomique basé sur les taux d'intérêt"""
    fred = Fred()
    try:
        # Récupérer les taux d'intérêt des deux pays
        base = pair[:3].upper()
        quote = pair[3:6].upper()
        
        if base == 'USD':
            us_rates = fred.get_series('FEDFUNDS')
        else:
            us_rates = pd.Series([0])
        
        if quote == 'USD':
            us_rates = pd.Series([0])
        
        # Simplification : utilisation des taux US comme proxy
        data['macro_regime'] = us_rates.pct_change().rolling(window=14).mean()
    except:
        data['macro_regime'] = pd.Series([0], index=data.index)
    return data['macro_regime']

def calculate_engle_granger_pval(data, pair):
    """Calcule la p-value du test Engle-Granger"""
    base = pair[:3].upper()
    quote = pair[3:6].upper()
    
    if base == 'USD' or quote == 'USD':
        # Utiliser l'or comme proxy pour USD
        gold = yf.download('GC=F', start=data.index[0], end=data.index[-1])['Close']
        returns = data['Close'].pct_change()
        gold_returns = gold.pct_change()
        
        df = pd.concat([returns, gold_returns], axis=1).dropna()
        x = sm.add_constant(df.iloc[:, 1])
        model = sm.OLS(df.iloc[:, 0], x).fit()
        residuals = model.resid
        p_value = adfuller(residuals)[1]
        
        return pd.Series([p_value], index=data.index)
    return pd.Series([0], index=data.index)

def calculate_wavelet_peak(data, window=14):
    """Calcule les pics des ondelettes"""
    returns = data['Close'].pct_change()
    peaks = returns.rolling(window=window).max()
    return peaks

# Fonction pour tester une feature individuellement
def test_individual_feature(ticker_target, feature_name, feature_func):
    # Téléchargement des données
    data = yf.download(ticker_target, start="2010-01-01", end="2023-06-23")
    data = data.dropna()
    
    # Calcul de la feature
    if feature_name == 'real_rate_diff':
        data[feature_name] = calculate_real_rate_diff(ticker_target)
    else:
        data[feature_name] = feature_func(data, ticker_target)
    
    # Calcul des rendements
    data['returns_target'] = data['Close'].pct_change()
    data = data.dropna()
    
    # Préparation des données
    features = data[[feature_name]]
    target = data['returns_target']
    
    # Séparation train/test
    train_size = int(len(features) * 0.8)
    X_train, X_test = features[:train_size], features[train_size:]
    y_train, y_test = target[:train_size], target[train_size:]
    
    # Entraînement du modèle
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(X_train, y_train)
    
    # Prédiction et évaluation
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    
    # Création du graphique
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Valeurs réelles', color='blue')
    plt.plot(y_test.index, y_pred, label='Valeurs prédites', color='red', linestyle='--')
    plt.title(f'Performance de {feature_name} pour {ticker_target}\nR² = {r2:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Rendements')
    plt.legend()
    plt.show()
    
    print(f"\nStatistiques pour {feature_name}:")
    print(f"Corrélation: {data[feature_name].corr(data['returns_target']):.4f}")
    print(f"R² du modèle: {r2:.4f}")
    
    return r2

# Fonction principale pour tester toutes les features
def features_testing(ticker_target):
    print(f"\nAnalyse de {ticker_target} avec différentes features")
    
    # Liste des features à tester
    features_list = [
        ('real_rate_diff', calculate_real_rate_diff),
        ('oi_change', calculate_oi_change),
        ('entropie_prix', calculate_entropy),
        ('skew_returns_14', calculate_skew),
        ('macro_regime', calculate_macro_regime),
        ('engle_granger_pval', calculate_engle_granger_pval),
        ('wavelet_peak', calculate_wavelet_peak)
    ]
    
    # Test individuel de chaque feature
    individual_scores = {}
    for feature_name, feature_func in features_list:
        print(f"\nTest de {feature_name}")
        score = test_individual_feature(ticker_target, feature_name, feature_func)
        individual_scores[feature_name] = score
    
    # Test combiné de toutes les features
    print("\nTest combiné de toutes les features")
    data = yf.download(ticker_target, start="2010-01-01", end="2023-06-23")
    data = data.dropna()
    
    # Calcul de toutes les features
    for feature_name, feature_func in features_list:
        if feature_name == 'real_rate_diff':
            data[feature_name] = calculate_real_rate_diff(ticker_target)
        else:
            data[feature_name] = feature_func(data, ticker_target)
    
    # Calcul des rendements
    data['returns_target'] = data['Close'].pct_change()
    data = data.dropna()
    
    # Préparation des données
    features = data[[feat[0] for feat in features_list]]
    target = data['returns_target']
    axes[0, 0].scatter(data['volatility_gold'], data['returns_target'], alpha=0.5)
    axes[0, 0].set_title('Volatilité de l\'or vs Rendements du Ticker')
    axes[0, 0].set_xlabel('Volatilité de l\'or')
    axes[0, 0].set_ylabel('Rendements du Ticker')
    
    # Graphique 2 : Rendements de l'or vs Rendements du ticker
    axes[0, 1].scatter(data['returns_gold'], data['returns_target'], alpha=0.5)
    axes[0, 1].set_title('Rendements de l\'or vs Rendements du Ticker')
    axes[0, 1].set_xlabel('Rendements de l\'or')
    axes[0, 1].set_ylabel('Rendements du Ticker')
    
    # Graphique 3 : Volatilité du ticker vs Rendements du ticker (avec ligne de régression)
    axes[1, 0].scatter(data['volatility_target'], data['returns_target'], alpha=0.5)
    axes[1, 0].set_title('Volatilité du Ticker vs Rendements du Ticker')
    axes[1, 0].set_xlabel('Volatilité du Ticker')
    axes[1, 0].set_ylabel('Rendements du Ticker')
    
    # Graphique 4 : Comparaison des valeurs réelles et prédites
    # Séparation des données en train/test (66/33)
    train_size = int(len(features) * 0.66)
    X_train, X_test = features[:train_size], features[train_size:]
    y_train, y_test = target[:train_size], target[train_size:]
    
    # Entraînement du modèle sur les données d'entraînement
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(X_train, y_train)
    
    # Prédiction sur les données de test
    y_pred = model.predict(X_test)
    
    # Création du graphique de comparaison
    axes[1, 1].plot(y_test.index, y_test, label='Valeurs réelles', color='blue')
    axes[1, 1].plot(y_test.index, y_pred, label='Valeurs prédites', color='red', linestyle='--')
    axes[1, 1].set_title('Comparaison Valeurs Réelles vs Prédites')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Rendements')
    axes[1, 1].legend()
    
    # Création d'un DataFrame pour la prédiction
    x_range = pd.DataFrame({
        'volatility_target': np.linspace(data['volatility_target'].min(), 
                                      data['volatility_target'].max(), 100),
        'returns_gold': data['returns_gold'].mean(),
        'volatility_gold': data['volatility_gold'].mean(),
        'returns_crude': data['returns_crude'].mean(),
        'volatility_crude': data['volatility_crude'].mean()
    })
    
    y_pred = model.predict(x_range)
    
    axes[1, 0].plot(x_range['volatility_target'], y_pred, color='red', linestyle='--', 
                    label=f'R² = {model.score(features, target):.4f}')
    axes[1, 0].legend()
    
    # Affichage des statistiques
    print(f"\nStatistiques pour {ticker_target}:")
    print(f"Nombre d'observations: {len(data)}")
    print(f"Corrélation volatilité-target: {data['volatility_target'].corr(data['returns_target']):.4f}")
    print(f"Corrélation rendements-or: {data['returns_gold'].corr(data['returns_target']):.4f}")
    print(f"R² du modèle de régression: {model.score(features, target):.4f}")
    
    plt.tight_layout()
    plt.show()
    
    return model, data

if __name__ == "__main__":
    # Analyser USD/JPY en utilisant les données de l'or comme features
    print("\nAnalyse de USD/JPY avec features de l'or")
    model, data = features_testing('USDJPY=X')