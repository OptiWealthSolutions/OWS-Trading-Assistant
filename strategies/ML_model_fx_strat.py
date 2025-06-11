import pandas as pd
import numpy as np
from fx_strategy_V1 import prepare_dataset_signal  # ajuster le chemin si nécessaire
import yfinance as yf
def prepare_polynomial_features(X):
    """
    Étend X avec des termes polynomiaux de degré 2.
    """
    from itertools import combinations_with_replacement
    cols = list(X.columns)
    X_poly = X.copy()
    for i, j in combinations_with_replacement(cols, 2):
        X_poly[f"{i}*{j}"] = X[i] * X[j]
    return X_poly

def cost_function_mse(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent_mse(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = X.dot(theta)
        theta -= (learning_rate / m) * X.T.dot(predictions - y)
        cost_history[i] = cost_function_mse(X, y, theta)
    return theta, cost_history

def train_model(X, y, learning_rate=0.1, iterations=1000):
    """
    Entraîne un modèle de régression polynomiale de degré 2 via descente de gradient sur MSE.
    """
    X_poly = prepare_polynomial_features(X)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    X_scaled = np.hstack([X_scaled, np.ones((X_scaled.shape[0], 1))])  # ajout biais
    theta = np.zeros((X_scaled.shape[1], 1))
    y = y.reshape(-1, 1)
    theta_final, cost_history = gradient_descent_mse(X_scaled, y, theta, learning_rate, iterations)
    return theta_final, scaler, X_poly.columns, cost_history

if __name__ == "__main__":


    # Exemple d'utilisation avec données réelles
    pair1 = yf.download("EURUSD=X", period="1y")["Close"]
    pair2 = yf.download("GBPUSD=X", period="1y")["Close"]
    gold = yf.download("GC=F", period="1y")["Close"]

    spread = pair1 - pair2
    spread = spread.dropna()
    zscore = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()
    adx_dummy = pd.Series(np.random.uniform(10, 30, size=len(spread)), index=spread.index)  # placeholder

    X, y_class = prepare_dataset_signal(spread, zscore, pair1, gold, adx_dummy)
    y = y_class.astype(float)  # convertit les signaux -1, 0, 1 en valeurs continues pour MSE

    theta_final, scaler, features, cost_history = train_model(X, y)

    print("Dernier coût :", cost_history[-1])