import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Télécharger les donnée

eurusd = yf.download("EURUSD=X", period="20y")['Close']
gold = yf.download("GC=F", period="20y")['Close']
oil = yf.download("CL=F", period="20y")['Close']

eurusd = pd.DataFrame(eurusd)
eurusd.columns = ['eurusd']

gold_return = gold.pct_change()
oil_return = oil.pct_change()

df = pd.concat([eurusd, gold_return, oil_return], axis=1)
df.columns = ['eurusd', 'gold_return', 'oil_return']

df['log_return'] = np.log(df['eurusd'] / df['eurusd'].shift(1))

df_model = df[['gold_return', 'oil_return', 'log_return']].dropna().copy()

# Features laggées, carrés et interaction
df_model['gold_return_lag1'] = df_model['gold_return'].shift(1)
df_model['oil_return_lag1'] = df_model['oil_return'].shift(1)
df_model['gold_return_sq'] = df_model['gold_return'] ** 2
df_model['oil_return_sq'] = df_model['oil_return'] ** 2
df_model['interaction'] = df_model['gold_return'] * df_model['oil_return']
df_model = df_model.dropna()

# Variable cible binaire (1 si variation positive, 0 sinon)
df_model['target'] = (df_model['log_return'] > 0).astype(int)

X = df_model[['gold_return', 'oil_return', 'gold_return_lag1', 'oil_return_lag1',
              'gold_return_sq', 'oil_return_sq', 'interaction']].values
y = df_model[['target']].values

# Normalisation et ajout du biais
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = np.hstack((X_scaled, np.ones((X.shape[0], 1))))

# Initialisation theta
theta = np.zeros((X.shape[1], 1))

# Fonctions régression logistique
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model_logistic(X, theta):
    return sigmoid(X.dot(theta))

def cost_function_logistic(X, y, theta):
    m = len(y)  
    h = model_logistic(X, theta)
    epsilon = 1e-5
    cost = -(1/m) * (y.T.dot(np.log(h + epsilon)) + (1 - y).T.dot(np.log(1 - h + epsilon)))
    return cost.flatten()[0]

def grad_logistic(X, y, theta):
    m = len(y)
    h = model_logistic(X, theta)
    return (1/m) * X.T.dot(h - y)

def gradient_descent_logistic(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        theta = theta - learning_rate * grad_logistic(X, y, theta)
        cost_history[i] = cost_function_logistic(X, y, theta)
    return theta, cost_history

# Paramètres apprentissage
learning_rate = 1
n_iterations = 1000


# Entraînement
theta_final, cost_history = gradient_descent_logistic(X, y, theta, learning_rate, n_iterations)

# Prédictions
pred_prob = model_logistic(X, theta_final)
pred_class = (pred_prob >= 0.5).astype(int)

# Évaluation
accuracy = (pred_class == y).mean()
print(f"Accuracy du modèle : {accuracy:.4f}")


# Affichage de la courbe de coût
plt.plot(cost_history)
plt.title("Coût en fonction des itérations")
plt.xlabel("Itérations")
plt.ylabel("Coût")
plt.grid(True)
plt.show()


plt.figure(figsize=(14, 6))

# Premier subplot : classes vraies vs classes prédites
plt.subplot(2,1,1)
plt.scatter(range(len(y)), y, label='Classe vraie (0 ou 1)', color='green', s=20, alpha=0.6)
plt.scatter(range(len(pred_class)), pred_class + 0.05, label='Classe prédite (0 ou 1)', color='red', s=20, alpha=0.6)
plt.yticks([0, 1])
plt.ylim(-0.2, 1.2)
plt.title("Classes vraies et classes prédites (légèrement décalées pour la lisibilité)")
plt.xlabel("Jour")
plt.ylabel("Classe")
plt.legend()
plt.grid(True)

# Deuxième subplot : probabilités prédite
plt.subplot(2,1,2)
plt.plot(range(len(pred_prob)), pred_prob, label='Probabilité prédite', color='blue')
plt.title("Probabilité prédite de variation positive")
plt.xlabel("Jour")
plt.ylabel("Probabilité")
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

def regression_poyl(ticker,ticker2_corr,ticker3_corr=None):
    return "caca"