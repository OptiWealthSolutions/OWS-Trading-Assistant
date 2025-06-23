import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif

# ==== Exemple de données simulées ====
# En pratique, tu rempliras ces colonnes avec tes vraies données.
np.random.seed(42)
n = 100000000000


df = pd.DataFrame({
    'z_score': np.random.normal(0, 1, n),
    'corr_gold': np.random.uniform(-1, 1, n),
    'corr_pair': np.random.uniform(-1, 1, n),
    'volatility': np.abs(np.random.normal(0, 1, n)),
    'volume': np.random.randint(1000, 10000, n),
    'sentiment': np.random.uniform(-1, 1, n),
    'adx': np.random.uniform(0, 50, n),
    'sma_50_200_cross': np.random.choice([0, 1], size=n)
})

# ==== Cible : direction simulée (à remplacer par la tienne) ====
df['return_future'] = np.random.normal(0, 1, n)
df['y'] = 0  # HOLD par défaut
df.loc[df['return_future'] > 0.5, 'y'] = 2  # BUY
df.loc[df['return_future'] < -0.5, 'y'] = 1  # SELL

# ==== Features (X) et target (y) ====
X = df[[
    'z_score', 'corr_gold', 'corr_pair',
    'volatility', 'volume', 'sentiment',
    'adx', 'sma_50_200_cross'
]]
y = df['y']

# ==== Analyse individuelle des features ====
def test_features_individually(X, y):
    print("=== Test des features indépendamment ===")
    for col in X.columns:
        model = LogisticRegression(max_iter=1000)
        score = cross_val_score(model, X[[col]], y, cv=5, scoring='accuracy').mean()
        mi = mutual_info_classif(X[[col]], y, discrete_features='auto')[0]
        print(f"{col:20s} ➤ Accuracy : {score:.3f} | Mutual Info : {mi:.3f}")

test_features_individually(X, y)