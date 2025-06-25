import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import pickle
import warnings
warnings.filterwarnings("ignore")

def load_model(model_path):
    """Charge le modèle sauvegardé"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def evaluate_predictions(y_true, y_pred):
    """Évalue les prédictions du modèle"""
    report = classification_report(y_true, y_pred, output_dict=True)
    return report

def generate_signals(model, features):
    """Génère les signaux de trading"""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    
    # Convertir les prédictions en signaux
    signals = pd.DataFrame({
        'prediction': predictions,
        'confidence': np.max(probabilities, axis=1)
    }, index=features.index)
    
    return signals

def main(model_path, features_path, output_path):
    """Fonction principale d'évaluation"""
    # Charger le modèle
    model = load_model(model_path)
    
    # Charger les features
    features = pd.read_csv(features_path, index_col='date', parse_dates=['date'])
    
    # Générer les signaux
    signals = generate_signals(model, features)
    
    # Sauvegarder les résultats
    signals.to_csv(output_path)
    
    print(f"Signaux générés et sauvegardés dans {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Chemin vers le modèle")
    parser.add_argument("--features", required=True, help="Chemin vers les features")
    parser.add_argument("--output", required=True, help="Chemin de sortie pour les signaux")
    args = parser.parse_args()
    
    main(args.model, args.features, args.output)
