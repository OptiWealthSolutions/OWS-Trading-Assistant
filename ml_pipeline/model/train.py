import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import warnings
warnings.filterwarnings("ignore")

def prepare_training_data(features, labels):
    """Prépare les données pour l'entraînement"""
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Entraîne le modèle RandomForest"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle sur les données de test"""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

def save_model(model, model_path):
    """Sauvegarde le modèle"""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def main(features_path, labels_path, model_path):
    """Fonction principale d'entraînement"""
    # Charger les données
    features = pd.read_csv(features_path, index_col='date', parse_dates=['date'])
    labels = pd.read_csv(labels_path, index_col='date', parse_dates=['date'])
    
    # Préparer les données
    X_train, X_test, y_train, y_test = prepare_training_data(features, labels)
    
    # Entraîner le modèle
    model = train_model(X_train, y_train)
    
    # Évaluer le modèle
    report = evaluate_model(model, X_test, y_test)
    print("\nRapport d'évaluation:")
    print(report)
    
    # Sauvegarder le modèle
    save_model(model, model_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Chemin vers le fichier des features")
    parser.add_argument("--labels", required=True, help="Chemin vers le fichier des labels")
    parser.add_argument("--model", required=True, help="Chemin pour sauvegarder le modèle")
    args = parser.parse_args()
    
    main(args.features, args.labels, args.model)
