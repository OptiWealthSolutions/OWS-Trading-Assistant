#!/bin/bash

echo "🔧 Création de l'environnement Python .venv..."

# Crée un environnement virtuel
python3 -m venv .venv

# Active l'environnement
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Mise à jour de pip..."
pip install --upgrade pip

# Installe les dépendances
echo "📦 Installation des dépendances..."
pip install -r requirements.txt

echo "✅ Environnement prêt. Activez-le avec : source .venv/bin/activate"

