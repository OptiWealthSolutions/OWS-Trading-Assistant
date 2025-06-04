#!/bin/bash

# ------------ Vérification de Python 3.10 ------------
if ! command -v python3.10 &> /dev/null
then
    echo "❌ Python 3.10 non trouvé. Installe-le avec brew : brew install python@3.10"
    exit 1
fi

# ------------ Création de l'environnement virtuel ------------
echo "📦 Création de l'environnement virtuel .venv..."
python3.10 -m venv .venv

# ------------ Activation (uniquement pour Unix/macOS) ------------
echo "⚙️ Activation de l'environnement..."
source .venv/bin/activate

# ------------ Installation des dépendances ------------
if [ -f requirements.txt ]; then
    echo "📚 Installation depuis requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "❌ Aucun fichier requirements.txt trouvé."
    deactivate
    exit 1
fi

# ------------ Ajout au .gitignore ------------
if ! grep -q "^.venv/$" .gitignore 2>/dev/null; then
    echo ".venv/" >> .gitignore
    echo "✅ .venv/ ajouté au .gitignore"
fi

echo "✅ Environnement prêt. Active-le avec : source .venv/bin/activate"