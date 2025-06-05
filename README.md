# 📈 OptiWealth Solutions – Trading Assistant

> Assistant de trading intelligent orienté Forex, combinant analyse technique, gestion de risque adaptative et modules d’aide à la décision, conçu pour les traders swing et intraday.

---

## 🧠 Objectif du projet

Ce projet vise à fournir un assistant modulaire pour :

- 📊 Étudier la volatilité et la tendance des paires de devises
- 🎯 Calculer dynamiquement les niveaux de stop-loss et les tailles de position
- 🔁 Intégrer des indicateurs techniques et de volatilité (ATR, SMA, RSI, etc.)
- 🧩 Développer un moteur de stratégie basé sur l’analyse quantitative

---

## ⚙️ Fonctionnalités clés

- **Gestion du risque adaptative** (`risk_management.py`)  
  Ajuste automatiquement le risque par trade en fonction de la volatilité et de la VaR.

- **Calcul de l’ATR & du coefficient K** (`atr_index`)  
  Permet un dimensionnement dynamique du stop-loss selon la volatilité du marché.

- **Sizing de position précis** (`position_sizing`)  
  Calcule les lots à trader selon le capital, le stop en pips, et la valeur du pip.

- **Analyse de tendance (trend following)** _(via vectorbt ou pandas-ta)_  
  Utilise les croisements de moyennes mobiles pour suivre les signaux de marché.

- **Génération de rapports PDF** _(volatilité & analyse technique)_  
  Archive automatique des graphiques dans `vol pdf reports/`.

---

## 🚀 Installation

### Clone le repo

```
git clone https://github.com/leolombardini/OWS_Trading_Assistant.git
cd OWS_Trading_Assistant
```

### Lancer le setup (Python 3.10 requis)

```
chmod +x setup.sh
./setup.sh
```

⸻

📊 À venir

- Ajout d’un module de backtesting vectorbt
- Intégration avec des API broker (MetaTrader, OANDA)
- Tableau de bord interactif en Streamlit ou Dash
- Ajout des sentiments de marché et des positions retails
- Gestion multi-devises optimisée
- Mise en place d'un dashboard qui reuni toutes les informations nécéssaires :
  - pouvoir choisir les paires et avoir les ratios, resultats, et informations complémentaires correspondantes --> inspiration du prime market terminal

⸻

#### 🧑‍💻 Dépendances techniques

Fichier `requirement.txt` :

```
yfinance
pandas
numpy
matplotlib
scipy
vectorbt
pandas_ta
statsmodels

```

⸻

🧠 Auteur

Léo Lombardini
Trading & Quantitative Strategy – Étudiant en economie et finance
📧 optiwealth.solutions@gmail.com
