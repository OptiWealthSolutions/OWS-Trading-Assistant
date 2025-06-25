import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_signals(signals_path):
    """Charge les signaux de trading"""
    return pd.read_csv(signals_path, index_col='date', parse_dates=['date'])

def backtest_strategy(signals, prices, initial_capital=100000, risk_per_trade=0.01):
    """Effectue le backtest de la stratégie"""
    # Initialisation des variables
    capital = initial_capital
    positions = []
    trades = []
    
    # Pour chaque signal
    for date, row in signals.iterrows():
        prediction = row['prediction']
        confidence = row['confidence']
        
        # Calculer la taille de position
        position_size = capital * risk_per_trade
        
        # Enregistrer la position
        positions.append({
            'date': date,
            'prediction': prediction,
            'confidence': confidence,
            'position_size': position_size
        })
        
        # Si c'est une nouvelle position
        if prediction != 1:  # 0: SELL, 2: BUY
            trades.append({
                'entry_date': date,
                'exit_date': None,
                'entry_price': prices.loc[date, 'close'],
                'exit_price': None,
                'position_size': position_size,
                'direction': 'BUY' if prediction == 2 else 'SELL'
            })
    
    # Créer un DataFrame des positions
    positions_df = pd.DataFrame(positions)
    
    # Créer un DataFrame des trades
    trades_df = pd.DataFrame(trades)
    
    return positions_df, trades_df

def calculate_metrics(positions, trades):
    """Calcule les métriques de performance"""
    # Calculer le nombre total de trades
    total_trades = len(trades)
    
    # Calculer le nombre de trades gagnants
    winning_trades = trades[trades['pnl'] > 0]
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    # Calculer le ratio gain/perte
    if len(winning_trades) > 0 and len(trades) - len(winning_trades) > 0:
        avg_win = winning_trades['pnl'].mean()
        avg_loss = trades[trades['pnl'] < 0]['pnl'].mean()
        gain_loss_ratio = abs(avg_win / avg_loss)
    else:
        gain_loss_ratio = 0
    
    metrics = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'gain_loss_ratio': gain_loss_ratio
    }
    
    return metrics

def plot_results(signals, prices, positions):
    """Trace les résultats du backtest"""
    plt.figure(figsize=(15, 8))
    
    # Tracer les prix
    plt.plot(prices.index, prices['close'], label='Prix')
    
    # Tracer les signaux
    buy_signals = signals[signals['prediction'] == 2]
    sell_signals = signals[signals['prediction'] == 0]
    
    plt.scatter(buy_signals.index, 
               prices.loc[buy_signals.index, 'close'], 
               marker='^', color='g', label='Buy Signal')
    
    plt.scatter(sell_signals.index, 
               prices.loc[sell_signals.index, 'close'], 
               marker='v', color='r', label='Sell Signal')
    
    plt.title('Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(signals_path, prices_path):
    """Fonction principale de backtest"""
    # Charger les données
    signals = load_signals(signals_path)
    prices = pd.read_csv(prices_path, index_col='date', parse_dates=['date'])
    
    # Effectuer le backtest
    positions, trades = backtest_strategy(signals, prices)
    
    # Calculer les métriques
    metrics = calculate_metrics(positions, trades)
    
    # Afficher les résultats
    print("\nRésultats du backtest:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Tracer les résultats
    plot_results(signals, prices, positions)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True, help="Chemin vers les signaux")
    parser.add_argument("--prices", required=True, help="Chemin vers les prix")
    args = parser.parse_args()
    
    main(args.signals, args.prices)
