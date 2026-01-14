"""
Full Pipeline: Train Model + Backtest

This script combines training and backtesting to give a complete picture
of how the model would have performed historically.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    print("Warning: yfinance import failed")

from models.lstm_model import TradingLSTM, create_sequences
from ai_model import ForexModel
from utils.data_loader import fetch_historical_data
from utils.preprocessing import DataPreprocessor
from feature_engine import FeatureEngine
from strategies.backtester import Backtester


def run_full_pipeline(ticker="EURUSD=X", 
                      start_date="2020-01-01", 
                      end_date="2023-12-31",
                      seq_length=30,
                      epochs=50,
                      threshold=0.5,
                      model_type="xgboost"):
    """
    Complete pipeline: Fetch data -> Preprocess -> Train -> Backtest.
    """
    print("=" * 60)
    print(f"AI TRADING BOT - FULL PIPELINE ({model_type.upper()})")
    print(f"Ticker: {ticker} | Period: {start_date} to {end_date}")
    print("=" * 60)
    
    # ========== STEP 1: DATA ==========
    print("\n[STEP 1/5] Fetching historical data...")
    
    # Check if trading synthetics
    from utils.config_loader import load_config
    config = load_config()
    
    market_type = config.get('bot', {}).get('market_type')
    
    if market_type == 'synthetics':
        # Use Deriv API for synthetics
        print("Fetching from Deriv API (synthetics)...")
        from data_loader import fetch_historical_data as fetch_deriv
        import asyncio
        df = asyncio.run(fetch_deriv(ticker, time_interval="5m", max_candles=50000))
    else:
        # Use MT5 for forex/indices
        from utils.data_loader import fetch_data_mt5
        data_map = fetch_data_mt5(ticker, n_candles=50000)
        if data_map and ticker in data_map:
            df = data_map[ticker]
        else:
            df = None

    if df is None or df.empty:
        print("CRITICAL ERROR: Could not fetch data. Please check your connection.")
        return
        
    print(f"Data rows: {len(df)}")
    
    # Initialize variables for backtest
    predictions = []
    test_prices = []
    test_dates = []
    
    if model_type == 'xgboost':
        # ========== XGBoost Pipeline ==========
        
        # 1. Feature Engineering
        print("\n[STEP 2/5] Feature Engineering (XGBoost)...")
        fe = FeatureEngine()
        df_features = fe.generate_features(df)
        
        # Prepare Targets
        # Target: 1 if Price[t+1] > Price[t], else 0 (Simplified)
        df_features['Future_Return'] = df_features['Close'].shift(-1) - df_features['Close']
        df_features['Target'] = (df_features['Future_Return'] > 0).astype(int)
        df_features.dropna(inplace=True)
        
        ignore_cols = ['Target', 'Future_Return', 'Date', 'index', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [c for c in df_features.columns if c not in ignore_cols]
        
        X = df_features[feature_cols]
        y = df_features['Target']
        
        # Split Data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Store for backtest
        test_prices = df_features['Close'].iloc[split_idx:].values
        test_dates = df_features.index[split_idx:]
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # 2. Train
        print("\n[STEP 3/5] Training XGBoost model...")
        model = ForexModel(model_path=f'models/{ticker}_xgboost.json') # Save to ticker-specific file
        model.train(X_train, y_train)
        
        # 3. Predict
        print("\n[STEP 4/5] Predicting...")
        probs = model.predict(X_test) # Returns probs for class 1
        predictions = probs
        
        # Evaluate
        preds_class = (probs > 0.5).astype(int)
        acc = (preds_class == y_test).mean()
        print(f"Model Accuracy: {acc:.1%}")
        
        model.save_model()

    else:
        # ========== LSTM Pipeline (Enhanced with FeatureEngine) ==========
        print("\n[STEP 2/5] Feature Engineering (LSTM - Enhanced)...")
        
        # Use the same FeatureEngine as XGBoost for consistency
        fe = FeatureEngine()
        df_features = fe.generate_features(df)
        
        # Normalize all numeric features for LSTM
        from sklearn.preprocessing import StandardScaler
        
        # Exclude non-numeric columns
        feature_columns = [col for col in df_features.columns if df_features[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        feature_columns = [col for col in feature_columns if col not in ['Date', 'Datetime']]
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_features[feature_columns].values)
        
        # Store original close prices for backtesting
        original_close = df_features['Close'].values
        dates = df_features.index
        
        print(f"Processed shape: {data_scaled.shape} ({len(feature_columns)} features)")
        
        # Create sequences
        print("\n[STEP 3/5] Creating sequences...")
        X, y = create_sequences(data_scaled, seq_length=seq_length)
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Align prices/dates
        test_prices = original_close[seq_length + split_idx:]
        test_dates = dates[seq_length + split_idx:]
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train
        print("\n[STEP 4/5] Training LSTM model...")
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_t = torch.FloatTensor(X_test)
        
        model = TradingLSTM(input_size=X_train.shape[2], hidden_size=64, num_layers=2)
        criterion = torch.nn.BCEWithLogitsLoss() # More stable than Sigmoid + BCELoss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # Lower LR for stability
        
        batch_size = 32
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_t[i:i+batch_size]
                batch_y = y_train_t[i:i+batch_size]
                optimizer.zero_grad()
                predictions_train = model(batch_X)
                loss = criterion(predictions_train, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/(len(X_train_t)//batch_size):.4f}")
        
        # Predict
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t).squeeze()
            predictions = torch.sigmoid(logits).numpy() # Apply sigmoid here since model returns logits
            
        torch.save(model.state_dict(), f'models/{ticker}_lstm.pth')

    # ========== STEP 5: BACKTEST ==========
    print("\n[STEP 5/5] Running backtest...")
    
    # Ensure consistency
    min_len = min(len(predictions), len(test_prices)-1)
    # Note: prices need to include the next day for PnL calculation in current Backtester logic
    # Backtester expects:
    # predictions[i] -> decision at close of i
    # prices[i] -> entry price
    # prices[i+1] -> exit price (next day)
    
    use_preds = predictions[:min_len]
    use_prices = test_prices[:min_len+1].squeeze()
    use_dates = test_dates[:min_len+1]
    
    print(f"Backtesting on {len(use_preds)} candles.")
    
    backtester = Backtester(initial_capital=10000)
    result = backtester.run_backtest(
        predictions=use_preds,
        actual_prices=use_prices,
        dates=use_dates,
        threshold=threshold
    )
    
    backtester.print_results(result)
    
    # Save Plot
    try:
        os.makedirs('data/processed', exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(result.equity_curve.values)
        plt.title(f'{ticker} ({model_type.upper()}) - Equity Curve')
        plt.xlabel('Trades')
        plt.ylabel('Equity ($)')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'data/processed/equity_curve_{model_type}.png')
        print(f"\nEquity curve saved to data/processed/equity_curve_{model_type}.png")
    except Exception as e:
        print(f"Could not save plot: {e}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="EURUSD=X")
    parser.add_argument("--model", type=str, default="xgboost", choices=["lstm", "xgboost"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    run_full_pipeline(ticker=args.ticker, model_type=args.model, epochs=args.epochs)
