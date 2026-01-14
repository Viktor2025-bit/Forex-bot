"""
Training Pipeline for the LSTM Trading Model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

from models.lstm_model import TradingLSTM, create_sequences
from feature_engine import FeatureEngine


def train_model(ticker="R_75", seq_length=30, epochs=20, batch_size=32, learning_rate=0.001):
    """
    Complete training pipeline with local data and fine-tuning.
    """
    print("=" * 50)
    print(f"LSTM Automated Retraining for {ticker}")
    print("=" * 50)
    
    # Step 1: Load Local Data
    print(f"\n[1/5] Loading local data for {ticker}...")
    file_path = f"data/raw/{ticker}.csv"
    if not os.path.exists(file_path):
        print(f"Error: Data file {file_path} not found.")
        return None
        
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    if len(df) < 250: # Increased requirement for feature generation
        print("Not enough data to train (min 250 rows required).")
        return None
        
    print(f"Loaded {len(df)} rows.")

    # Step 2: Preprocess using FeatureEngine
    print("\n[2/5] Preprocessing data with FeatureEngine...")
    try:
        feature_engine = FeatureEngine(ticker=ticker)
        
        # Generate features
        df_features = feature_engine.generate_features(df)
        
        # Check if we have enough data after feature generation (SMA(200) etc.)
        if len(df_features) < seq_length * 2:
            print(f"Not enough data remaining after feature generation ({len(df_features)} rows) to create sequences.")
            return None

        # Normalize the data and fit the scaler
        df_normalized = feature_engine.normalize_data(df_features, fit_scaler=True)
        
        # Define feature columns - all numeric columns from the normalized df
        feature_columns = [col for col in df_normalized.columns if df_normalized[col].dtype in [np.int64, np.float64]]
        data = df_normalized[feature_columns].values
        print(f"Processed data shape: {data.shape}")
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Step 3: Create Sequences
    print("\n[3/5] Creating sequences...")
    # Target is the 'Close' price, which we want to predict
    target_col_index = feature_columns.index('Close')
    X, y = create_sequences(data, seq_length=seq_length, target_col_index=target_col_index)
    
    if len(X) == 0:
        print("No sequences created. Check data length and sequence length.")
        return None
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Step 4: Initialize or Load Model
    print("\n[4/5] Initializing model...")
    input_size = X_train.shape[2]
    model = TradingLSTM(input_size=input_size, hidden_size=64, num_layers=2, output_size=1) # Output size is 1 (predicting 'Close')
    
    model_path = f"models/{ticker}_lstm.pth"
    if os.path.exists(model_path):
        try:
            print(f"Loading existing weights from {model_path} for fine-tuning...")
            checkpoint = torch.load(model_path)
            # Check if input size matches
            if checkpoint.get('input_size') == input_size:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("Input size mismatch (new features?), starting fresh.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
    
    criterion = nn.MSELoss() # Changed to MSELoss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}: Loss {avg_loss:.6f}") # Increased precision for MSE

    # Step 5: Evaluate
    print("\n[5/5] Evaluating...")
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_t)
        # For regression, evaluation metric could be something like RMSE
        mse = criterion(test_predictions, y_test_t)
        rmse = torch.sqrt(mse)
        
    print(f"Test RMSE: {rmse.item():.6f}")

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'seq_length': seq_length,
        'ticker': ticker,
        'rmse': rmse.item(),
        'feature_columns': feature_columns # Save feature columns
    }, model_path)
    print(f"Model saved to {model_path}")
    
    return rmse.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="R_75")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train_model(ticker=args.ticker, epochs=args.epochs)
