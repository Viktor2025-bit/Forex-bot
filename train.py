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
from utils.preprocessing import DataPreprocessor


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
    if len(df) < 100:
        print("Not enough data to train.")
        return None
        
    print(f"Loaded {len(df)} rows.")

    # Step 2: Preprocess
    print("\n[2/5] Preprocessing data...")
    try:
        preprocessor = DataPreprocessor()
        # For synthetics/stored CSVs, we might not need extensive cleaning depending on source
        # But let's run standard pipeline
        df_indicators = preprocessor.add_technical_indicators(df)
        df_normalized = preprocessor.normalize_data(df_indicators)
        
        # Features match those in trading_bot.py prepare_features
        # Ensure we use numeric columns
        feature_columns = [col for col in df_normalized.columns if col not in ['Date', 'Datetime', 'Target', 'Future_Return', 'index']]
        data = df_normalized[feature_columns].values
        print(f"Processed data shape: {data.shape}")
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

    # Step 3: Create Sequences
    print("\n[3/5] Creating sequences...")
    X, y = create_sequences(data, seq_length=seq_length)
    if len(X) == 0:
        print("No sequences created.")
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
    model = TradingLSTM(input_size=input_size, hidden_size=64, num_layers=2)
    
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
    
    criterion = nn.BCELoss()
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
            print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")

    # Step 5: Evaluate
    print("\n[5/5] Evaluating...")
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_t)
        predicted_labels = (test_predictions > 0.5).float()
        accuracy = (predicted_labels == y_test_t).float().mean()
        
    print(f"Test Accuracy: {accuracy:.2%}")

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'seq_length': seq_length,
        'ticker': ticker,
        'accuracy': accuracy.item()
    }, model_path)
    print(f"Model saved to {model_path}")
    
    return accuracy.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="R_75")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train_model(ticker=args.ticker, epochs=args.epochs)
