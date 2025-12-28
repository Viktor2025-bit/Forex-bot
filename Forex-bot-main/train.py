"""
Training Pipeline for the LSTM Trading Model.

This script handles:
1. Loading and preprocessing data
2. Creating train/test sequences
3. Training the LSTM model
4. Saving the trained model
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

from models.lstm_model import TradingLSTM, create_sequences
from utils.data_loader import fetch_historical_data
from utils.preprocessing import DataPreprocessor


def train_model(ticker="AAPL", start_date="2020-01-01", end_date="2023-12-31",
                seq_length=30, epochs=50, batch_size=32, learning_rate=0.001):
    """
    Complete training pipeline.
    """
    print("=" * 50)
    print("LSTM Trading Model Training Pipeline")
    print("=" * 50)
    
    # Step 1: Fetch Data
    print("\n[1/5] Fetching historical data...")
    df = fetch_historical_data(ticker, start_date, end_date)
    if df is None:
        print("Failed to fetch data. Exiting.")
        return None
    print(f"Downloaded {len(df)} rows of data.")

    # Step 2: Preprocess Data
    print("\n[2/5] Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_indicators = preprocessor.add_technical_indicators(df_clean)
    df_normalized = preprocessor.normalize_data(df_indicators)
    
    # Extract features as numpy array
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_14', 'RSI_14']
    data = df_normalized[feature_columns].values
    print(f"Processed data shape: {data.shape}")

    # Step 3: Create Sequences
    print("\n[3/5] Creating sequences...")
    X, y = create_sequences(data, seq_length=seq_length)
    print(f"Sequences created: X={X.shape}, y={y.shape}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle time-series!
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Step 4: Initialize Model
    print("\n[4/5] Training model...")
    input_size = X_train.shape[2]  # Number of features
    model = TradingLSTM(input_size=input_size, hidden_size=64, num_layers=2)
    
    criterion = nn.BCELoss()  # Binary Cross-Entropy for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
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
        
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Step 5: Evaluate Model
    print("\n[5/5] Evaluating model...")
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_t)
        test_loss = criterion(test_predictions, y_test_t)
        
        # Convert probabilities to binary predictions
        predicted_labels = (test_predictions > 0.5).float()
        accuracy = (predicted_labels == y_test_t).float().mean()
        
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2%}")

    # Save Model
    model_path = "models/trained_lstm.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'seq_length': seq_length,
        'ticker': ticker,
        'accuracy': accuracy.item()
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    return model, accuracy.item()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LSTM Trading Model.")
    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Stock ticker or Forex symbol to train on (e.g., 'AAPL', 'EURUSD=X').")
    parser.add_argument("--start_date", type=str, default="2020-01-01",
                        help="Start date for historical data (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default="2023-12-31",
                        help="End date for historical data (YYYY-MM-DD).")
    parser.add_argument("--seq_length", type=int, default=30,
                        help="Sequence length for LSTM input.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for the optimizer.")
    
    args = parser.parse_args()

    train_model(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        seq_length=args.seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
