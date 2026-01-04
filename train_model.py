import pandas as pd
import numpy as np
from feature_engine import FeatureEngine
from ai_model import ForexModel
import os
import argparse

def prepare_data(data, lookahead=1):
    """
    Prepare data for training:
    - Generate features
    - Create target variable
    """
    # 1. Generate Technical Features
    fe = FeatureEngine()
    df = fe.generate_features(data)
    
    # 2. Create Target Variable
    # Target: 1 if Price[t + lookahead] > Price[t], else 0
    # Future returns
    df['Future_Return'] = df['Close'].shift(-lookahead) - df['Close']
    df['Target'] = (df['Future_Return'] > 0).astype(int)
    
    # Drop NaNs created by shifting
    df.dropna(inplace=True)
    
    print(f"Data prepared. Shape: {df.shape}")
    print(f"Class balance: {df['Target'].value_counts(normalize=True)}")
    
    return df

def train_pipeline(symbol="R_75"):
    print(f"=== Starting AI Training Pipeline for {symbol} ===")
    
    # 1. Get Data from local storage
    file_path = f"data/raw/{symbol}.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Data file not found: {file_path}. Run the bot to collect data first.")
        
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Debug: Print what we loaded
    print(f"Raw data shape: {data.shape}")
    print(f"First few rows:\n{data.head()}")
    
    # Check for empty data
    if len(data) < 100:
        raise ValueError(f"❌ Not enough data! Only {len(data)} rows. need at least 100.")
    
    # 2. Prepare Data (Features + Targets)
    try:
        df_processed = prepare_data(data)
    except Exception as e:
        input_cols = data.columns.tolist()
        raise ValueError(f"Feature engineering failed on columns {input_cols}. Error: {e}")
    
    # Check if we still have data after processing
    if len(df_processed) == 0:
        raise ValueError("❌ No data after feature engineering! Check your FeatureEngine and NaN handling.")
    
    # Define features to use for training (exclude non-feature columns)
    ignore_cols = ['Target', 'Future_Return', 'Date', 'index', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [c for c in df_processed.columns if c not in ignore_cols]
    
    print(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")
    
    X = df_processed[feature_cols]
    y = df_processed['Target']
    
    # 3. Train/Test Split (Time-based split, NOT random shuffle)
    # Train on first 80%, Test on last 20%
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    # 4. Train Model
    print("\nTraining XGBoost model...")
    model_path = f"models/{symbol}_xgboost.json"
    model = ForexModel(model_path=model_path, n_estimators=500, max_depth=6) # Slightly boosted params
    model.train(X_train, y_train)
    
    # 5. Evaluate
    print("\nEvaluating Model...")
    preds_prob = model.predict(X_test)
    preds_class = (preds_prob > 0.5).astype(int)
    
    accuracy = (preds_class == y_test).mean()
    print(f"Test Accuracy: {accuracy:.2%}")
    
    # 6. Save Model
    model.save_model()
    print(f"\n✅ Training Complete for {symbol}. Model saved to {model_path}.")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Model for Synthetics")
    parser.add_argument("--symbol", type=str, default="R_75", help="Symbol to train (e.g. R_75, CRASH500)")
    args = parser.parse_args()
    
    try:
        train_pipeline(args.symbol)
    except Exception as e:
        print(f"Training Failed: {e}")