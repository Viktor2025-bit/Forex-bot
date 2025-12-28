import pandas as pd
import numpy as np
from data_loader import download_data
from feature_engine import FeatureEngine
from ai_model import ForexModel
import os

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

def train_pipeline():
    print("=== Starting AI Training Pipeline ===")
    
    # 1. Get Data (Enough for training)
    # Using 2 years of data for better generalization
    file_path = download_data(pair="EURUSD=X", period="2y", interval="1h")
    
    # Load CSV - FIX: Read the multi-level header properly
    data = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
    
    # Debug: Print what we loaded
    print(f"Raw data shape: {data.shape}")
    print(f"Raw columns: {data.columns.tolist()}")
    
    # Flatten multi-index columns - yfinance creates ('Close', 'EURUSD=X') format
    data.columns = [col[0] for col in data.columns.values]
    
    # Verify we have the expected columns
    print(f"Columns after flattening: {data.columns.tolist()}")
    print(f"Data shape: {data.shape}")
    print(f"First few rows:\n{data.head()}")
    
    # Check for empty data
    if len(data) == 0:
        raise ValueError("❌ No data loaded! Check your CSV file structure.")
    
    # 2. Prepare Data (Features + Targets)
    df_processed = prepare_data(data)
    
    # Check if we still have data after processing
    if len(df_processed) == 0:
        raise ValueError("❌ No data after feature engineering! Check your FeatureEngine and NaN handling.")
    
    # Define features to use for training (exclude non-feature columns)
    feature_cols = [c for c in df_processed.columns if c not in ['Target', 'Future_Return', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    print(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")  # Print first 5
    
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
    model = ForexModel()
    model.train(X_train, y_train)
    
    # 5. Evaluate
    print("\nEvaluating Model...")
    preds_prob = model.predict(X_test)
    preds_class = (preds_prob > 0.5).astype(int)
    
    accuracy = (preds_class == y_test).mean()
    print(f"Test Accuracy: {accuracy:.2%}")
    
    # Feature Importance
    try:
        import matplotlib.pyplot as plt
        from xgboost import plot_importance
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_importance(model.model, max_num_features=10, ax=ax)
        plt.title("Top 10 Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        print("Feature importance plot saved to feature_importance.png")
    except Exception as e:
        print(f"Could not create feature importance plot: {e}")

    # 6. Save Model
    model.save_model()
    print("\n✅ Training Complete. Model ready for use.")

if __name__ == "__main__":
    train_pipeline()