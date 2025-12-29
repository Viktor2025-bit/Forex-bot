"""Simple test script to verify the bot works."""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing AI Trading Bot Components...")
print("=" * 50)

# Test 1: Data Loading
print("\n[1] Testing Data Loader...")
try:
    from utils.data_loader import fetch_historical_data
    df = fetch_historical_data("AAPL", "2023-01-01", "2023-06-30")
    if df is not None and len(df) > 0:
        print(f"   ✓ Downloaded {len(df)} rows of AAPL data")
    else:
        print("   ✗ Failed to download data")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Preprocessing
print("\n[2] Testing Preprocessor...")
try:
    from utils.preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_indicators = preprocessor.add_technical_indicators(df_clean)
    df_normalized = preprocessor.normalize_data(df_indicators)
    print(f"   ✓ Preprocessed data shape: {df_normalized.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Model
print("\n[3] Testing LSTM Model...")
try:
    import torch
    from models.lstm_model import TradingLSTM
    model = TradingLSTM(input_size=7, hidden_size=64, num_layers=2)
    dummy_input = torch.randn(1, 30, 7)
    output = model(dummy_input)
    print(f"   ✓ Model output shape: {output.shape}, value: {output.item():.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Backtester
print("\n[4] Testing Backtester...")
try:
    import numpy as np
    import pandas as pd
    from strategies.backtester import Backtester
    backtester = Backtester(initial_capital=10000)
    # Quick test with random data
    predictions = np.random.random(50)
    prices = 100 + np.cumsum(np.random.randn(51))
    dates = pd.date_range(start="2023-01-01", periods=51)
    result = backtester.run_backtest(predictions, prices, dates, threshold=0.6)
    print(f"   ✓ Backtest complete: {result.num_trades} trades, Return: {result.total_return_pct:.2f}%")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("All tests passed! ✓")
print("=" * 50)
