
import sys
import os
import pandas as pd
import numpy as np

# Ensure we can import from the main directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_bot import TradingBot

def test_xgboost_bot():
    print("Testing TradingBot with XGBoost configuration...")
    
    # Mock config
    config = {
        "bot": {
            "market_type": "forex",
            "paper_trading": True,
            "initial_capital": 10000,
            "trading_interval_seconds": 1,
            "seq_length": 30
        },
        "markets": {
            "stock": {"symbols": ["AAPL"]},
            "forex": {"symbols": ["EURUSD=X"]}
        },
        "model": {
            "type": "xgboost",
            "path": "models/xgboost_forex.json",
            "xgboost": {
                "max_depth": 3,
                "n_estimators": 10
            }
        },
        "risk": {
            "max_position_size_pct": 0.1,
            "min_confidence": 0.5,
            "forex_risk": {
                "risk_per_trade_pct": 0.01,
                "stop_loss_pips": 20,
                "risk_reward_ratio": 2.0
            }
        }
    }
    
    try:
        # Initialize Bot
        bot = TradingBot(config)
        print("Bot initialized successfully")
        
        # Test Feature Preparation
        print("Testing feature preparation...")
        # Create dummy OHLCV data
        dates = pd.date_range(start="2023-01-01", periods=300, freq='h')
        data = {
            'Open': np.random.rand(300) * 10 + 100,
            'High': np.random.rand(300) * 10 + 100,
            'Low': np.random.rand(300) * 10 + 100,
            'Close': np.random.rand(300) * 10 + 100,
            'Volume': np.random.rand(300) * 1000
        }
        df = pd.DataFrame(data, index=dates)
        
        # Inject some basic technical patterns to ensure indicators work
        df['Close'] = df['Close'].rolling(window=5).mean().fillna(method='bfill')
        
        features = bot.prepare_features(df)
        if features is not None and len(features) > 0:
            print(f"Features prepared successfully. Shape: {features.shape}")
        else:
            print("Feature preparation failed")
            return
            
        # Test Prediction
        print("Testing prediction...")
        prediction = bot.make_prediction(features)
        print(f"Prediction received: {prediction}")
        
        print("\nIntegration Test Passed!")
        
    except Exception as e:
        print(f"\nTest Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_xgboost_bot()
