import pandas as pd
import numpy as np
from backtester import Backtester
from strategies.risk_manager import RiskManager, RiskParameters

def simple_ma_strategy(df, short_window=20, long_window=50):
    """Generate signals for a simple MA crossover strategy."""
    signals = pd.DataFrame(index=df.index)
    signals['Signal'] = 0.0

    # Create short simple moving average
    signals['short_mavg'] = df['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

    # Create long simple moving average
    signals['long_mavg'] = df['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['Signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                                > signals['long_mavg'][short_window:], 1.0, 0.0)   

    # Generate trading orders
    signals['Position'] = signals['Signal'].diff()
    
    # Map 1.0 to 1 (Buy), -1.0 to -1 (Sell) in Position
    # Signal column needs to be 1 for Long, -1 for Short state? 
    # Backtester expects: Signal=1(long), -1(short), 0(neutral)
    # The code above produces Signal=1 when Short > Long (Bullish).
    # We want Signal column to represent the *State* (1=Long, -1=Short)? 
    # Backtester.py Line 60: Strategy_Return = Daily_Return * Signal.shift(1)
    # So Signal should be the HELD position.
    # But wait, Backtester.py Line 178: position = 1 if row['Signal'] == 1 else -1.
    # This implies Signal is the *Entry Command*.
    
    # Let's standardize:
    # Signal: 1 (Go Long), -1 (Go Short), 0 (Neutral/Hold/No Change)
    # But Backtester logic is a bit mixed. 
    # Let's clean up logic: 
    # If SMA_Short > SMA_Long -> We want to be Long (1).
    # If SMA_Short < SMA_Long -> We want to be Short (-1).
    
    df['Signal'] = 0
    df.loc[signals['short_mavg'] > signals['long_mavg'], 'Signal'] = 1
    df.loc[signals['short_mavg'] < signals['long_mavg'], 'Signal'] = -1
    
    # Backtester uses 'Position' column check for changes?
    df['Position'] = df['Signal'].diff()
    
    return df

def run_backtest_session(symbol, data, initial_capital, risk_pct=0.01):
    """
    Run a backtest session.
    data: DataFrame with OHLCV
    """
    # 1. Apply Strategy (Using MA for now as placeholder for AI)
    # Ideally we'd load the XGBoost model here.
    processed_data = simple_ma_strategy(data)
    
    # 2. Setup Risk Manager
    # Create simple params
    # We need to mock RiskParameters since it's a dataclass in risk_manager.py
    # But we can just pass specific params if we init RiskManager manually?
    # RiskManager init takes: config dictionary.
    
    risk_config = {
        "max_position_size_pct": 1.0, # Backtester handles sizing logic mostly
        "max_daily_loss_pct": 0.05,
        "risk_per_trade_pct": risk_pct,
        "stop_loss_pct": 0.02, # 2% SL
        "take_profit_pct": 0.04 # 4% TP
    }
    
    # We might need to adjust RiskManager or usage in Backtester.
    # Backtester init: risk_manager=None
    
    tester = Backtester(processed_data, initial_capital=initial_capital)
    results = tester.run()
    
    return results, processed_data
