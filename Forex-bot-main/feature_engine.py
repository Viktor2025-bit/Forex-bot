import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange

class FeatureEngine:
    """
    Generates advanced technical indicators and custom features for the AI model.
    Focuses on Forex-specific patterns: Sessions, Trend Efficiency, and Volatility.
    """
    
    def __init__(self):
        pass
        
    def generate_features(self, df):
        """
        Add technical indicators to the dataframe.
        """
        data = df.copy()
        
        # Ensure we have date information depending on index
        if not isinstance(data.index, pd.DatetimeIndex):
            # Try to find a date column or convert index
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
            else:
                # If no date info, we can't do session/time features accurately
                # default to index 
                pass

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
             # Try simple mapping if case differs
             data.columns = [c.capitalize() for c in data.columns]
             if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Dataframe must contain columns: {required_cols}")

        # --- 1. Forex Session Features (Cyclical Encoding) ---
        if isinstance(data.index, pd.DatetimeIndex):
            data['Hour_Sin'] = np.sin(2 * np.pi * data.index.hour / 24)
            data['Hour_Cos'] = np.cos(2 * np.pi * data.index.hour / 24)
            data['Day_Sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
            data['Day_Cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
        else:
            data['Hour_Sin'] = 0
            data['Hour_Cos'] = 0
            data['Day_Sin'] = 0
            data['Day_Cos'] = 0

        # --- 2. Trend & Efficiency ---
        
        # SMAs
        data['SMA_50'] = SMAIndicator(close=data['Close'], window=50).sma_indicator()
        data['SMA_200'] = SMAIndicator(close=data['Close'], window=200).sma_indicator()
        data['Dist_SMA_50'] = (data['Close'] - data['SMA_50']) / data['Close']
        data['Dist_SMA_200'] = (data['Close'] - data['SMA_200']) / data['Close']
        
        # EMA Crossover Proxy (Price vs EMA 20)
        data['EMA_20'] = EMAIndicator(close=data['Close'], window=20).ema_indicator()
        data['Dist_EMA_20'] = (data['Close'] - data['EMA_20']) / data['Close']

        # ADX (Trend Strength)
        adx = ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=14)
        data['ADX'] = adx.adx()
        data['ADX_Pos'] = adx.adx_pos()
        data['ADX_Neg'] = adx.adx_neg()

        # Kaufman Efficiency Ratio (ER) - Custom
        # ER = Direction / Volatility
        # Direction = Abs(Close_t - Close_t-n)
        # Volatility = Sum(Abs(Close_i - Close_i-1))
        n = 10
        direction = np.abs(data['Close'] - data['Close'].shift(n))
        volatility = data['Close'].diff().abs().rolling(window=n).sum()
        data['Efficiency_Ratio'] = direction / volatility
        data['Efficiency_Ratio'].replace([np.inf, -np.inf], 0, inplace=True) # Handle div by zero

        # --- 3. Momentum ---
        
        # RSI
        data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
        
        # Rate of Change (ROC)
        data['ROC'] = ROCIndicator(close=data['Close'], window=12).roc()

        # Stochastic
        stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
        data['Stoch_K'] = stoch.stoch()
        data['Stoch_D'] = stoch.stoch_signal()

        # --- 4. Volatility ---
        
        # Bollinger Bands
        bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
        data['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / data['Close']
        data['BB_Pos'] = (data['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

        # Normalized ATR
        atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14)
        data['ATR'] = atr.average_true_range()
        data['ATR_Pct'] = data['ATR'] / data['Close'] # Volatility relative to price

        # --- 5. Price Action ---
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Lagged Returns
        for lag in [1, 2, 3, 5]:
            data[f'Return_Lag_{lag}'] = data['Log_Return'].shift(lag)
            
        # Volume
        if 'Volume' in data.columns: 
             # Safe pct change
             data['Volume_Change'] = data['Volume'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        else:
             data['Volume_Change'] = 0

        # --- Cleanup ---
        # Instead of generic dropna, we drop only the initialization rows
        # SMA_200 requires 200 rows.
        data.dropna(inplace=True)
        
        return data

if __name__ == "__main__":
    # Test with dummy data
    print("Testing Advanced Feature Engine (Dummy Data)...")
    
    # Create 500 hours of data to handle 200-period SMA
    dates = pd.date_range(start="2024-01-01", periods=500, freq='h')
    data = {
        'Open': np.random.rand(500) * 10 + 100,
        'High': np.random.rand(500) * 10 + 100,
        'Low': np.random.rand(500) * 10 + 100,
        'Close': np.random.rand(500) * 10 + 100,
        'Volume': np.random.rand(500) * 1000
    }
    df = pd.DataFrame(data, index=dates)
    
    # Add trend for ADX to pick up
    df['Close'] = df['Close'].rolling(window=10).mean().bfill()
    
    fe = FeatureEngine()
    try:
        df_features = fe.generate_features(df)
        print("\nFeatures generated successfully!")
        print(f"Shape: {df_features.shape}")
        print("\nColumns list:")
        print(list(df_features.columns))
        print("\nSample Data (Last 5 rows):")
        print(df_features[['Close', 'Efficiency_Ratio', 'Hour_Sin', 'ADX', 'ATR_Pct']].tail())
    except Exception as e:
        print(f"Feature generation failed: {e}")