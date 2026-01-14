import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, AroonIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, AwesomeOscillatorIndicator
from ta.volatility import BollingerBands, AverageTrueRange

class FeatureEngine:
    """
    Generates advanced technical indicators and custom features for the AI model.
    Focuses on Forex-specific patterns: Sessions, Trend Efficiency, and Volatility.
    """
    
    def __init__(self, ticker=""):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_path = f"models/{ticker}_scaler.gz"
        self.ticker = ticker

    def normalize_data(self, df, fit_scaler=False):
        """
        Normalizes the feature columns using MinMaxScaler.
        If fit_scaler is True, it fits the scaler to the data and saves it.
        Otherwise, it loads and uses the existing scaler.
        """
        df_scaled = df.copy()
        
        # Identify feature columns (all columns except non-numeric/identifiers)
        feature_columns = [col for col in df.columns if df[col].dtype in [np.int64, np.float64]]

        if fit_scaler:
            print("Fitting scaler and transforming data...")
            df_scaled[feature_columns] = self.scaler.fit_transform(df[feature_columns])
            joblib.dump(self.scaler, self.scaler_path)
            print(f"Scaler saved to {self.scaler_path}")
        else:
            print("Loading existing scaler and transforming data...")
            try:
                scaler = joblib.load(self.scaler_path)
                df_scaled[feature_columns] = scaler.transform(df[feature_columns])
            except FileNotFoundError:
                raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}. Please fit the scaler first.")

        return df_scaled

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
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Standardize column names
        data.columns = [c.capitalize() for c in data.columns]
        
        # Ensure all required columns exist and are 1D Series
        for col in required_cols:
            if col not in data.columns:
                 raise ValueError(f"Dataframe missing column: {col}")
            # Force squeeze if 2D (e.g. shape (N, 1))
            if isinstance(data[col], pd.DataFrame):
                data[col] = data[col].iloc[:, 0]

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
        data['ATR_Pct'] = data['ATR'] / data['Close']

        # --- 5. Institutional Indicators (NEW) ---
        
        # VWAP (Volume Weighted Average Price) - Intraday mainly, but valid for sessions
        # Approximation: Cumulative(Price * Volume) / Cumulative(Volume)
        # We reset on daily basis roughly by using a large rolling window or just cumulative for the dataset if short
        v = data['Volume'].replace(0, 1) # Avoid div by zero
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        data['VWAP'] = (tp * v).cumsum() / v.cumsum()
        data['Dist_VWAP'] = (data['Close'] - data['VWAP']) / data['VWAP']

        # SuperTrend
        # Basic implementation: (High + Low) / 2 + Multiplier * ATR
        atr_period = 10
        atr_multiplier = 3.0
        hl2 = (data['High'] + data['Low']) / 2
        # Note: True SuperTrend requires recursive calculation, using a simplified vectorial proxy here
        # or we just use the bands as features
        data['SuperTrend_Upper'] = hl2 + (atr_multiplier * data['ATR'])
        data['SuperTrend_Lower'] = hl2 - (atr_multiplier * data['ATR'])
        # Feature: Position relative to bands
        data['Dist_ST_Upper'] = (data['Close'] - data['SuperTrend_Upper']) / data['Close'] 
        data['Dist_ST_Lower'] = (data['Close'] - data['SuperTrend_Lower']) / data['Close']

        # Pivot Points (Classic)
        # PP = (H + L + C) / 3
        # R1 = 2*PP - L
        # S1 = 2*PP - H
        pp = (data['High'] + data['Low'] + data['Close']) / 3
        data['Pivot_Point'] = pp
        data['Pivot_R1'] = (2 * pp) - data['Low']
        data['Pivot_S1'] = (2 * pp) - data['High']
        data['Dist_Pivot'] = (data['Close'] - data['Pivot_Point']) / data['Close']

        # --- 6. Price Action ---
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        
        # --- 7. Time Features (NEW - Fix for Input Size 57) ---
        # We need one more feature to reach 57 (currently 56)
        # Adding normalized Day of Week (0-1)
        # data['Day_of_Week'] = data.index.dayofweek / 6.0  # DISABLED - Not in training data

        
        # Lagged Returns
        for lag in [1, 2, 3, 5]:
            data[f'Return_Lag_{lag}'] = data['Log_Return'].shift(lag)
            
        # Volume
        if 'Volume' in data.columns: 
             # Safe pct change
             data['Volume_Change'] = data['Volume'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        else:
             data['Volume_Change'] = 0

        # --- 7. Advanced Indicators for Synthetics (R_75 Optimized) ---
        
        # Aroon Oscillator - Trend strength detection
        aroon = AroonIndicator(high=data['High'], low=data['Low'], window=25)
        data['Aroon_Up'] = aroon.aroon_up()
        data['Aroon_Down'] = aroon.aroon_down()
        data['Aroon_Osc'] = data['Aroon_Up'] - data['Aroon_Down']
        
        # Awesome Oscillator - Momentum on clean synthetic patterns
        ao = AwesomeOscillatorIndicator(high=data['High'], low=data['Low'], window1=5, window2=34)
        data['AO'] = ao.awesome_oscillator()
        
        # Historical Volatility (Annualized) - Detect deviations from R_75's baseline 75% vol
        data['Hist_Vol'] = data['Log_Return'].rolling(window=20).std() * np.sqrt(252)
        
        # Relative Volatility Index (RVI) - RSI applied to volatility
        change = data['Close'].diff()
        std_up = change.where(change > 0, 0).rolling(14).std()
        std_down = change.where(change < 0, 0).abs().rolling(14).std()
        data['RVI'] = 100 * std_up / (std_up + std_down + 1e-10)  # Avoid div by zero
        
        # Williams Fractals - Clear reversal points on algorithmic data
        # Fractal Up: Current high is highest in 5-bar window
        # Fractal Down: Current low is lowest in 5-bar window
        data['Fractal_Up'] = (
            (data['High'] > data['High'].shift(1)) & 
            (data['High'] > data['High'].shift(2)) &
            (data['High'] > data['High'].shift(-1)) & 
            (data['High'] > data['High'].shift(-2))
        ).astype(int)
        
        data['Fractal_Down'] = (
            (data['Low'] < data['Low'].shift(1)) & 
            (data['Low'] < data['Low'].shift(2)) &
            (data['Low'] < data['Low'].shift(-1)) & 
            (data['Low'] < data['Low'].shift(-2))
        ).astype(int)
        
        # Distance to last fractal (useful feature for LSTM)
        data['Bars_Since_Fractal_Up'] = data['Fractal_Up'].groupby((data['Fractal_Up'] != data['Fractal_Up'].shift()).cumsum()).cumcount()
        data['Bars_Since_Fractal_Down'] = data['Fractal_Down'].groupby((data['Fractal_Down'] != data['Fractal_Down'].shift()).cumsum()).cumcount()
        
        # Enhanced Bollinger Bands features
        bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Lower'] = bb.bollinger_lband()
        data['BB_Mid'] = bb.bollinger_mavg()
        data['BB_PctB'] = bb.bollinger_pband()  # %B indicator (0-1 scale)

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