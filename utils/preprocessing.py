import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ta.trend import SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def clean_data(self, df):
        """
        Removes missing values and resets index.
        """
        if df is None:
            return None
        df = df.dropna()
        df = df.reset_index()
        return df

    def add_technical_indicators(self, df):
        """
        Adds basic technical indicators: SMA, RSI, ADX, ATR.
        """
        # SMA
        df['SMA_14'] = SMAIndicator(close=df['Close'].squeeze(), window=14).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'].squeeze(), window=50).sma_indicator()
        
        # RSI
        df['RSI_14'] = RSIIndicator(close=df['Close'].squeeze(), window=14).rsi()

        # ADX
        adx_indicator = ADXIndicator(high=df['High'].squeeze(), low=df['Low'].squeeze(), close=df['Close'].squeeze(), window=14)
        df['ADX'] = adx_indicator.adx()

        # ATR
        atr_indicator = AverageTrueRange(high=df['High'].squeeze(), low=df['Low'].squeeze(), close=df['Close'].squeeze(), window=14)
        df['ATR'] = atr_indicator.average_true_range()
        
        # Drop NaN values created by indicators
        df = df.dropna()
        return df

    def normalize_data(self, df, feature_columns=['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_14', 'RSI_14']):
        """
        Normalizes the specified columns using MinMaxScaler.
        """
        if df.empty:
            print("Dataframe is empty, cannot normalize.")
            return df

        df_scaled = df.copy()
        df_scaled[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        
        return df_scaled


if __name__ == "__main__":
    # Test execution
    # Create a dummy dataframe for testing if no file exists
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = np.random.rand(100, 5) * 100
    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Volume'], index=dates)
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_indicators = preprocessor.add_technical_indicators(df_clean)
    df_normalized = preprocessor.normalize_data(df_indicators)
    
    print("Original Shape:", df.shape)
    print("Processed Shape:", df_normalized.shape)
    print("Head:\n", df_normalized.head())
