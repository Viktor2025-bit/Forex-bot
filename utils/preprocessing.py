import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
        Adds basic technical indicators: SMA, RSI.
        """
        # Simple Moving Average (14 period)
        df['SMA_14'] = df['Close'].rolling(window=14).mean()
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
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
