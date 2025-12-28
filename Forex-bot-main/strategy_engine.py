import pandas as pd
import numpy as np

class BaseStrategy:
    def __init__(self, data):
        self.data = data.copy()
        
    def generate_signals(self):
        """
        Logic to generate Buy/Sell signals.
        Must return the dataframe with a 'Signal' column.
        """
        raise NotImplementedError("Should implement generate_signals()")

class MovingAverageCrossover(BaseStrategy):
    def __init__(self, data, short_window=50, long_window=200):
        super().__init__(data)
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signals(self):
        print(f"Running MA Crossover Strategy ({self.short_window}/{self.long_window})...")
        
        # Calculate Moving Averages
        self.data['Short_MA'] = self.data['Close'].rolling(window=self.short_window).mean()
        self.data['Long_MA'] = self.data['Close'].rolling(window=self.long_window).mean()
        
        # Initialize Signal column
        self.data['Signal'] = 0.0
        
        # Generate Signal: 1 (Bullish) when Short > Long, -1 (Bearish) otherwise
        # This creates proper buy and sell signals
        self.data.loc[self.data.index[self.short_window:], 'Signal'] = \
            np.where(self.data['Short_MA'][self.short_window:] > self.data['Long_MA'][self.short_window:], 1.0, -1.0)
            
        # Generate Positions: Captures crossover moments
        # Position = 2 (buy signal: -1 to 1), -2 (sell signal: 1 to -1), 0 (no change)
        self.data['Position'] = self.data['Signal'].diff()
        
        return self.data

if __name__ == "__main__":
    # Test with dummy data if run directly
    print("This module defines the strategy logic.")
