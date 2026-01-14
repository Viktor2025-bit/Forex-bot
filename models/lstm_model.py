"""
LSTM-based Trading Model for Price Prediction (Regression).

This model takes a sequence of historical price data (with indicators) and
predicts the value of a target feature for the next time step.
"""

import torch
import torch.nn as nn
import numpy as np

class TradingLSTM(nn.Module):
    """
    An LSTM model for time-series regression.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        """
        Args:
            input_size (int): Number of input features (e.g., OHLCV + indicators).
            hidden_size (int): Number of LSTM hidden units.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout rate for regularization.
            output_size (int): The number of output values (usually 1 for regression).
        """
        super(TradingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size).
            
        Returns:
            torch.Tensor: Prediction of shape (batch, output_size).
        """
        # LSTM output
        lstm_out, _ = self.lstm(x)
        
        # Take the last time-step's output
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        prediction = self.fc(last_output)
        
        return prediction


def create_sequences(data, seq_length=30, target_col_index=3):
    """
    Creates sequences for LSTM training (regression).
    
    Args:
        data (np.ndarray): Normalized feature array of shape (num_samples, num_features).
        seq_length (int): Number of past time steps to use for each prediction.
        target_col_index (int): The index of the column we want to predict.
        
    Returns:
        X (np.ndarray): Input sequences of shape (num_sequences, seq_length, num_features).
        y (np.ndarray): Labels (the value of the target column at the next time step).
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # Input sequence
        X.append(data[i:i + seq_length])
        
        # Label: The value of the target column at the next time step
        y.append(data[i + seq_length, target_col_index])
        
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Quick test of the model architecture
    print("Testing TradingLSTM model for regression...")
    
    # Example: 7 features (OHLCV + SMA + RSI)
    model = TradingLSTM(input_size=7, hidden_size=64, num_layers=2, output_size=1)
    
    # Dummy input: batch of 16, sequence of 30 days, 7 features
    dummy_input = torch.randn(16, 30, 7)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output[:5].squeeze().detach().numpy()}")
    print("Model test passed!")
