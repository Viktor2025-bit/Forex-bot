"""
LSTM-based Trading Model for Price Direction Prediction.

This model takes a sequence of historical price data (with indicators) and
predicts whether the next period's close price will be HIGHER (1) or LOWER (0)
than the current close.
"""

import torch
import torch.nn as nn
import numpy as np

class TradingLSTM(nn.Module):
    """
    An LSTM model for binary classification (price UP or DOWN).
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Args:
            input_size (int): Number of input features (e.g., OHLCV + indicators).
            hidden_size (int): Number of LSTM hidden units.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout rate for regularization.
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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size).
            
        Returns:
            torch.Tensor: Prediction probability of shape (batch, 1).
        """
        # LSTM output
        lstm_out, _ = self.lstm(x)
        
        # Take the last time-step's output
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        prediction = self.fc(last_output)
        
        return prediction


def create_sequences(data, seq_length=30):
    """
    Creates sequences for LSTM training.
    
    Args:
        data (np.ndarray): Normalized feature array of shape (num_samples, num_features).
        seq_length (int): Number of past time steps to use for each prediction.
        
    Returns:
        X (np.ndarray): Input sequences of shape (num_sequences, seq_length, num_features).
        y (np.ndarray): Labels (1 if next close > current close, else 0).
    """
    X, y = [], []
    
    # Assuming 'Close' is at index 3 (after Open, High, Low)
    close_idx = 3
    
    for i in range(len(data) - seq_length - 1):
        # Input sequence
        X.append(data[i:i + seq_length])
        
        # Label: 1 if price goes up, 0 if down
        current_close = data[i + seq_length - 1, close_idx]
        next_close = data[i + seq_length, close_idx]
        y.append(1 if next_close > current_close else 0)
        
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Quick test of the model architecture
    print("Testing TradingLSTM model...")
    
    # Example: 7 features (OHLCV + SMA + RSI)
    model = TradingLSTM(input_size=7, hidden_size=64, num_layers=2)
    
    # Dummy input: batch of 16, sequence of 30 days, 7 features
    dummy_input = torch.randn(16, 30, 7)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output[:5].squeeze().detach().numpy()}")
    print("Model test passed!")
