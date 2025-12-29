"""
Backtesting Engine for the AI Trading Bot.

This module simulates trading on historical data using model predictions
and calculates key performance metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    position: str  # 'long' or 'short'
    profit_loss: float
    return_pct: float


@dataclass
class BacktestResult:
    """Contains all backtesting results and metrics."""
    initial_capital: float
    final_capital: float
    total_return_pct: float
    num_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown_pct: float
    trades: List[Trade]
    equity_curve: pd.Series


class Backtester:
    """
    Simulates trading based on model predictions.
    """
    
    def __init__(self, initial_capital=10000, commission_pct=0.001):
        """
        Args:
            initial_capital (float): Starting capital in USD.
            commission_pct (float): Commission per trade (0.1% = 0.001).
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        
    def run_backtest(self, predictions, actual_prices, dates, threshold=0.5):
        """
        Run backtest simulation.
        
        Args:
            predictions (np.ndarray): Model predictions (probability of price going up).
            actual_prices (np.ndarray): Actual close prices for each prediction day.
            dates (pd.DatetimeIndex): Dates corresponding to predictions.
            threshold (float): Probability threshold for entering a trade.
            
        Returns:
            BacktestResult: Complete backtesting results.
        """
        capital = self.initial_capital
        trades = []
        equity = [capital]
        position = None  # None, 'long', or 'short'
        entry_price = 0
        entry_date = None
        
        for i in range(len(predictions) - 1):
            pred = float(predictions[i])  # Ensure scalar
            current_price = float(actual_prices[i])  # Ensure scalar
            next_price = float(actual_prices[i + 1])  # Ensure scalar
            current_date = dates[i]
            
            # Decision logic
            if position is None:
                # Not in a position - check if we should enter
                if pred > threshold:
                    # BUY signal - go long
                    position = 'long'
                    entry_price = current_price
                    entry_date = current_date
                    # Deduct commission
                    capital *= (1 - self.commission_pct)
                elif pred < (1 - threshold):
                    # SELL signal - go short (betting price will go down)
                    position = 'short'
                    entry_price = current_price
                    entry_date = current_date
                    capital *= (1 - self.commission_pct)
            else:
                # In a position - check if we should exit (next day we exit)
                exit_price = next_price
                exit_date = dates[i + 1]
                
                if position == 'long':
                    return_pct = (exit_price - entry_price) / entry_price
                else:  # short
                    return_pct = (entry_price - exit_price) / entry_price
                
                profit_loss = capital * return_pct
                capital += profit_loss
                capital *= (1 - self.commission_pct)  # Exit commission
                
                trades.append(Trade(
                    entry_date=str(entry_date),
                    exit_date=str(exit_date),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position=position,
                    profit_loss=profit_loss,
                    return_pct=return_pct * 100
                ))
                
                position = None
                
            equity.append(float(capital))  # Ensure scalar
        
        # Calculate metrics
        equity_series = pd.Series(equity)
        result = self._calculate_metrics(trades, equity_series)
        
        return result
    
    def _calculate_metrics(self, trades: List[Trade], equity: pd.Series) -> BacktestResult:
        """Calculate performance metrics."""
        
        final_capital = equity.iloc[-1]
        total_return_pct = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Win rate
        if len(trades) > 0:
            winning_trades = sum(1 for t in trades if t.profit_loss > 0)
            win_rate = winning_trades / len(trades)
        else:
            win_rate = 0.0
        
        # Sharpe Ratio (annualized, assuming daily returns)
        returns = equity.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Max Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown_pct = abs(drawdown.min()) * 100
        
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return_pct,
            num_trades=len(trades),
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            trades=trades,
            equity_curve=equity
        )
    
    def print_results(self, result: BacktestResult):
        """Pretty print backtesting results."""
        print("\n" + "=" * 50)
        print("BACKTESTING RESULTS")
        print("=" * 50)
        print(f"Initial Capital:    ${result.initial_capital:,.2f}")
        print(f"Final Capital:      ${result.final_capital:,.2f}")
        print(f"Total Return:       {result.total_return_pct:+.2f}%")
        print("-" * 50)
        print(f"Number of Trades:   {result.num_trades}")
        print(f"Win Rate:           {result.win_rate:.1%}")
        print(f"Sharpe Ratio:       {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown:       {result.max_drawdown_pct:.2f}%")
        print("=" * 50)


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing Backtester with dummy data...")
    
    # Simulate 100 days
    np.random.seed(42)
    predictions = np.random.random(100)  # Random predictions
    prices = 100 + np.cumsum(np.random.randn(100) * 2)  # Random walk prices
    dates = pd.date_range(start="2023-01-01", periods=100)
    
    backtester = Backtester(initial_capital=10000)
    result = backtester.run_backtest(predictions, prices, dates, threshold=0.6)
    backtester.print_results(result)
