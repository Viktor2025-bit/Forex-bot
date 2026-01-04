import numpy as np
import pandas as pd

class RiskManager:
    """
    Risk Management System for Forex Trading Bot.
    
    Manages position sizing, stop-loss, take-profit, and drawdown protection
    to ensure controlled risk exposure on every trade.
    
    Key Features:
    - Position sizing based on risk percentage
    - Automatic stop-loss calculation
    - Automatic take-profit calculation (risk-reward ratio)
    - Maximum drawdown protection
    """
    
    def __init__(self, 
                 risk_per_trade=2.0, 
                 stop_loss_pct=2.0, 
                 risk_reward_ratio=2.0,
                 max_drawdown_pct=15.0):
        """
        Initialize the Risk Manager.
            
        Args:
            risk_per_trade (float): Percentage of capital to risk per trade (default: 2%)
            stop_loss_pct (float): Stop-loss distance as percentage from entry (default: 2%)
            risk_reward_ratio (float): Ratio of profit target to risk (default: 2.0)
            max_drawdown_pct (float): Maximum drawdown before halting trading (default: 15%)
        
        Example:
            With $500 account and 2% risk:
            - Risk per trade = $10 maximum loss
            - Stop-loss at 2% from entry
            - Take-profit at 4% from entry (2:1 ratio)
            - Trading halts if account drops to $425 (15% drawdown)
        """
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.risk_reward_ratio = risk_reward_ratio
        self.max_drawdown_pct = max_drawdown_pct
        
        # Track peak account value for drawdown calculation
        self.peak_balance = 0
        self.trading_enabled = True
        
    def calculate_position_size(self, account_balance, entry_price):
        """
        Calculate safe position size based on risk parameters.
        
        This ensures that if stop-loss is hit, we only lose the predetermined
        risk amount (e.g., 2% of account).
        
        Args:
            account_balance (float): Current account balance
            entry_price (float): Price at which we're entering the trade
            
        Returns:
            float: Position size (number of units to trade)
            
        Example:
            Account: $500, Risk: 2% = $10, Entry: 1.1000, Stop: 2%
            Risk per unit: 1.1000 * 0.02 = 0.022
            Position Size: $10 / 0.022 = 454 units
        """
        # Calculate maximum dollar amount we're willing to lose
        max_risk_amount = account_balance * (self.risk_per_trade / 100)
        
        # Calculate risk per unit (how much we lose per unit if stop-loss hits)
        risk_per_unit = entry_price * (self.stop_loss_pct / 100)
        
        # Position size = How much we can risk / Risk per unit
        position_size = max_risk_amount / risk_per_unit
        
        return position_size
    
    def calculate_stop_loss(self, entry_price, direction):
        """
        Calculate stop-loss price based on entry price and direction.
        
        Args:
            entry_price (float): Price at which trade was entered
            direction (str): 'LONG' or 'SHORT'
            
        Returns:
            float: Stop-loss price
            
        Example:
            Entry: 1.1000, Direction: LONG, Stop-loss: 2%
            Stop-loss price = 1.1000 * (1 - 0.02) = 1.0780
        """
        if direction == 'LONG':
            # For long positions, stop-loss is below entry price
            stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
        else:
            # For short positions, stop-loss is above entry price
            stop_loss = entry_price * (1 + self.stop_loss_pct / 100)
            
        return stop_loss
    
    def calculate_take_profit(self, entry_price, direction):
        """
        Calculate take-profit price based on risk-reward ratio.
        
        Args:
            entry_price (float): Price at which trade was entered
            direction (str): 'LONG' or 'SHORT'
            
        Returns:
            float: Take-profit price
            
        Example:
            Entry: 1.1000, Direction: LONG, Stop-loss: 2%, Risk-Reward: 2.0
            Take-profit distance = 2% * 2.0 = 4%
            Take-profit price = 1.1000 * (1 + 0.04) = 1.1440
        """
        # Take-profit distance is stop-loss distance multiplied by risk-reward ratio
        take_profit_pct = self.stop_loss_pct * self.risk_reward_ratio
        
        if direction == 'LONG':
            # For long positions, take-profit is above entry price
            take_profit = entry_price * (1 + take_profit_pct / 100)
        else:
            # For short positions, take-profit is below entry price
            take_profit = entry_price * (1 - take_profit_pct / 100)
            
        return take_profit
    
    def check_stop_loss_hit(self, current_price, stop_loss, direction):
        """
        Check if stop-loss has been triggered.
        
        Args:
            current_price (float): Current market price
            stop_loss (float): Stop-loss price
            direction (str): 'LONG' or 'SHORT'
            
        Returns:
            bool: True if stop-loss hit, False otherwise
        """
        if direction == 'LONG':
            # Long position: stop-loss hit if price drops below stop-loss
            return current_price <= stop_loss
        else:
            # Short position: stop-loss hit if price rises above stop-loss
            return current_price >= stop_loss
    
    def check_take_profit_hit(self, current_price, take_profit, direction):
        """
        Check if take-profit has been triggered.
        
        Args:
            current_price (float): Current market price
            take_profit (float): Take-profit price
            direction (str): 'LONG' or 'SHORT'
            
        Returns:
            bool: True if take-profit hit, False otherwise
        """
        if direction == 'LONG':
            # Long position: take-profit hit if price rises above take-profit
            return current_price >= take_profit
        else:
            # Short position: take-profit hit if price drops below take-profit
            return current_price <= take_profit
    
    def check_drawdown(self, current_balance, initial_capital):
        """
        Check if maximum drawdown limit has been exceeded.
        
        This is a circuit breaker to protect your account from catastrophic losses.
        If drawdown exceeds the limit, trading is halted.
        
        Args:
            current_balance (float): Current account balance
            initial_capital (float): Starting capital
            
        Returns:
            tuple: (is_safe: bool, current_drawdown_pct: float)
            
        Example:
            Initial: $500, Current: $450, Max Drawdown: 15%
            Drawdown = (500 - 450) / 500 = 10% -> Still safe
            
            Initial: $500, Current: $400, Max Drawdown: 15%
            Drawdown = (500 - 400) / 500 = 20% -> HALT TRADING
        """
        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calculate drawdown from peak
        if self.peak_balance > 0:
            drawdown_pct = ((self.peak_balance - current_balance) / self.peak_balance) * 100
        else:
            drawdown_pct = 0
        
        # Check if we've exceeded max drawdown
        if drawdown_pct >= self.max_drawdown_pct:
            self.trading_enabled = False
            
        return self.trading_enabled, drawdown_pct
    
    def get_risk_metrics(self):
        """
        Get current risk management configuration.
        
        Returns:
            dict: Dictionary of risk parameters
        """
        return {
            'risk_per_trade_pct': self.risk_per_trade,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.stop_loss_pct * self.risk_reward_ratio,
            'risk_reward_ratio': self.risk_reward_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'trading_enabled': self.trading_enabled
        }
    
    def reset_trading(self):
        """Reset trading status (useful for new backtest runs)."""
        self.trading_enabled = True
        self.peak_balance = 0

    def check_daily_loss(self, current_daily_pnl, initial_capital=1000):
        """
        Check if the daily loss limit has been hit.
        
        Args:
            current_daily_pnl (float): Net PnL for the day (negative = loss)
            initial_capital (float): Starting capital for reference (default 1000 or config)
            
        Returns:
            bool: True if safe to trade, False if limit hit.
        """
        # E.g. max_daily_loss_pct = 2.0 (2%)
        # If capital = 1000, max loss = 20.
        # If PnL is -25, we return False (Stop Trading).
        
        # Default to 2% if not set (though bot sets it)
        limit_pct = getattr(self, 'max_daily_loss_pct', 2.0)
        max_loss_amount = initial_capital * (limit_pct / 100.0)
        
        # Check against negative PnL
        if current_daily_pnl < -max_loss_amount:
            return False
            
        return True


if __name__ == "__main__":
    # Test the Risk Manager with example scenarios
    print("=== Risk Manager Test ===\n")
    
    rm = RiskManager(risk_per_trade=2.0, stop_loss_pct=2.0, risk_reward_ratio=2.0)
    
    # Scenario 1: Calculate position size
    account = 500
    entry = 1.1000
    position_size = rm.calculate_position_size(account, entry)
    print(f"Account: ${account}, Entry: {entry}")
    print(f"Position Size: {position_size:.2f} units")
    print(f"Max Risk: ${account * 0.02:.2f}\n")
    
    # Scenario 2: Calculate stop-loss and take-profit for LONG
    direction = "LONG"
    stop_loss = rm.calculate_stop_loss(entry, direction)
    take_profit = rm.calculate_take_profit(entry, direction)
    print(f"Direction: {direction}")
    print(f"Entry: {entry}")
    print(f"Stop-Loss: {stop_loss:.4f} ({rm.stop_loss_pct}% away)")
    print(f"Take-Profit: {take_profit:.4f} ({rm.stop_loss_pct * rm.risk_reward_ratio}% away)")
    print(f"Risk-Reward Ratio: 1:{rm.risk_reward_ratio}\n")
    
    # Scenario 3: Test drawdown protection
    print("Drawdown Protection Test:")
    rm.peak_balance = 500
    safe, dd = rm.check_drawdown(450, 500)
    print(f"Balance: $450, Drawdown: {dd:.2f}%, Trading Enabled: {safe}")
    
    safe, dd = rm.check_drawdown(400, 500)
    print(f"Balance: $400, Drawdown: {dd:.2f}%, Trading Enabled: {safe}")
