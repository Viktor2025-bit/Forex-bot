"""
Risk Management Module for the AI Trading Bot.

Handles position sizing, stop-loss, take-profit, and overall risk controls.
"""

from dataclasses import dataclass
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ForexRiskParameters:
    """Configuration for forex-specific risk."""
    risk_per_trade_pct: float = 0.01   # Risk 1% of account per trade
    stop_loss_pips: int = 20
    risk_reward_ratio: float = 2.0

@dataclass
class RiskParameters:
    """Configuration for risk management."""
    max_position_size_pct: float = 0.1
    max_daily_loss_pct: float = 0.02
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_open_positions: int = 3
    min_confidence: float = 0.6
    forex_risk: Optional[ForexRiskParameters] = None


class RiskManager:
    """
    Manages trading risk and enforces safety limits.
    """
    
    def __init__(self, params: Optional[RiskParameters] = None):
        self.params = params or RiskParameters()
        if self.params.forex_risk is None:
            self.params.forex_risk = ForexRiskParameters()
            
        self.daily_pnl = 0.0
        self.open_positions = {}
        self.initial_portfolio_value = 0.0
        
    def set_portfolio_value(self, value: float):
        """Set the initial portfolio value for daily tracking."""
        self.initial_portfolio_value = value
        self.daily_pnl = 0.0
        
    def can_trade(self, portfolio_value: float) -> tuple[bool, str]:
        """Check if we're allowed to place new trades."""
        if self.initial_portfolio_value > 0:
            daily_return = (portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value
            if daily_return < -self.params.max_daily_loss_pct:
                return False, f"Daily loss limit reached ({daily_return:.2%})"
        
        if len(self.open_positions) >= self.params.max_open_positions:
            return False, f"Max positions reached ({self.params.max_open_positions})"
        
        return True, "OK"
    
    def should_enter_trade(self, prediction_confidence: float) -> tuple[bool, str]:
        """Check if prediction confidence is high enough to enter a trade."""
        if prediction_confidence < self.params.min_confidence:
            return False, f"Confidence too low ({prediction_confidence:.2%} < {self.params.min_confidence:.2%})"
        return True, "OK"
    
    def calculate_position_size(self, portfolio_value: float, current_price: float) -> int:
        """Calculate how many shares to buy (for stocks)."""
        max_position_value = portfolio_value * self.params.max_position_size_pct
        shares = int(max_position_value / current_price)
        return max(shares, 0)
        
    def calculate_forex_position_size(self, portfolio_value: float, symbol: str) -> int:
        """
        Calculate position size in lots for Forex.
        This assumes a standard pip value for simplicity. A more advanced version
        would calculate the precise pip value based on the quote currency.
        """
        # Simplified pip value assumption
        # For a standard lot (100,000 units) of a XXX/USD pair, one pip is ~$10.
        # We will trade in mini lots (10,000 units), so one pip is ~$1.
        pip_value_per_mini_lot = 1.0 

        # 1. Amount to risk in account currency (e.g., USD)
        risk_amount = portfolio_value * self.params.forex_risk.risk_per_trade_pct
        
        # 2. Value of stop loss in account currency
        stop_loss_value = self.params.forex_risk.stop_loss_pips * pip_value_per_mini_lot
        
        if stop_loss_value == 0:
            return 0
            
        # 3. Number of mini lots to trade
        num_mini_lots = int(risk_amount / stop_loss_value)
        
        # OANDA and MT5 expect units, not lots. 1 mini lot = 10,000 units.
        units = num_mini_lots * 10000
        
        logger.info(f"Forex Position Size: Risking ${risk_amount:.2f} to make {self.params.forex_risk.risk_reward_ratio * risk_amount:.2f}, SL: {self.params.forex_risk.stop_loss_pips} pips, Size: {units} units")
        return max(units, 0)
    
    def _get_pip_size(self, symbol: str) -> float:
        """Returns the pip size for a given symbol. JPY pairs are different."""
        return 0.0001 if 'JPY' not in symbol.upper() else 0.01

    def check_forex_stop_loss(self, entry_price: float, current_price: float, symbol: str, position_type: str) -> bool:
        """Check stop-loss based on pips for Forex."""
        pip_size = self._get_pip_size(symbol)
        
        if position_type == 'long':
            pips_lost = round((entry_price - current_price) / pip_size, 1)
        else: # short
            pips_lost = round((current_price - entry_price) / pip_size, 1)
            
        if pips_lost >= self.params.forex_risk.stop_loss_pips:
            logger.warning(f"FOREX STOP-LOSS triggered: {pips_lost:.1f} pips")
            return True
        return False
        
    def check_forex_take_profit(self, entry_price: float, current_price: float, symbol: str, position_type: str) -> bool:
        """Check take-profit based on pips and risk/reward for Forex."""
        pip_size = self._get_pip_size(symbol)
        take_profit_pips = self.params.forex_risk.stop_loss_pips * self.params.forex_risk.risk_reward_ratio

        if position_type == 'long':
            pips_gained = round((current_price - entry_price) / pip_size, 1)
        else: # short
            pips_gained = round((entry_price - current_price) / pip_size, 1)
            
        if pips_gained >= take_profit_pips:
            logger.info(f"FOREX TAKE-PROFIT triggered: {pips_gained:.1f} pips")
            return True
        return False

    def check_stop_loss(self, entry_price: float, current_price: float, position_type: str) -> bool:
        """Check if stop-loss should be triggered (for stocks)."""
        if position_type == 'long':
            loss_pct = (entry_price - current_price) / entry_price
        else:
            loss_pct = (current_price - entry_price) / entry_price
            
        if loss_pct >= self.params.stop_loss_pct:
            logger.warning(f"STOP-LOSS triggered: {loss_pct:.2%} loss")
            return True
        return False
    
    def check_take_profit(self, entry_price: float, current_price: float, position_type: str) -> bool:
        """Check if take-profit should be triggered (for stocks)."""
        if position_type == 'long':
            gain_pct = (current_price - entry_price) / entry_price
        else:
            gain_pct = (entry_price - current_price) / entry_price
            
        if gain_pct >= self.params.take_profit_pct:
            logger.info(f"TAKE-PROFIT triggered: {gain_pct:.2%} gain")
            return True
        return False
    
    def register_position(self, symbol: str, entry_price: float, qty: int, position_type: str):
        """Register a new open position."""
        self.open_positions[symbol] = {
            'entry_price': entry_price,
            'qty': qty,
            'type': position_type
        }
        logger.info(f"Registered position: {symbol} {position_type} {qty} units/shares @ ${entry_price:.4f}")
        
    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close a position and return the P&L."""
        if symbol not in self.open_positions:
            return 0.0
            
        pos = self.open_positions[symbol]
        
        # P&L calculation is the same for stocks and Forex units
        if pos['type'] == 'long':
            pnl = (exit_price - pos['entry_price']) * pos['qty']
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['qty']
            
        self.daily_pnl += pnl
        del self.open_positions[symbol]
        
        logger.info(f"Closed position: {symbol} @ ${exit_price:.4f}, P&L: ${pnl:.2f}")
        return pnl


if __name__ == "__main__":
    # Test the risk manager
    print("Testing Risk Manager...")
    
    # Stock risk
    rm_stock = RiskManager(RiskParameters())
    shares = rm_stock.calculate_position_size(10000, 150.0)
    print(f"Stock Position Size: {shares} shares")
    
    # Forex risk
    forex_params = ForexRiskParameters(risk_per_trade_pct=0.01, stop_loss_pips=50)
    rm_forex = RiskManager(RiskParameters(forex_risk=forex_params))
    
    units = rm_forex.calculate_forex_position_size(10000, "EUR/USD")
    print(f"Forex Position Size: {units} units for a $10k portfolio with 50 pip SL")
    
    # Test stop-loss
    sl_triggered = rm_forex.check_forex_stop_loss(1.1000, 1.0950, "EUR/USD", 'long')
    print(f"Forex SL at 50 pips: {sl_triggered}")
    
    # Test take-profit
    tp_triggered = rm_forex.check_forex_take_profit(1.1000, 1.1100, "EUR/USD", 'long')
    print(f"Forex TP at 100 pips (50 * 2.0 R:R): {tp_triggered}")
    
    print("Risk Manager test complete!")
