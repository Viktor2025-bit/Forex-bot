"""
Risk Management Module for the AI Trading Bot.

Hangles position sizing, stop-loss, take-profit, and overall risk controls.
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
    stop_loss_atr_multiplier: float = 1.5  # Stop loss distance in multiples of ATR
    risk_reward_ratio: float = 2.0

@dataclass
class RiskParameters:
    """Configuration for risk management."""
    risk_per_trade_pct: float = 0.01  # Risk 1% of account per trade (for non-forex)
    max_daily_loss_pct: float = 0.02
    stop_loss_pct: float = 0.02 # Used for calculating non-forex SL price
    take_profit_pct: float = 0.04 # Used for calculating non-forex TP price
    max_open_positions: int = 3
    min_confidence: float = 0.6
    trailing_stop_pct: float = 0.015  # Trailing stop distance (1.5%)
    trailing_activation_pct: float = 0.01  # Activate trailing after 1% profit
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
        
        # Trailing Stop tracking: {symbol: {'best_price': float, 'trailing_stop': float}}
        self.trailing_stops = {}
        
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
    
    def calculate_position_size(self, portfolio_value: float, entry_price: float, stop_loss_price: float, position_type: str) -> float:
        """
        Calculate how many shares/units to buy based on fixed-risk.
        """
        risk_amount_per_trade = portfolio_value * self.params.risk_per_trade_pct
        
        if position_type == 'long':
            per_share_risk = entry_price - stop_loss_price
        else: # short
            per_share_risk = stop_loss_price - entry_price

        if per_share_risk <= 0:
            logger.warning(f"Cannot calculate position size: per-share risk is zero or negative (Entry: {entry_price}, SL: {stop_loss_price})")
            return 0.0

        shares = risk_amount_per_trade / per_share_risk
        logger.info(f"Stock Position Size: Risking ${risk_amount_per_trade:.2f}, SL: ${stop_loss_price:.2f}, Size: {shares:.2f} shares")
        return max(shares, 0.0)
        
    def calculate_forex_position_size(self, portfolio_value: float, symbol: str, atr: float) -> int:
        """
        Calculate position size in lots for Forex based on ATR.
        """
        pip_size = self._get_pip_size(symbol)
        pip_value_per_mini_lot = 1.0  # Simplified assumption for mini lots (10k units)

        # 1. Amount to risk in account currency (e.g., USD)
        risk_amount = portfolio_value * self.params.forex_risk.risk_per_trade_pct
        
        # 2. Stop loss distance in pips, based on ATR
        stop_loss_pips = (atr / pip_size) * self.params.forex_risk.stop_loss_atr_multiplier
        
        # 3. Value of stop loss in account currency
        stop_loss_value = stop_loss_pips * pip_value_per_mini_lot
        
        if stop_loss_value <= 0:
            logger.warning(f"[{symbol}] Invalid stop loss value ({stop_loss_value}), cannot calculate position size.")
            return 0
            
        # 4. Number of mini lots to trade
        num_mini_lots = int(risk_amount / stop_loss_value)
        
        # OANDA and MT5 expect units, not lots. 1 mini lot = 10,000 units.
        units = num_mini_lots * 10000
        
        logger.info(f"Forex Position Size: Risking ${risk_amount:.2f}, SL: {stop_loss_pips:.1f} pips (ATR Multiplier: {self.params.forex_risk.stop_loss_atr_multiplier}), Size: {units} units")
        return max(units, 0)
    
    def _get_pip_size(self, symbol: str) -> float:
        """Returns the pip size for a given symbol. JPY pairs are different."""
        return 0.0001 if 'JPY' not in symbol.upper() else 0.01

    def get_forex_exit_prices(self, entry_price: float, symbol: str, position_type: str, atr: float) -> dict:
        """
        Calculate stop-loss and take-profit prices based on ATR.
        """
        stop_distance = atr * self.params.forex_risk.stop_loss_atr_multiplier
        take_profit_distance = stop_distance * self.params.forex_risk.risk_reward_ratio

        if position_type == 'long':
            stop_loss_price = entry_price - stop_distance
            take_profit_price = entry_price + take_profit_distance
        else: # short
            stop_loss_price = entry_price + stop_distance
            take_profit_price = entry_price - take_profit_distance
            
        return {'stop_loss': stop_loss_price, 'take_profit': take_profit_price}

    def get_stock_exit_prices(self, entry_price: float, position_type: str) -> dict:
        """
        Calculate stop-loss and take-profit prices for stocks based on percentages.
        """
        if position_type == 'long':
            stop_loss_price = entry_price * (1 - self.params.stop_loss_pct)
            take_profit_price = entry_price * (1 + self.params.take_profit_pct)
        else: # short
            stop_loss_price = entry_price * (1 + self.params.stop_loss_pct)
            take_profit_price = entry_price * (1 - self.params.take_profit_pct)
        
        return {'stop_loss': stop_loss_price, 'take_profit': take_profit_price}

    def register_position(self, symbol: str, entry_price: float, qty: int, position_type: str, stop_loss_price: float = None):
        """
        Register a new open position and initialize trailing stop.
        The initial stop_level for the trailing stop is now the calculated SL price.
        """
        self.open_positions[symbol] = {
            'entry_price': entry_price,
            'qty': qty,
            'type': position_type,
            'stop_loss_price': stop_loss_price
        }
        
        # Determine initial stop level for trailing stop
        initial_stop = stop_loss_price
        if initial_stop is None: # Fallback for non-forex trades (should be rare now)
            initial_stop = entry_price * (1 - self.params.stop_loss_pct) if position_type == 'long' \
                           else entry_price * (1 + self.params.stop_loss_pct)

        self.trailing_stops[symbol] = {
            'best_price': entry_price,
            'trailing_active': False,
            'stop_level': initial_stop
        }
        logger.info(f"Registered position: {symbol} {position_type} {qty} units @ ${entry_price:.4f} with initial SL @ ${initial_stop:.4f}")
        
    def update_trailing_stop(self, symbol: str, current_price: float) -> dict:
        """
        Update trailing stop based on current price movement.
        Returns dict with 'triggered': bool and 'stop_level': float
        """
        if symbol not in self.open_positions or symbol not in self.trailing_stops:
            return {'triggered': False, 'stop_level': 0}
            
        pos = self.open_positions[symbol]
        ts = self.trailing_stops[symbol]
        entry = pos['entry_price']
        
        if pos['type'] == 'long':
            # For longs: track highest price, stop trails below
            profit_pct = (current_price - entry) / entry
            
            # Check if trailing should activate
            if profit_pct >= self.params.trailing_activation_pct:
                ts['trailing_active'] = True
                
            # Update best price if we have a new high
            if current_price > ts['best_price']:
                ts['best_price'] = current_price
                if ts['trailing_active']:
                    # Move stop up (but never down)
                    new_stop = current_price * (1 - self.params.trailing_stop_pct)
                    ts['stop_level'] = max(ts['stop_level'], new_stop)
                    logger.info(f"[{symbol}] Trailing Stop Updated: ${ts['stop_level']:.4f} (locked {((ts['stop_level']/entry)-1)*100:.2f}% profit)")
            
            # Check if stop is hit
            if current_price <= ts['stop_level']:
                logger.warning(f"[{symbol}] TRAILING STOP HIT @ ${current_price:.4f} (Stop: ${ts['stop_level']:.4f})")
                return {'triggered': True, 'stop_level': ts['stop_level']}
                
        else:  # short
            # For shorts: track lowest price, stop trails above
            profit_pct = (entry - current_price) / entry
            
            if profit_pct >= self.params.trailing_activation_pct:
                ts['trailing_active'] = True
                
            if current_price < ts['best_price']:
                ts['best_price'] = current_price
                if ts['trailing_active']:
                    new_stop = current_price * (1 + self.params.trailing_stop_pct)
                    ts['stop_level'] = min(ts['stop_level'], new_stop)
                    logger.info(f"[{symbol}] Trailing Stop Updated: ${ts['stop_level']:.4f}")
            
            if current_price >= ts['stop_level']:
                logger.warning(f"[{symbol}] TRAILING STOP HIT @ ${current_price:.4f}")
                return {'triggered': True, 'stop_level': ts['stop_level']}
        
        return {'triggered': False, 'stop_level': ts['stop_level']}
    
    def check_trailing_stop(self, symbol: str, current_price: float) -> bool:
        """Convenience method: returns True if trailing stop is triggered."""
        result = self.update_trailing_stop(symbol, current_price)
        return result['triggered']
        
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
        
        # Clean up trailing stop tracking
        if symbol in self.trailing_stops:
            del self.trailing_stops[symbol]
        
        logger.info(f"Closed position: {symbol} @ ${exit_price:.4f}, P&L: ${pnl:.2f}")
        return pnl
    
    def check_daily_loss(self, current_daily_pnl, initial_capital=None):
        """
        Check if trading should continue based on daily loss limit.
        """
        reference_capital = self.initial_portfolio_value if self.initial_portfolio_value > 0 else (initial_capital or 1000)
        max_loss_amount = reference_capital * self.params.max_daily_loss_pct
        
        if current_daily_pnl < -max_loss_amount:
            logger.warning(f"Daily loss limit hit: ${current_daily_pnl:.2f} exceeds -${max_loss_amount:.2f}")
            return False
            
        return True


if __name__ == "__main__":
    # Test the risk manager
    print("Testing Risk Manager...")
    
    # Stock risk (new method)
    stock_params = RiskParameters(risk_per_trade_pct=0.01, stop_loss_pct=0.05) # Risk 1%, 5% SL
    rm_stock = RiskManager(stock_params)
    portfolio_val = 50000
    entry_price = 150.0
    
    stock_exits = rm_stock.get_stock_exit_prices(entry_price, 'long')
    sl_price = stock_exits['stop_loss']
    
    shares = rm_stock.calculate_position_size(portfolio_val, entry_price, sl_price, 'long')
    print(f"Stock Position Size: {shares:.2f} shares for a ${portfolio_val} portfolio")
    print(f"Risking ${portfolio_val * 0.01:.2f} with SL at ${sl_price:.2f}")

    # Forex risk with ATR
    forex_params = ForexRiskParameters(risk_per_trade_pct=0.01, stop_loss_atr_multiplier=2.0)
    rm_forex = RiskManager(RiskParameters(forex_risk=forex_params))
    
    # Example values
    atr_val = 0.0050 
    forex_entry = 1.1000
    symbol = "EUR/USD"

    units = rm_forex.calculate_forex_position_size(portfolio_val, symbol, atr=atr_val)
    print(f"\nForex Position Size (ATR based): {units} units for a ${portfolio_val} portfolio")
    
    # Test exit price calculation
    exit_prices = rm_forex.get_forex_exit_prices(forex_entry, symbol, 'long', atr=atr_val)
    print(f"Forex Exit prices for LONG: SL={exit_prices['stop_loss']:.4f}, TP={exit_prices['take_profit']:.4f}")
    
    exit_prices_short = rm_forex.get_forex_exit_prices(forex_entry, symbol, 'short', atr=atr_val)
    print(f"Forex Exit prices for SHORT: SL={exit_prices_short['stop_loss']:.4f}, TP={exit_prices_short['take_profit']:.4f}")
    
    print("\nRisk Manager test complete!")
