"""
OANDA Order Executor for live and paper trading.
Requires 'oandapyV20' package.
"""

import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions
import logging
from .order_executor import OrderExecutor, Order, OrderSide, OrderStatus

logger = logging.getLogger(__name__)

class OandaExecutor(OrderExecutor):
    """
    Executor implementation for OANDA broker.
    Supports both Practice (Paper) and Live environments.
    """
    
    def __init__(self, access_token: str, account_id: str, environment: str = "practice"):
        """
        Args:
            access_token: API Access Token from OANDA
            account_id: Account ID (e.g. "101-001-...")
            environment: "practice" or "live"
        """
        super().__init__()
        self.account_id = account_id
        self.client = oandapyV20.API(access_token=access_token, environment=environment)
        
        # Test connection
        try:
            self.get_account()
            logger.info("Successfully connected to OANDA")
        except Exception as e:
            logger.error(f"Failed to connect to OANDA: {e}")

    def submit_order(self, symbol: str, side: OrderSide, quantity: float, 
                    order_type: str = "MARKET", price: float = None) -> Order:
        """
        Submit an order to OANDA.
        Note: OANDA quantities are in 'units', not lots.
        """
        # Convert symbol (e.g., "EURUSD=X" -> "EUR_USD")
        instrument = self._format_symbol(symbol)
        qty = int(quantity) if side == OrderSide.BUY else -int(quantity)
        
        order_body = {
            "order": {
                "instrument": instrument,
                "units": str(qty),
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }
        
        try:
            r = orders.OrderCreate(self.account_id, data=order_body)
            self.client.request(r)
            
            response = r.response
            if 'orderFillTransaction' in response:
                fill = response['orderFillTransaction']
                return Order(
                    order_id=fill['orderID'],
                    symbol=symbol,
                    side=side,
                    quantity=abs(int(fill['units'])),
                    order_type=order_type,
                    status=OrderStatus.FILLED,
                    filled_price=float(fill['price'])
                )
            else:
                logger.warning(f"Order not immediately filled: {response}")
                return Order(
                    order_id=response.get('orderCreateTransaction', {}).get('id'),
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    status=OrderStatus.PENDING
                )
                
        except Exception as e:
            logger.error(f"Failed to submit OANDA order: {e}")
            return Order(order_id="", symbol=symbol, side=side, quantity=quantity, status=OrderStatus.FAILED)

    def get_position(self, symbol: str) -> dict:
        """Get current position for a symbol."""
        instrument = self._format_symbol(symbol)
        
        try:
            r = positions.PositionDetails(self.account_id, instrument=instrument)
            self.client.request(r)
            
            p = r.response.get('position', {})
            long_units = int(p.get('long', {}).get('units', 0))
            short_units = int(p.get('short', {}).get('units', 0))
            
            net_units = long_units + short_units  # Short units are negative in OANDA API? Check docs.
            # OANDA separates long and short. Usually for net position we sum them.
            
            return {
                'symbol': symbol,
                'qty': abs(net_units),
                'market_value': 0.0, # Need price to calc
                'current_price': 0.0,
                'unrealized_pl': float(p.get('unrealizedPL', 0.0))
            }
            
        except Exception as e:
            # If no position exists, OANDA might raise error or return empty
            return {'symbol': symbol, 'qty': 0, 'market_value': 0.0}

    def get_account(self) -> dict:
        """Get account summary."""
        try:
            r = accounts.AccountSummary(self.account_id)
            self.client.request(r)
            
            summary = r.response.get('account', {})
            return {
                'cash': float(summary.get('balance', 0.0)),
                'total_equity': float(summary.get('NAV', 0.0)),
                'buying_power': float(summary.get('marginAvailable', 0.0))
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {'cash': 0.0, 'total_equity': 0.0}

    def _format_symbol(self, symbol: str) -> str:
        """
        Convert standard symbol to OANDA format.
        Ex: 'EURUSD=X' -> 'EUR_USD'
        """
        s = symbol.replace('=X', '').upper()
        if '_' not in s and len(s) == 6:
            return f"{s[:3]}_{s[3:]}"
        return s
