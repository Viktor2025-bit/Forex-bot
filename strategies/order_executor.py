"""
Order Execution Module for the AI Trading Bot.

Handles order creation, submission, and tracking.
Supports both paper trading (simulation) and live trading (Alpaca API).
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: str  # 'market', 'limit'
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_at: Optional[datetime] = None


class OrderExecutor(ABC):
    """Abstract base class for order execution."""

    @abstractmethod
    def submit_order(self, symbol: str, side: OrderSide, quantity: int,
                     order_type: str = "market", limit_price: Optional[float] = None,
                     stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Order:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> dict:
        pass
    
    @abstractmethod
    def get_account(self) -> dict:
        pass

    @abstractmethod
    def get_open_positions(self) -> list:
        pass


class PaperTradingExecutor(OrderExecutor):
    """
    Paper trading executor for testing without real money.
    Simulates order execution instantly at current price.
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.paper_trading = True # Interface compatibility
        self.positions = {}  # symbol -> {qty, avg_price, sl, tp}
        self.orders = {}
        self.order_counter = 0
        self.current_prices = {}  # symbol -> price (must be updated externally)
        
    def set_price(self, symbol: str, price: float):
        """Update the current price for a symbol and check for SL/TP triggers."""
        self.current_prices[symbol] = price
        self._check_triggers(symbol, price)
        
    def _check_triggers(self, symbol: str, price: float):
        """Check for stop-loss or take-profit triggers for a given symbol."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        if pos['qty'] == 0:
            return

        triggered = False
        if pos.get('stop_loss') and price <= pos['stop_loss']:
            logger.info(f"[PAPER] Stop-loss triggered for {symbol} at ${price:.2f}")
            triggered = True
        elif pos.get('take_profit') and price >= pos['take_profit']:
            logger.info(f"[PAPER] Take-profit triggered for {symbol} at ${price:.2f}")
            triggered = True
            
        if triggered:
            self.submit_order(symbol, OrderSide.SELL, pos['qty'], stop_loss=None, take_profit=None)

    def submit_order(self, symbol: str, side: OrderSide, quantity: int,
                     order_type: str = "market", limit_price: Optional[float] = None,
                     stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Order:
        """Submit a paper order (executed immediately for market orders)."""
        
        self.order_counter += 1
        order_id = f"PAPER-{self.order_counter}"
        
        # Get current price
        price = self.current_prices.get(symbol, limit_price or 100.0)
        
        # Create order
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Execute immediately for market orders (paper trading)
        if order_type == "market":
            success = self._execute_order(order, price)
            if success:
                order.status = OrderStatus.FILLED
                order.filled_price = price
                order.filled_at = datetime.now()
                # Don't log fills for trigger-based sells to avoid confusion
                if "triggered" not in order.id:
                    logger.info(f"[PAPER] Order filled: {side.value} {quantity} {symbol} @ ${price:.2f}")
            else:
                order.status = OrderStatus.REJECTED
                logger.warning(f"[PAPER] Order rejected: Insufficient funds/shares")
        
        self.orders[order_id] = order
        return order
    
    def _execute_order(self, order: Order, price: float) -> bool:
        """Execute the order and update positions/cash."""
        total_cost = price * order.quantity
        
        # Simple cash model: Buying cost money, Selling gives money
        if order.side == OrderSide.BUY:
            if self.cash < total_cost:
                return False
            self.cash -= total_cost
        else:
            self.cash += total_cost
            
        # Update Position
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = {'qty': 0, 'avg_price': 0, 'stop_loss': None, 'take_profit': None}
            
        pos = self.positions[symbol]
        current_qty = pos.get('qty', 0)
        signed_order_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        new_qty = current_qty + signed_order_qty
        
        # Update Avg Price
        # Case 1: Increasing position (Long->More Long, Short->More Short, Flat->Any)
        increasing = (current_qty == 0) or (current_qty > 0 and signed_order_qty > 0) or (current_qty < 0 and signed_order_qty < 0)
        
        # Case 2: Flipping position (Long->Short, Short->Long)
        flipping = (current_qty > 0 and new_qty < 0) or (current_qty < 0 and new_qty > 0)
        
        if increasing:
            total_current_val = abs(current_qty) * pos.get('avg_price', 0)
            total_order_val = order.quantity * price
            if abs(new_qty) > 0:
                pos['avg_price'] = (total_current_val + total_order_val) / abs(new_qty)
            else:
                 pos['avg_price'] = 0
                 
        elif flipping:
            # If flipping, the avg price becomes the new entry price for the remainder
            pos['avg_price'] = price
            
        # If decreasing but not flipping, avg_price stays same (realizing PnL)
        
        pos['qty'] = new_qty
        
        # Update SL/TP only if provided in new order
        if order.stop_loss: pos['stop_loss'] = order.stop_loss
        if order.take_profit: pos['take_profit'] = order.take_profit
        
        # Cleanup if flat
        if abs(new_qty) < 1e-9:
            if symbol in self.positions:
                del self.positions[symbol]
        
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self.orders and self.orders[order_id].status == OrderStatus.PENDING:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        if order_id in self.orders:
            return self.orders[order_id].status
        return OrderStatus.REJECTED
    
    def get_position(self, symbol: str) -> dict:
        """Get position for a symbol."""
        return self.positions.get(symbol, {'qty': 0, 'avg_price': 0})
    
    def get_account(self) -> dict:
        """Get account summary."""
        # Calculate total position value
        position_value = sum(
            self.current_prices.get(sym, pos['avg_price']) * pos['qty']
            for sym, pos in self.positions.items()
        )
        total_equity = self.cash + position_value
        
        return {
            'cash': self.cash,
            'position_value': position_value,
            'total_equity': total_equity,
            'initial_capital': self.initial_capital,
            'total_return_pct': ((total_equity - self.initial_capital) / self.initial_capital) * 100
        }

    def get_open_positions(self) -> list:
        """Return a list of open positions."""
        positions_list = []
        for symbol, pos_data in self.positions.items():
            if pos_data and pos_data.get('qty', 0) != 0:
                positions_list.append({
                    "symbol": symbol,
                    "qty": abs(pos_data['qty']),
                    "side": "long" if pos_data['qty'] > 0 else "short",
                    "entry_price": pos_data.get('avg_price', 0)
                })
        return positions_list


class AlpacaExecutor(OrderExecutor):
    """
    Live trading executor using Alpaca API.
    Requires APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.
    """
    
    def __init__(self, paper: bool = True):
        """
        Args:
            paper: If True, use Alpaca paper trading. If False, use live trading.
        """
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            raise ImportError("Please install alpaca-trade-api: pip install alpaca-trade-api")
        
        api_key = os.environ.get('APCA_API_KEY_ID')
        api_secret = os.environ.get('APCA_API_SECRET_KEY')
        
        if not api_key or not api_secret:
            raise ValueError("Alpaca API keys not found. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
        
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.paper = paper
        logger.info(f"Connected to Alpaca ({'Paper' if paper else 'LIVE'} trading)")
        
    def submit_order(self, symbol: str, side: OrderSide, quantity: int,
                     order_type: str = "market", limit_price: Optional[float] = None,
                     stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Order:
        """Submit order to Alpaca."""
        order_data = {
            'symbol': symbol,
            'qty': quantity,
            'side': side.value,
            'type': order_type,
            'time_in_force': 'day'
        }
        if order_type == 'limit':
            order_data['limit_price'] = limit_price

        if stop_loss is not None or take_profit is not None:
            order_data['order_class'] = 'bracket'
            order_data['stop_loss'] = {'stop_price': stop_loss}
            order_data['take_profit'] = {'limit_price': take_profit}

        try:
            alpaca_order = self.api.submit_order(**order_data)
            
            order = Order(
                id=alpaca_order.id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status=OrderStatus.PENDING
            )
            
            logger.info(f"[ALPACA] Order submitted: {side.value} {quantity} {symbol}")
            return order
            
        except Exception as e:
            logger.error(f"[ALPACA] Order failed: {e}")
            return Order(
                id="ERROR",
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                status=OrderStatus.REJECTED
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Alpaca."""
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"[ALPACA] Cancel failed: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from Alpaca."""
        try:
            order = self.api.get_order(order_id)
            status_map = {
                'new': OrderStatus.PENDING,
                'filled': OrderStatus.FILLED,
                'partially_filled': OrderStatus.PARTIAL,
                'cancelled': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED
            }
            return status_map.get(order.status, OrderStatus.PENDING)
        except Exception:
            return OrderStatus.REJECTED
    
    def get_position(self, symbol: str) -> dict:
        """Get position from Alpaca."""
        try:
            pos = self.api.get_position(symbol)
            return {'qty': int(pos.qty), 'avg_price': float(pos.avg_entry_price)}
        except Exception:
            return {'qty': 0, 'avg_price': 0}
    
    def get_account(self) -> dict:
        """Get account from Alpaca."""
        try:
            account = self.api.get_account()
            return {
                'cash': float(account.cash),
                'position_value': float(account.long_market_value),
                'total_equity': float(account.equity),
                'buying_power': float(account.buying_power)
            }
        except Exception as e:
            logger.error(f"[ALPACA] Get account failed: {e}")
            return {}

    def get_open_positions(self) -> list:
        """Get all open positions from Alpaca."""
        positions_list = []
        try:
            portfolio = self.api.list_positions()
            for position in portfolio:
                positions_list.append({
                    "symbol": position.symbol,
                    "qty": float(position.qty),
                    "side": position.side,
                    "entry_price": float(position.avg_entry_price)
                })
        except Exception as e:
            logger.error(f"[ALPACA] Failed to get open positions: {e}")
        return positions_list

class OandaExecutor(OrderExecutor):

    """

    Live trading executor using OANDA API.

    Requires OANDA_ACCOUNT_ID and OANDA_ACCESS_TOKEN environment variables.

    """

    def __init__(self, account_id: str, access_token: str, practice: bool = True):

        """

        Args:

            account_id: OANDA account ID.

            access_token: OANDA API access token.

            practice: If True, use OANDA practice (demo) account.

        """

        try:

            import oandapyV20.endpoints.orders as orders

            import oandapyV20.endpoints.accounts as accounts

            import oandapyV20.endpoints.positions as positions

            from oandapyV20 import API

        except ImportError:

            raise ImportError("Please install oandapyV20: pip install oandapyV20")



        self.account_id = account_id

        self.access_token = access_token

        self.practice = practice

        

        if not self.account_id or not self.access_token:

            raise ValueError("OANDA account ID and access token are required.")



        environment = "practice" if practice else "live"

        self.api = API(access_token=self.access_token, environment=environment)

        

        self.orders_api = orders

        self.accounts_api = accounts

        self.positions_api = positions



        logger.info(f"Connected to OANDA ({'Practice' if practice else 'LIVE'} trading)")



    def submit_order(self, symbol: str, side: OrderSide, quantity: int,
                     order_type: str = "market", limit_price: Optional[float] = None,
                     stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Order:
        """Submit order to OANDA."""
        oanda_instrument = symbol.replace("/", "_") # OANDA uses "_" instead of "/"

        order_data = {
            "order": {
                "units": str(quantity) if side == OrderSide.BUY else str(-quantity),
                "instrument": oanda_instrument,
                "timeInForce": "FOK", # Fill Or Kill
                "type": order_type.upper(),
                "positionFill": "DEFAULT"
            }
        }

        if order_type == 'limit' and limit_price:
            order_data['order']['price'] = str(limit_price)

        if stop_loss:
            order_data['order']['stopLossOnFill'] = {'price': str(stop_loss)}
        if take_profit:
            order_data['order']['takeProfitOnFill'] = {'price': str(take_profit)}

        r = self.orders_api.OrderCreate(self.account_id, data=order_data)

        try:
            self.api.request(r)
            response = r.response
            
            order_id = response.get('orderCreateTransaction', {}).get('id')
            if order_id:
                return Order(
                    id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    status=OrderStatus.FILLED # OANDA FOK orders are either filled or rejected
                )
            else:
                 logger.error(f"[OANDA] Order failed: {response}")
                 return self._create_rejected_order(symbol, side, quantity, order_type)
        except Exception as e:
            logger.error(f"[OANDA] Order failed: {e}")
            return self._create_rejected_order(symbol, side, quantity, order_type)



    def _create_rejected_order(self, symbol, side, quantity, order_type):

        return Order(

            id="ERROR",

            symbol=symbol,

            side=side,

            quantity=quantity,

            order_type=order_type,

            status=OrderStatus.REJECTED

        )



    def cancel_order(self, order_id: str) -> bool:

        """Cancel order on OANDA. Not applicable for FOK orders."""

        logger.warning("[OANDA] Cancel order is not applicable for 'Fill Or Kill' orders.")

        return False



    def get_order_status(self, order_id: str) -> OrderStatus:

        """Get order status from OANDA."""

        r = self.orders_api.OrderDetails(self.account_id, orderID=order_id)

        try:

            self.api.request(r)

            state = r.response.get('order', {}).get('state')

            status_map = {

                'PENDING': OrderStatus.PENDING,

                'FILLED': OrderStatus.FILLED,

                'CANCELLED': OrderStatus.CANCELLED,

                'REJECTED': OrderStatus.REJECTED,

            }

            return status_map.get(state, OrderStatus.REJECTED)

        except Exception:

            return OrderStatus.REJECTED



    def get_position(self, symbol: str) -> dict:

        """Get position from OANDA."""

        oanda_instrument = symbol.replace("/", "_")

        r = self.positions_api.PositionDetails(self.account_id, instrument=oanda_instrument)

        try:

            self.api.request(r)

            position = r.response.get('position', {})

            long_qty = int(position.get('long', {}).get('units', 0))

            short_qty = int(position.get('short', {}).get('units', 0))

            

            if long_qty > 0:

                return {'qty': long_qty, 'avg_price': float(position.get('long',{}).get('averagePrice',0))}

            elif short_qty > 0:

                 return {'qty': -short_qty, 'avg_price': float(position.get('short',{}).get('averagePrice',0))}

            else:

                return {'qty': 0, 'avg_price': 0}

        except Exception:

            return {'qty': 0, 'avg_price': 0}



        def get_account(self) -> dict:



            """Get account from OANDA."""



            r = self.accounts_api.AccountSummary(self.account_id)



            try:



                self.api.request(r)



                account = r.response.get('account', {})



                return {



                    'cash': float(account.get('balance', 0)),



                    'position_value': float(account.get('positionValue', 0)),



                    'total_equity': float(account.get('NAV', 0)),



                    'buying_power': float(account.get('marginAvailable', 0))



                }



            except Exception as e:



                logger.error(f"[OANDA] Get account failed: {e}")



                return {}



    



        def get_open_positions(self) -> list:



            """Get all open positions from OANDA."""



            positions_list = []



            r = self.accounts_api.AccountDetails(self.account_id)



            try:



                self.api.request(r)



                account_details = r.response



                open_positions = account_details.get('account', {}).get('positions', [])



                for position in open_positions:



                    instrument = position['instrument']



                    # OANDA provides separate 'long' and 'short' objects



                    if 'long' in position and int(position['long']['units']) != 0:



                        pos_data = position['long']



                        positions_list.append({



                            "symbol": instrument.replace("_", "/"),



                            "qty": float(pos_data['units']),



                            "side": "long",



                            "entry_price": float(pos_data.get('averagePrice', 0))



                        })



                    if 'short' in position and int(position['short']['units']) != 0:



                        pos_data = position['short']



                        positions_list.append({



                            "symbol": instrument.replace("_", "/"),



                            "qty": abs(float(pos_data['units'])),



                            "side": "short",



                            "entry_price": float(pos_data.get('averagePrice', 0))



                        })



            except Exception as e:



                logger.error(f"[OANDA] Failed to get open positions: {e}")



            return positions_list



class MT5Executor(OrderExecutor):

    """

    Live trading executor using MetaTrader 5 API.

    Requires MetaTrader5 library and a running MT5 terminal.

    """

    def __init__(self, login: int, password: str, server: str):

        try:

            import MetaTrader5 as mt5

        except ImportError:

            raise ImportError("Please install MetaTrader5: pip install MetaTrader5")



        self.login = login

        self.password = password

        self.server = server
        self.paper_trading = "Demo" in server or "demo" in server



        if not mt5.initialize(login=self.login, password=self.password, server=self.server):

            logger.error(f"MT5 initialization failed, error code: {mt5.last_error()}")

            raise ConnectionError("Could not connect to MetaTrader 5 terminal.")

        logger.info(f"Connected to MetaTrader 5 account {self.login} on server {self.server}")

    def _map_symbol(self, symbol: str) -> str:
        """Map common short codes to MT5 full names."""
        mapping = {
            "R_75": "Volatility 75 Index",
            "R_100": "Volatility 100 Index",
            "R_50": "Volatility 50 Index",
            "R_25": "Volatility 25 Index",
            "R_10": "Volatility 10 Index",
            "CRASH500": "Crash 500 Index",
            "CRASH1000": "Crash 1000 Index",
            "BOOM500": "Boom 500 Index",
            "BOOM1000": "Boom 1000 Index",
            "BOOM300": "Boom 300 Index",
            "CRASH300": "Crash 300 Index"
        }
        return mapping.get(symbol, symbol)



    def _create_rejected_order(self, symbol, side, quantity, order_type):
        return Order(
            id="ERROR",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            status=OrderStatus.REJECTED
        )

    def submit_order(self, symbol: str, side: OrderSide, quantity: int,
                     order_type: str = "market", limit_price: Optional[float] = None,
                     stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Order:
        import MetaTrader5 as mt5
        
        # Map symbol name
        symbol = self._map_symbol(symbol)
        
        # Prepare the request
        if not mt5.symbol_select(symbol, True):
            logger.error(f"[MT5] Failed to select symbol {symbol}")
            return self._create_rejected_order(symbol, side, quantity, order_type)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"[MT5] Failed to get tick info for {symbol}")
            return self._create_rejected_order(symbol, side, quantity, order_type)

        # Prepare the request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(quantity),
            "type": mt5.ORDER_TYPE_BUY if side == OrderSide.BUY else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if side == OrderSide.BUY else tick.bid,
            "type_filling": mt5.ORDER_FILLING_FOK,
            "deviation": 20,
            "comment": "python script order",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        if stop_loss:
            request["sl"] = float(stop_loss)
        if take_profit:
            request["tp"] = float(take_profit)
        
        # Execute the order
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[MT5] Order placed successfully: {side.value} {quantity} {symbol}")
            return Order(
                id=str(result.order),
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status=OrderStatus.FILLED,
                filled_price=result.price
            )
        else:
            logger.error(f"[MT5] Order failed, retcode={result.retcode}, comment={result.comment}")
            return Order(
                id="ERROR",
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                status=OrderStatus.REJECTED
            )



    def cancel_order(self, order_id: str) -> bool:

        import MetaTrader5 as mt5

        request = {

            "action": mt5.TRADE_ACTION_REMOVE,

            "order": int(order_id),

            "comment": "cancel order",

        }

        result = mt5.order_send(request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:

            logger.info(f"[MT5] Order {order_id} cancelled successfully.")

            return True

        else:

            logger.error(f"[MT5] Order {order_id} cancellation failed, retcode={result.retcode}")

            return False



    def get_order_status(self, order_id: str) -> OrderStatus:

        import MetaTrader5 as mt5

        orders = mt5.orders_get(ticket=int(order_id))

        if orders is None:

            return OrderStatus.REJECTED # Order not found

        

        order = orders[0]

        status_map = {

            mt5.ORDER_STATE_STARTED: OrderStatus.PENDING,

            mt5.ORDER_STATE_PLACED: OrderStatus.PENDING,

            mt5.ORDER_STATE_PARTIAL: OrderStatus.PARTIAL,

            mt5.ORDER_STATE_FILLED: OrderStatus.FILLED,

            mt5.ORDER_STATE_CANCELED: OrderStatus.CANCELLED,

            mt5.ORDER_STATE_REJECTED: OrderStatus.REJECTED,

            mt5.ORDER_STATE_EXPIRED: OrderStatus.CANCELLED, # Treat expired as cancelled

        }

        return status_map.get(order.state, OrderStatus.PENDING)



    def get_position(self, symbol: str) -> dict:

        import MetaTrader5 as mt5

        positions = mt5.positions_get(symbol=symbol)

        if positions is None or len(positions) == 0:

            return {'qty': 0, 'avg_price': 0}

        

        # Assuming only one position per symbol for simplicity

        pos = positions[0]

        return {'qty': pos.volume_current if pos.type == mt5.POSITION_TYPE_BUY else -pos.volume_current, 'avg_price': pos.price_open}



    def get_position(self, symbol: str) -> dict:

        import MetaTrader5 as mt5

        positions = mt5.positions_get(symbol=symbol)

        if positions is None or len(positions) == 0:

            return {'qty': 0, 'avg_price': 0}

        

        # Assuming only one position per symbol for simplicity

        pos = positions[0]

        return {'qty': pos.volume_current if pos.type == mt5.POSITION_TYPE_BUY else -pos.volume_current, 'avg_price': pos.price_open}



    def get_account(self) -> dict:

        import MetaTrader5 as mt5

        account_info = mt5.account_info()

        if account_info is None:

            logger.error("Failed to get MT5 account info.")

            return {}

        

        return {

            'cash': account_info.balance,

            'position_value': account_info.margin, # Margin can be used as a proxy for position value

            'total_equity': account_info.equity,

            'buying_power': account_info.margin_free

        }



    def get_open_positions(self) -> list:

        """Get all open positions from MetaTrader 5."""

        import MetaTrader5 as mt5

        positions_list = []

        

        # Invert the mapping from _map_symbol to find the short name if possible

        reverse_mapping = {v: k for k, v in self._get_symbol_mapping().items()}



        try:

            positions = mt5.positions_get()

            if positions is None:

                return []



            for pos in positions:

                # Use the shorter, internal symbol name if available

                symbol_name = reverse_mapping.get(pos.symbol, pos.symbol)

                

                positions_list.append({

                    "symbol": symbol_name,

                    "qty": float(pos.volume),

                    "side": "long" if pos.type == mt5.POSITION_TYPE_BUY else "short",

                    "entry_price": float(pos.price_open)

                })

        except Exception as e:

            logger.error(f"[MT5] Failed to get open positions: {e}")

        return positions_list



    def _get_symbol_mapping(self) -> dict:

        """Helper to centralize the symbol map."""

        return {

            "R_75": "Volatility 75 Index",

            "R_100": "Volatility 100 Index",

            "R_50": "Volatility 50 Index",

            "R_25": "Volatility 25 Index",

            "R_10": "Volatility 10 Index",

            "CRASH500": "Crash 500 Index",

            "CRASH1000": "Crash 1000 Index",

            "BOOM500": "Boom 500 Index",

            "BOOM1000": "Boom 1000 Index",

            "BOOM300": "Boom 300 Index",

            "CRASH300": "Crash 300 Index"

        }
