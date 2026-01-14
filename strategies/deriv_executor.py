"""
Deriv Order Executor

Handles order execution and account management for Deriv synthetic indices.
Uses the Deriv API WebSocket for real-time trading.
"""
import asyncio
import json
from datetime import datetime
from deriv_api import DerivAPI


class DerivExecutor:
    def __init__(self, api_token: str, server: str = "demo"):
        """
        Initialize Deriv executor.
        
        Args:
            api_token: Deriv API token (from account settings)
            server: 'demo' or 'real'
        """
        self.api_token = api_token
        self.server = server
        self.api = None
        self.paper_trading = (server == "demo")
        self.account_info = {}
        
    async def connect(self):
        """Establish connection to Deriv API."""
        try:
            self.api = DerivAPI(app_id=1089)
            await self.api.authorize(self.api_token)
            print(f"✅ Connected to Deriv ({self.server} mode)")
            await self.update_account_info()
            return True
        except Exception as e:
            print(f"❌ Deriv connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Deriv API."""
        if self.api:
            await self.api.disconnect()
            print("Disconnected from Deriv API")
    
    async def update_account_info(self):
        """Fetch current account balance and open positions."""
        try:
            balance_response = await self.api.balance()
            self.account_info = {
                'cash': float(balance_response.get('balance', {}).get('balance', 0)),
                'currency': balance_response.get('balance', {}).get('currency', 'USD'),
                'total_equity': float(balance_response.get('balance', {}).get('balance', 0))
            }
        except Exception as e:
            print(f"Error fetching account info: {e}")
    
    def get_account(self):
        """Return current account status (sync wrapper)."""
        return {
            'cash': self.account_info.get('cash', 0),
            'total_equity': self.account_info.get('total_equity', 0),
            'buying_power': self.account_info.get('cash', 0),
            'return_pct': 0.0  # Calculate based on initial capital if needed
        }
    
    async def execute_buy(self, symbol: str, amount: float, stop_loss: float = None, take_profit: float = None):
        """
        Execute a BUY (CALL) contract on Deriv.
        
        Args:
            symbol: Symbol to trade (e.g., 'R_75')
            amount: Stake amount in account currency
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
        """
        try:
            # For synthetics, we use 'MULTUP' or simple 'CALL' contracts
            proposal = await self.api.proposal({
                'proposal': 1,
                'amount': amount,
                'basis': 'stake',
                'contract_type': 'CALL',
                'currency': self.account_info.get('currency', 'USD'),
                'duration': 5,  # 5 ticks duration (adjust based on strategy)
                'duration_unit': 't',
                'symbol': symbol
            })
            
            if proposal.get('error'):
                print(f"Proposal error: {proposal['error'].get('message')}")
                return None
            
            # Buy the contract
            buy_response = await self.api.buy({
                'buy': proposal['proposal']['id'],
                'price': amount
            })
            
            if buy_response.get('error'):
                print(f"Buy error: {buy_response['error'].get('message')}")
                return None
            
            contract_id = buy_response['buy']['contract_id']
            print(f"✅ BUY executed: {symbol} | Stake: {amount} | Contract ID: {contract_id}")
            
            return {
                'contract_id': contract_id,
                'symbol': symbol,
                'type': 'CALL',
                'amount': amount,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Execute buy failed: {e}")
            return None
    
    async def execute_sell(self, symbol: str, amount: float, stop_loss: float = None, take_profit: float = None):
        """
        Execute a SELL (PUT) contract on Deriv.
        """
        try:
            proposal = await self.api.proposal({
                'proposal': 1,
                'amount': amount,
                'basis': 'stake',
                'contract_type': 'PUT',
                'currency': self.account_info.get('currency', 'USD'),
                'duration': 5,
                'duration_unit': 't',
                'symbol': symbol
            })
            
            if proposal.get('error'):
                print(f"Proposal error: {proposal['error'].get('message')}")
                return None
            
            buy_response = await self.api.buy({
                'buy': proposal['proposal']['id'],
                'price': amount
            })
            
            if buy_response.get('error'):
                print(f"Buy error: {buy_response['error'].get('message')}")
                return None
            
            contract_id = buy_response['buy']['contract_id']
            print(f"✅ SELL executed: {symbol} | Stake: {amount} | Contract ID: {contract_id}")
            
            return {
                'contract_id': contract_id,
                'symbol': symbol,
                'type': 'PUT',
                'amount': amount,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Execute sell failed: {e}")
            return None
    
    async def close_position(self, contract_id: str):
        """
        Sell/close an open contract.
        
        Note: Deriv contracts auto-close after duration. Manual selling only works
        for certain contract types (e.g., MULTUP/MULTDOWN).
        """
        try:
            sell_response = await self.api.sell({
                'sell': contract_id,
                'price': 0  # Market price
            })
            
            if sell_response.get('error'):
                print(f"Close position error: {sell_response['error'].get('message')}")
                return False
            
            print(f"✅ Position closed: Contract {contract_id}")
            return True
            
        except Exception as e:
            print(f"Close position failed: {e}")
            return False


# Synchronous wrapper for compatibility with trading_bot.py
class DerivExecutorSync:
    """Synchronous wrapper for DerivExecutor to match OrderExecutor interface."""
    
    def __init__(self, api_token: str, server: str = "demo"):
        self.executor = DerivExecutor(api_token, server)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.paper_trading = (server == "demo")
        
        # Connect on initialization
        self.loop.run_until_complete(self.executor.connect())
    
    def get_account(self):
        # Refresh account info from Deriv before returning
        self.loop.run_until_complete(self.executor.update_account_info())
        return self.executor.get_account()
    
    def execute_buy(self, symbol: str, qty: float, stop_loss: float = None, take_profit: float = None):
        return self.loop.run_until_complete(
            self.executor.execute_buy(symbol, qty, stop_loss, take_profit)
        )
    
    def execute_sell(self, symbol: str, qty: float, stop_loss: float = None, take_profit: float = None):
        return self.loop.run_until_complete(
            self.executor.execute_sell(symbol, qty, stop_loss, take_profit)
        )
    
    def close_position(self, contract_id: str):
        return self.loop.run_until_complete(
            self.executor.close_position(contract_id)
        )
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.loop.run_until_complete(self.executor.disconnect())
            self.loop.close()
        except:
            pass


if __name__ == "__main__":
    # Test the executor
    from utils.config_loader import load_config
    
    config = load_config()
    
    api_token = config['brokers']['deriv']['api_token']
    server = config['brokers']['deriv']['server']
    
    executor = DerivExecutorSync(api_token, server)
    
    print("\n📊 Account Info:")
    print(executor.get_account())
    
    print("\n✅ Deriv Executor ready for trading!")
