"""
WebSocket Client for Real-Time Price Streaming

Provides real-time price updates from Deriv API for improved exit monitoring.
Runs in a separate thread to avoid blocking the main trading loop.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Dict, List, Callable, Optional
import websockets

logger = logging.getLogger(__name__)


class DerivWebSocketClient:
    """
    WebSocket client for Deriv API tick streaming.
    
    Features:
    - Real-time tick updates for synthetic indices
    - Thread-safe price storage
    - Auto-reconnection with exponential backoff
    - Graceful shutdown handling
    """
    
    # Deriv symbol mapping: Internal name -> Deriv API symbol
    SYMBOL_MAP = {
        'Volatility 75 Index': 'R_75',
        'Crash 500 Index': '1HZ500V',
        'Boom 1000 Index': '1HZ1000V',
        'R_75': 'R_75',
        'R_100': 'R_100',
        '1HZ500V': '1HZ500V',
        '1HZ1000V': '1HZ1000V'
    }
    
    def __init__(self, symbols: List[str], api_token: Optional[str] = None):
        """
        Initialize WebSocket client.
        
        Args:
            symbols: List of symbols to subscribe to (e.g., ['Volatility 75 Index'])
            api_token: Optional Deriv API token for authentication
        """
        self.symbols = symbols
        self.api_token = api_token
        self.ws_url = "wss://ws.derivws.com/websockets/v3?app_id=1089"
        
        # Thread-safe price storage
        self.prices: Dict[str, float] = {}
        self.price_lock = threading.Lock()
        
        # Connection state
        self.websocket = None
        self.running = False
        self.connected = False
        
        # Threading
        self.thread = None
        self.loop = None
        
        # Reconnection settings
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # seconds
        
        logger.info(f"DerivWebSocketClient initialized for {len(symbols)} symbols")
    
    def connect(self):
        """Start WebSocket connection in a separate thread."""
        if self.running:
            logger.warning("WebSocket already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        
        # Wait for connection (with timeout)
        timeout = 10
        start = time.time()
        while not self.connected and (time.time() - start) < timeout:
            time.sleep(0.1)
        
        if self.connected:
            logger.info("WebSocket connected successfully")
        else:
            logger.warning("WebSocket connection timeout")
    
    def _run_async_loop(self):
        """Run asyncio event loop in thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            logger.error(f"WebSocket loop error: {e}")
        finally:
            self.loop.close()
    
    async def _connect_and_listen(self):
        """Connect to WebSocket and listen for price updates."""
        while self.running:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20) as websocket:
                    self.websocket = websocket
                    self.connected = True
                    self.reconnect_attempts = 0
                    
                    logger.info("WebSocket connected, subscribing to symbols...")
                    
                    # Subscribe to tick streams for all symbols
                    await self._subscribe_to_symbols()
                    
                    # Listen for messages
                    async for message in websocket:
                        if not self.running:
                            break
                        await self._handle_message(message)
                        
            except websockets.exceptions.ConnectionClosed:
                self.connected = False
                logger.warning("WebSocket connection closed")
                if self.running:
                    await self._attempt_reconnect()
            except Exception as e:
                self.connected = False
                logger.error(f"WebSocket error: {e}")
                if self.running:
                    await self._attempt_reconnect()
    
    async def _subscribe_to_symbols(self):
        """Subscribe to tick streams for all configured symbols."""
        for symbol in self.symbols:
            deriv_symbol = self.SYMBOL_MAP.get(symbol, symbol)
            
            subscribe_msg = {
                "ticks": deriv_symbol,
                "subscribe": 1
            }
            
            try:
                await self.websocket.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to {symbol} ({deriv_symbol})")
            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol}: {e}")
    
    async def _handle_message(self, message: str):
        """Process incoming WebSocket messages."""
        try:
            data = json.loads(message)
            
            # Handle tick updates
            if 'tick' in data:
                tick_data = data['tick']
                symbol = tick_data.get('symbol')
                quote = tick_data.get('quote')
                
                if symbol and quote:
                    # Reverse map: Deriv symbol -> Internal name
                    internal_symbol = self._get_internal_symbol(symbol)
                    
                    # Thread-safe price update
                    with self.price_lock:
                        self.prices[internal_symbol] = float(quote)
                    
                    logger.debug(f"{internal_symbol}: {quote}")
            
            # Handle subscription confirmation
            elif 'subscription' in data:
                logger.debug(f"Subscription confirmed: {data['subscription']}")
            
            # Handle errors
            elif 'error' in data:
                logger.error(f"WebSocket error: {data['error']}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _get_internal_symbol(self, deriv_symbol: str) -> str:
        """Convert Deriv API symbol to internal symbol name."""
        # Reverse lookup
        for internal, deriv in self.SYMBOL_MAP.items():
            if deriv == deriv_symbol:
                return internal
        return deriv_symbol  # Fallback
    
    async def _attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached. Stopping WebSocket.")
            self.running = False
            return
        
        self.reconnect_attempts += 1
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 60)
        
        logger.info(f"Reconnecting in {delay}s (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})...")
        await asyncio.sleep(delay)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol (thread-safe).
        
        Args:
            symbol: Symbol name (e.g., 'Volatility 75 Index')
        
        Returns:
            Latest price or None if not available
        """
        with self.price_lock:
            return self.prices.get(symbol)
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all current prices (thread-safe)."""
        with self.price_lock:
            return self.prices.copy()
    
    def disconnect(self):
        """Gracefully disconnect WebSocket."""
        logger.info("Disconnecting WebSocket...")
        self.running = False
        self.connected = False
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        logger.info("WebSocket disconnected")
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.connected


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Deriv WebSocket Client...")
    print("Connecting to Volatility 75 Index...")
    
    client = DerivWebSocketClient(['Volatility 75 Index', 'Crash 500 Index'])
    client.connect()
    
    try:
        # Monitor prices for 30 seconds
        for i in range(30):
            time.sleep(1)
            prices = client.get_all_prices()
            if prices:
                print(f"\n[{i+1}s] Current Prices:")
                for symbol, price in prices.items():
                    print(f"  {symbol}: {price}")
            else:
                print(f"[{i+1}s] Waiting for price data...")
    
    except KeyboardInterrupt:
        print("\nStopping test...")
    finally:
        client.disconnect()
        print("Test complete!")
