from strategies.order_executor import MT5Executor, OrderSide
from utils.config_loader import load_config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_exec")

def test_order():
    logger.info("Loading config...")
    config = load_config()
    mt5_cfg = config['brokers']['mt5']
    
    logger.info("Initializing Executor...")
    executor = MT5Executor(
        login=int(mt5_cfg['login']), 
        password=mt5_cfg['password'], 
        server=mt5_cfg['server']
    )
    
    symbol = "Volatility 75 Index" # Test symbol
    qty = 0.01
    
    logger.info(f"Attempting BUY {qty} {symbol}...")
    order = executor.submit_order(symbol, OrderSide.BUY, qty)
    
    logger.info(f"Order Status: {order.status}")
    logger.info(f"Order ID: {order.id}")
    
    if order.status.value == "filled":
        print("SUCCESS: Order Filled")
        # Attempt close immediately
        print("Closing position...")
        executor.submit_order(symbol, OrderSide.SELL, qty)
    else:
        print("FAILURE: Order Rejected")

if __name__ == "__main__":
    test_order()
