from strategies.order_executor import MT5Executor
from utils.config_loader import load_config
import MetaTrader5 as mt5

config = load_config()
mt5_cfg = config['brokers']['mt5']

if not mt5.initialize(login=int(mt5_cfg['login']), password=mt5_cfg['password'], server=mt5_cfg['server']):
    print("Failed to init MT5")
else:
    print("Connected.")
    symbols = mt5.symbols_get()
    print(f"Total Symbols: {len(symbols)}")
    print("Scanning for '75', 'Bo', 'Cr'...")
    
    matches = []
    for s in symbols:
        if "75" in s.name or "Boom" in s.name or "Crash" in s.name or "R_" in s.name:
            matches.append(s.name)
            
    print("Found Matches:")
    for m in matches:
        print(m)
        
    mt5.shutdown()
