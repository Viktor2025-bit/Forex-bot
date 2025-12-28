import json
import MetaTrader5 as mt5
import os

def check_mt5_connection():
    print("--- MT5 Connection Verification ---")
    
    # Load config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            mt5_config = config['brokers']['mt5']
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    login = int(mt5_config['login'])
    password = mt5_config['password']
    server = mt5_config['server']

    print(f"Attempting connection to {server} for account {login}...")

    # Initialize
    if not mt5.initialize(login=login, password=password, server=server):
        print("initialize() failed, error code =", mt5.last_error())
        return False
    
    print("MetaTrader5 package version:", mt5.__version__)

    # Login check
    authorized = mt5.login(login, password=password, server=server)
    if authorized:
        print("Connected to MT5 account #{}".format(login))
        account_info = mt5.account_info()
        if account_info!=None:
            print(f"Balance: {account_info.balance} {account_info.currency}")
            print(f"Equity: {account_info.equity}")
            print(f"Leverage: 1:{account_info.leverage}")
            print("Connection Verified Successfully!")
        else:
            print("Failed to retrieve account info")
    else:
        print("failed to connect at account #{}, error code: {}".format(login, mt5.last_error()))

    # Shutdown
    mt5.shutdown()

if __name__ == "__main__":
    check_mt5_connection()
