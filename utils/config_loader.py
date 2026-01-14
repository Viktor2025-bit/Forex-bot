import json
import os
from dotenv import load_dotenv

def load_config():
    """
    Loads configuration from config.json and overrides secrets 
    with environment variables.
    """
    # Load .env file
    load_dotenv()

    # Load base config from json
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Override with environment variables
    # Alerts
    config['alerts']['smtp_password'] = os.getenv('SMTP_PASSWORD', config['alerts']['smtp_password'])

    # Brokers
    config['brokers']['alpaca']['api_key'] = os.getenv('ALPACA_API_KEY', config['brokers']['alpaca']['api_key'])
    config['brokers']['alpaca']['api_secret'] = os.getenv('ALPACA_API_SECRET', config['brokers']['alpaca']['api_secret'])
    config['brokers']['oanda']['account_id'] = os.getenv('OANDA_ACCOUNT_ID', config['brokers']['oanda']['account_id'])
    config['brokers']['oanda']['access_token'] = os.getenv('OANDA_ACCESS_TOKEN', config['brokers']['oanda']['access_token'])
    config['brokers']['mt5']['login'] = os.getenv('MT5_LOGIN', config['brokers']['mt5']['login'])
    config['brokers']['mt5']['password'] = os.getenv('MT5_PASSWORD', config['brokers']['mt5']['password'])
    config['brokers']['deriv']['api_token'] = os.getenv('DERIV_API_TOKEN', config['brokers']['deriv']['api_token'])

    return config
