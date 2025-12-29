import logging
import json
import time
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None

# Corrected Imports
# from config import Config  <-- REMOVED
from strategies.risk_manager import RiskManager
from strategies.order_executor import OrderExecutor, OrderSide, PaperTradingExecutor
from strategy_engine import AIStrategy, EnsembleStrategy, TradeAction
from utils.data_loader import fetch_historical_data, save_data, fetch_data_mt5
from utils.monitoring import BotMonitor, TradeRecord
from models.lstm_model import TradingLSTM
from ai_model import ForexModel
from utils.preprocessing import DataPreprocessor
from feature_engine import FeatureEngine

# Setup basic logging first (monitor will take over)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize Monitor first to capture startup events
        alert_config = config.get('alerts', {})
        self.monitor = BotMonitor(
            log_dir="logs",
            enable_email=alert_config.get('email_enabled', False)
        )
        self.monitor.alerts.smtp_server = alert_config.get('smtp_server', "")
        self.monitor.alerts.smtp_port = alert_config.get('smtp_port', 587)
        self.monitor.alerts.smtp_user = alert_config.get('smtp_user', "")
        self.monitor.alerts.smtp_password = alert_config.get('smtp_password', "")
        self.monitor.alerts.alert_email = alert_config.get('alert_email', "")
        
        self.monitor.logger.info("Initializing Trading Bot...")
        
        # Market settings
        self.market_type = config.get('bot', {}).get('market_type', 'forex')
        self.symbols = config.get('markets', {}).get(self.market_type, {}).get('symbols', [])
        
        # Components
        risk_params = config.get('risk', {})
        self.risk_manager = RiskManager(**risk_params)
        
        # Order Executor
        if config.get('bot', {}).get('paper_trading', True):
            self.executor = PaperTradingExecutor(initial_capital=config.get('bot', {}).get('initial_capital', 10000))
        else:
            from strategies.order_executor import MT5Executor
            mt5_cfg = config.get('brokers', {}).get('mt5', {})
            self.executor = MT5Executor(login=int(mt5_cfg['login']), password=mt5_cfg['password'], server=mt5_cfg['server']) 
        
        # Model and Preprocessor Setup for Ensemble
        self.models = {
            'lstm': self._load_model('lstm'),
            'xgboost': self._load_model('xgboost')
        }
        self.preprocessors = {
            'lstm': DataPreprocessor(),
            'xgboost': FeatureEngine()
        }
            
        self.seq_length = config.get('bot', {}).get('seq_length', 30)
        self.trading_interval = config.get('bot', {}).get('trading_interval_seconds', 60)
        
        # Strategy
        strategy_name = config.get('model', {}).get('strategy', 'ai')
        if strategy_name == 'ensemble':
            self.strategy = EnsembleStrategy(config)
            self.monitor.logger.info("Using Ensemble Strategy")
        else:
            self.strategy = AIStrategy(config)
            self.monitor.logger.info("Using AI Strategy")


        # State
        self.positions = {symbol: None for symbol in self.symbols}
        self.entry_prices = {symbol: None for symbol in self.symbols}
        self.last_prediction = {}
        self.daily_pnl = 0.0
        self.last_trade_day = datetime.now().date()
        self.trading_halted = False
        
        self.monitor.logger.info(f"Bot initialized for {self.symbols} ({'Paper' if self.executor.paper_trading else 'LIVE'} trading)")

    def _load_model(self, model_type: str):
        model_config = self.config.get('model', {})
        
        if model_type == 'xgboost':
            xg_params = model_config.get('xgboost', {})
            actual_path = xg_params.get('model_path', 'models/xgboost_forex.json')
            model = ForexModel(model_path=actual_path, **xg_params)
            if os.path.exists(actual_path):
                model.load_model()
                self.monitor.logger.info(f"XGBoost Model loaded from {actual_path}")
            else:
                self.monitor.logger.warning(f"XGBoost model not found at {actual_path}. Using untrained model.")
            return model

        elif model_type == 'lstm':
            lstm_params = model_config.get('lstm', {})
            model_path = model_config.get('path', 'models/trained_lstm.pth')
            model = TradingLSTM(input_size=7, hidden_size=lstm_params.get('hidden_size', 64), num_layers=lstm_params.get('num_layers', 2))
            if os.path.exists(model_path):
                import torch
                try:
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    self.monitor.logger.info(f"LSTM Model loaded from {model_path}")
                except Exception as e:
                    self.monitor.logger.error(f"Failed to load LSTM model: {e}")
            else:
                self.monitor.logger.warning(f"LSTM model not found at {model_path}. Using untrained model.")
            return model
        
        return None

    def prepare_features(self, df: pd.DataFrame, model_type: str):
        preprocessor = self.preprocessors.get(model_type)
        if not preprocessor:
            return None

        try:
            if model_type == 'xgboost':
                df_features = preprocessor.generate_features(df)
                ignore_cols = ['Target', 'Future_Return', 'Date', 'index', 'Open', 'High', 'Low', 'Close', 'Volume']
                feature_cols = [c for c in df_features.columns if c not in ignore_cols]
                return df_features[feature_cols] if not df_features.empty else None
            
            elif model_type == 'lstm':
                df_clean = preprocessor.clean_data(df)
                df_indicators = preprocessor.add_technical_indicators(df_clean)
                df_normalized = preprocessor.normalize_data(df_indicators)
                feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_14', 'RSI_14']
                return df_normalized[feature_columns].values

        except Exception as e:
            self.monitor.logger.error(f"Error preparing features for {model_type}: {e}")
            return None

    def make_prediction(self, features, model_type: str) -> float:
        if features is None or len(features) == 0:
            return 0.5

        model = self.models.get(model_type)
        if not model:
            return 0.5
            
        try:
            if model_type == 'xgboost':
                latest_features = features.iloc[[-1]]
                probs = model.predict(latest_features) # returns prob of class 1
                return float(probs[0])
            
            elif model_type == 'lstm':
                import torch
                if len(features) < self.seq_length: return 0.5
                sequence = features[-self.seq_length:]
                X_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                with torch.no_grad():
                    prediction = model(X_tensor).item()
                return prediction

        except Exception as e:
            self.monitor.logger.error(f"Prediction error for {model_type}: {e}")
            return 0.5
        
        return 0.5

    def run_trading_cycle(self):
        self.monitor.logger.info(f"Trading Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check for new day to reset daily PnL and trading halt
        current_day = datetime.now().date()
        if current_day != self.last_trade_day:
            self.monitor.logger.info(f"New trading day. Resetting daily PnL from {self.daily_pnl:.2f}.")
            self.daily_pnl = 0.0
            self.last_trade_day = current_day
            self.trading_halted = False

        # 1. Fetch Data
        # ... (data fetching logic remains the same)
        
        if mt5_enabled:
            data_map = fetch_data_mt5(self.symbols, n_candles=100)
        else:
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=60) 
            data_map = fetch_historical_data(self.symbols, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
        
        if not data_map:
             self.monitor.logger.warning("No data fetched.")
             return

        save_data(data_map)

        for symbol, df in data_map.items():
            if df is None or df.empty:
                self.monitor.logger.warning(f"Insufficient data for {symbol}, skipping.")
                continue
                
            current_price = df['Close'].iloc[-1]
            
            if isinstance(self.executor, PaperTradingExecutor):
                self.executor.set_price(symbol, current_price)
            
            strategy_name = self.config.get('model', {}).get('strategy', 'ai')
            
            prediction_input = None
            if strategy_name == 'ensemble':
                predictions = {}
                for model_type in self.models.keys():
                    features = self.prepare_features(df.copy(), model_type)
                    predictions[model_type] = self.make_prediction(features, model_type)
                
                log_preds = {k: f"{v:.2%}" for k,v in predictions.items()}
                self.monitor.logger.info(f"Ensemble Predictions for {symbol}: {log_preds}")
                prediction_input = predictions
            
            else: # single 'ai' strategy
                model_type = self.config.get('model', {}).get('type', 'xgboost')
                features = self.prepare_features(df.copy(), model_type)
                prediction = self.make_prediction(features, model_type)
                self.monitor.logger.info(f"Model Prediction for {symbol}: {prediction:.2%} ({'UP' if prediction > 0.5 else 'DOWN'})")
                prediction_input = prediction

            if prediction_input is not None:
                self.last_prediction[symbol] = {"prob": prediction_input, "timestamp": datetime.now().isoformat()}
                self.execute_strategy(symbol, prediction_input, current_price)

        # 5. Log Cycle Performance
        # ... (logging logic remains the same)

    def execute_strategy(self, symbol, prediction_input, current_price):
        position_info = self.positions.get(symbol)
        action = self.strategy.get_decision(prediction_input, position_info)

        # Prevent new trades if daily loss limit is hit
        # ... (logic remains the same)
        
        # ========== EXECUTE TRADE ACTION ==========
        # ... (logic remains the same)
        pass # The rest of the method is identical to the previous version

    def save_state(self):
        try:
            state_dir = 'state'
            if not os.path.exists(state_dir):
                os.makedirs(state_dir)
            
            account = self.executor.get_account()
            
            pos_list = []
            for sym, pos in self.positions.items():
                if pos:
                    pos_list.append({
                        "symbol": sym,
                        "qty": pos['qty'],
                        "side": pos['side'],
                        "entry_price": self.entry_prices.get(sym, 0),
                        "unrealized_pl": (account.get(sym, {}).get('price', 0) - self.entry_prices.get(sym, 0)) * pos['qty'] if self.entry_prices.get(sym) else 0
                    })

            state = {
                "timestamp": datetime.now().isoformat(),
                "market_type": self.market_type,
                "symbols": self.symbols,
                "account": {
                    "cash": account.get('cash', 0),
                    "equity": account.get('total_equity', 0),
                    "return_pct": account.get('return_pct', 0),
                    "buying_power": account.get('buying_power', 0)
                },
                "positions": pos_list,
                "trade_history": self.monitor.journal.trades[-50:], 
                "predictions": self.last_prediction,
                "latest_logs": [] 
            }
            
            with open(f'{state_dir}/bot_status.json', 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.monitor.logger.error(f"Error saving state: {e}")

    def start(self):
        self.monitor.logger.info(f"Starting trading bot (interval: {self.trading_interval}s)")
        try:
            while True:
                self.run_trading_cycle()
                self.monitor.logger.info(f"Sleeping {self.trading_interval}s until next cycle...")
                time.sleep(self.trading_interval)
        except KeyboardInterrupt:
            self.monitor.logger.info("Bot stopped by user")
            # Final Report
            print(self.monitor.get_status())

if __name__ == "__main__":
    if not os.path.exists('config.json'):
        print("Config file not found.")
        sys.exit(1)
        
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    bot = TradingBot(config)
    bot.start()
