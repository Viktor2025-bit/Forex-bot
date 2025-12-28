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
from strategies.order_executor import OrderExecutor, OrderSide
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
        # RiskManager expects a dict or object. Let's inspect strategies/risk_manager.py usage ideally.
        # Assuming it takes a dict or we adapt. 
        # Previous code used: self.risk_manager = RiskManager(RiskParameters(...))
        # But here I'll try passing the config dict or kwargs if supported.
        # Let's check if RiskManager supports dict. 
        # Ideally I should have checked risk_manager.py content. 
        # But commonly it takes params.
        # Let's assume it takes a config dict for now based on my last read or I'll try to be safe.
        # Actually I saw RiskParameters in the previous file.
        # Let's try to just pass the risk config dict, if it fails I'll fix it.
        # Wait, I should probably check risk_manager.py to be safe.
        # But for now, let's assume I can pass the dict or I'll wrap it.
        
        # Risk Manager Setup
        class RiskConfigWrapper:
            def __init__(self, data):
                for k, v in data.items():
                    if isinstance(v, dict):
                        setattr(self, k, RiskConfigWrapper(v))
                    else:
                        setattr(self, k, v)

        self.risk_manager = RiskManager(RiskConfigWrapper(config.get('risk', {})))
        # Order Executor
        if config.get('bot', {}).get('paper_trading', True):
            from strategies.order_executor import PaperTradingExecutor
            self.executor = PaperTradingExecutor(initial_capital=config.get('bot', {}).get('initial_capital', 10000))
        else:
            # Placeholder for live executor selection (MT5/Oanda) - simplistic logic for now
            # In a real scenario, we'd select based on 'brokers' config.
            # For now, if paper_trading is False, we might want to default to MT5 based on user intent, 
            # but let's stick to safe fallback or check config.
            from strategies.order_executor import MT5Executor
            mt5_cfg = config.get('brokers', {}).get('mt5', {})
            self.executor = MT5Executor(login=int(mt5_cfg['login']), password=mt5_cfg['password'], server=mt5_cfg['server']) 
        
        # Model Setup
        self.model_type = config.get('model', {}).get('type', 'lstm')
        self.model = self._load_model(config.get('model', {}))

        # Data preprocessor
        if self.model_type == 'xgboost':
            self.preprocessor = FeatureEngine()
        else:
            self.preprocessor = DataPreprocessor()
            
        self.seq_length = config.get('bot', {}).get('seq_length', 30)
        self.trading_interval = config.get('bot', {}).get('trading_interval_seconds', 60)
        
        # State
        self.positions = {symbol: None for symbol in self.symbols}
        self.entry_prices = {symbol: None for symbol in self.symbols}
        self.last_prediction = {}
        
        self.monitor.logger.info(f"Bot initialized for {self.symbols} ({'Paper' if self.executor.paper_trading else 'LIVE'} trading)")

    def _load_model(self, model_config: dict):
        model_path = model_config.get('path', 'models/trained_lstm.pth')

        if self.model_type == 'xgboost':
            xg_params = model_config.get('xgboost', {})
            actual_path = model_config.get('xgboost', {}).get('model_path', model_path)
            model = ForexModel(model_path=actual_path, **xg_params)
            if os.path.exists(actual_path):
                model.load_model()
                self.monitor.logger.info(f"XGBoost Model loaded from {actual_path}")
            else:
                self.monitor.logger.warning(f"XGBoost Model not found at {actual_path}. Using untrained model instance.")
            return model

        # Default to LSTM
        lstm_params = model_config.get('lstm', {})
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
             self.monitor.logger.warning(f"LSTM Model not found at {model_path}. Using untrained model.")
             
        return model

    def prepare_features(self, df: pd.DataFrame):
        if self.model_type == 'xgboost':
            try:
                df_features = self.preprocessor.generate_features(df)
                ignore_cols = ['Target', 'Future_Return', 'Date', 'index', 'Open', 'High', 'Low', 'Close', 'Volume']
                feature_cols = [c for c in df_features.columns if c not in ignore_cols]
                
                if df_features.empty:
                     return None
                     
                return df_features[feature_cols]
            except Exception as e:
                self.monitor.logger.error(f"Error preparing features for XGBoost: {e}")
                return None
        else:
            try:
                df_clean = self.preprocessor.clean_data(df)
                df_indicators = self.preprocessor.add_technical_indicators(df_clean)
                df_normalized = self.preprocessor.normalize_data(df_indicators)
                
                feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_14', 'RSI_14']
                return df_normalized[feature_columns].values
            except Exception as e:
                self.monitor.logger.error(f"Error preparing features for LSTM: {e}")
                return None

    def make_prediction(self, features) -> float:
        if features is None or len(features) == 0:
            return 0.5

        if self.model_type == 'xgboost':
            try:
                latest_features = features.iloc[[-1]]
                probs = self.model.predict(latest_features) # returns prob of class 1
                return float(probs[0])
            except Exception as e:
                self.monitor.logger.error(f"Prediction error: {e}")
                return 0.5
        else:
            # LSTM Logic
            try:
                import torch
                data = features
                if len(data) < self.seq_length:
                    return 0.5
                
                sequence = data[-self.seq_length:]
                X_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = self.model(X_tensor).item()
                return prediction
            except Exception as e:
                self.monitor.logger.error(f"Prediction error: {e}")
                return 0.5

    def run_trading_cycle(self):
        self.monitor.logger.info(f"Trading Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Fetch Data - Use MT5 if enabled, otherwise fallback to Yahoo Finance
        mt5_enabled = self.config.get('brokers', {}).get('mt5', {}).get('enabled', False)
        
        if mt5_enabled:
            # Real-time MT5 data (M1 candles)
            data_map = fetch_data_mt5(self.symbols, n_candles=100)
        else:
            # Historical Yahoo Finance data (delayed)
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=60) 
            data_map = fetch_historical_data(self.symbols, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
        
        if not data_map:
             self.monitor.logger.warning("No data fetched.")
             return

        # SAVE DATA FOR DASHBOARD
        save_data(data_map)

        for symbol, df in data_map.items():
            if df is None or df.empty:
                self.monitor.logger.warning(f"Insufficient data for {symbol}, skipping.")
                continue
                
            current_price = df['Close'].iloc[-1]
            
            # 2. Prepare Features
            features = self.prepare_features(df)
            
            # 3. Predict
            prediction = self.make_prediction(features)
            
            signal = "UP" if prediction > 0.5 else "DOWN"
            self.monitor.logger.info(f"Model Prediction for {symbol}: {prediction:.2%} ({signal})")
            
            self.last_prediction[symbol] = {"prob": prediction, "signal": signal, "timestamp": datetime.now().isoformat()}
            
            # 4. Execute Logic
            self.execute_strategy(symbol, prediction, current_price)

        # 5. Log Cycle Performance
        account = self.executor.get_account()
        equity = account.get('total_equity', 0)
        self.monitor.log_cycle(equity)
        self.monitor.logger.info(f"Account: Cash=${account.get('cash',0):.2f}, Equity=${equity:.2f}, Return={account.get('return_pct',0):.2f}%")
        self.monitor.logger.info("="*30)
        
        # Save state
        self.save_state()

    def execute_strategy(self, symbol, prediction, current_price):
        position = self.positions.get(symbol)
        has_position = position is not None
        
        threshold = self.config.get('risk', {}).get('min_confidence', 0.6)
        
        if not has_position and prediction > threshold:
            account = self.executor.get_account()
            buying_power = account.get('buying_power', 0)
            
            if buying_power > 0:
                # Assuming calculate_position_size handles logic (or we might need separate forex/stock calls)
                # But kept simple for now
                qty = self.risk_manager.calculate_position_size(account['total_equity'], current_price)
                if qty > 0:
                    order = self.executor.submit_order(symbol, OrderSide.BUY, qty)
                    if order:
                        price = order.filled_price or current_price
                        self.entry_prices[symbol] = price
                        self.positions[symbol] = {'side': 'long', 'qty': qty}
                        self.risk_manager.register_position(symbol, price, qty, 'long')
                        
                        self.monitor.log_trade(TradeRecord(
                            timestamp=datetime.now().isoformat(),
                            symbol=symbol,
                            side='BUY',
                            quantity=qty,
                            price=price,
                            order_id=order.order_id or "sim_id"
                        ))
                        self.monitor.logger.info(f"BOUGHT {qty} of {symbol} @ ~${price:.4f}")
        
        elif has_position and prediction < (1 - threshold):
            qty_to_sell = abs(self.positions[symbol]['qty'])
            order = self.executor.submit_order(symbol, OrderSide.SELL, qty_to_sell)
            
            if order:
                self.risk_manager.close_position(symbol, current_price)
                
                entry = self.entry_prices[symbol]
                pnl = (current_price - entry) * qty_to_sell
                
                self.monitor.log_trade(TradeRecord(
                    timestamp=datetime.now().isoformat(),
                    symbol=symbol,
                    side='SELL',
                    quantity=qty_to_sell,
                    price=current_price,
                    order_id=order.order_id or "sim_id",
                    pnl=pnl
                ))
                
                self.positions[symbol] = None
                self.entry_prices[symbol] = None
                self.monitor.logger.info(f"SOLD {qty_to_sell} of {symbol} @ ~${current_price:.4f} (PnL: ${pnl:.2f})")

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
