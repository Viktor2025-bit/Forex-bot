
import subprocess
import logging
import threading
import sys
from collections import deque
from datetime import datetime

class RetrainingManager:
    """
    Manages the lifecycle of AI model retraining.
    Monitors performance and triggers retraining scripts when necessary.
    Also tracks per-model accuracy for weighted ensemble voting.
    """
    
    def __init__(self, accuracy_threshold=0.55, min_trades=20):
        self.logger = logging.getLogger("trading_bot.retrainer")
        self.accuracy_threshold = accuracy_threshold
        self.min_trades = min_trades
        
        # Store rolling history of wins (1) and losses (0)
        self.trade_history = {} 
        self.is_training = {} # Lock to prevent multiple trainings for same symbol
        self.reload_needed = set() # Flags for the bot to reload models
        
        # Per-model prediction tracking for weighted ensemble
        # Structure: {symbol: {model_name: deque of (prediction, was_correct)}}
        self.model_predictions = {}

    def record_prediction(self, symbol: str, model_name: str, prediction: float, actual_direction: int):
        """
        Record a model's prediction and whether it was correct.
        actual_direction: 1 for UP, -1 for DOWN
        """
        if symbol not in self.model_predictions:
            self.model_predictions[symbol] = {}
        if model_name not in self.model_predictions[symbol]:
            self.model_predictions[symbol][model_name] = deque(maxlen=30)  # Last 30 predictions
            
        # Was the prediction correct?
        predicted_up = prediction > 0.5
        actual_up = actual_direction > 0
        was_correct = (predicted_up == actual_up)
        
        self.model_predictions[symbol][model_name].append(was_correct)
        
    def get_model_weights(self, symbol: str) -> dict:
        """
        Calculate accuracy-based weights for each model.
        Returns normalized weights summing to 1.
        """
        if symbol not in self.model_predictions:
            return {'xgboost': 0.5, 'lstm': 0.5}  # Default equal weights
            
        weights = {}
        for model_name, history in self.model_predictions[symbol].items():
            if len(history) >= 5:  # Need at least 5 predictions
                accuracy = sum(history) / len(history)
                # Weight is accuracy squared (rewards high accuracy more)
                weights[model_name] = accuracy ** 2
            else:
                weights[model_name] = 0.5  # Default until enough data
                
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        else:
            weights = {'xgboost': 0.5, 'lstm': 0.5}
            
        return weights

    def record_result(self, symbol: str, pnl: float):
        """Record the outcome of a trade (Win/Loss)."""
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque(maxlen=50) # Look at last 50 trades
            self.is_training[symbol] = False
            
        is_win = 1 if pnl > 0 else 0
        self.trade_history[symbol].append(is_win)
        
        # Log current stats
        win_rate = sum(self.trade_history[symbol]) / len(self.trade_history[symbol])
        self.logger.info(f"[{symbol}] Trade recorded. Win Rate (Last {len(self.trade_history[symbol])}): {win_rate:.1%}")
        
        # Check if we need to retrain
        self._check_and_trigger(symbol, win_rate)

    def _check_and_trigger(self, symbol: str, win_rate: float):
        """Check criteria and trigger retraining if needed."""
        if len(self.trade_history[symbol]) < self.min_trades:
            return
            
        if self.is_training.get(symbol, False):
            return

        if win_rate < self.accuracy_threshold:
            self.logger.warning(f"[{symbol}] Performance Drop Detected! Win Rate {win_rate:.1%} < {self.accuracy_threshold:.1%}. Initiating Retraining...")
            
            # Run in separate thread to not block trading
            t = threading.Thread(target=self._run_retraining_script, args=(symbol,))
            t.daemon = True
            t.start()
            
    def _run_retraining_script(self, symbol: str):
        """Execute the training script in a subprocess."""
        self.is_training[symbol] = True
        try:
            # Using python from current environment
            python_exe = sys.executable
            
            # 1. Retrain XGBoost
            self.logger.info(f"[{symbol}] Starting XGBoost Retraining...")
            cmd_xgb = [python_exe, "train_model.py", "--symbol", symbol]
            res_xgb = subprocess.run(cmd_xgb, capture_output=True, text=True, check=False)
            
            xgb_success = (res_xgb.returncode == 0)
            if xgb_success:
                self.logger.info(f"[{symbol}] XGBoost Retraining Output:\n{res_xgb.stdout}")
            else:
                self.logger.error(f"[{symbol}] XGBoost Retraining Failed:\n{res_xgb.stderr}")

            # 2. Retrain LSTM
            self.logger.info(f"[{symbol}] Starting LSTM Retraining (Fine-tuning)...")
            cmd_lstm = [python_exe, "train.py", "--ticker", symbol, "--epochs", "10"] # 10 epochs for quick fine-tuning
            res_lstm = subprocess.run(cmd_lstm, capture_output=True, text=True, check=False)
            
            lstm_success = (res_lstm.returncode == 0)
            if lstm_success:
                self.logger.info(f"[{symbol}] LSTM Retraining Output:\n{res_lstm.stdout}")
            else:
                self.logger.error(f"[{symbol}] LSTM Retraining Failed:\n{res_lstm.stderr}")

            if xgb_success or lstm_success:
                self.logger.info(f"[{symbol}] ✅ Retraining Cycle Complete.")
                self.reload_needed.add(symbol) # Signal bot to reload
                self.trade_history[symbol].clear()
                self.logger.info(f"[{symbol}] Trade history reset. Monitoring new model performance.")
            else:
                self.logger.error(f"[{symbol}] ❌ Full Retraining Cycle Failed.")
                
        except Exception as e:
            self.logger.error(f"[{symbol}] Error during retraining execution: {e}")
        finally:
            self.is_training[symbol] = False
