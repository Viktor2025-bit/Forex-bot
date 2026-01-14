"""
Logging and Monitoring System for the AI Trading Bot.

Provides:
1. Structured logging to files and console
2. Trade journal tracking
3. Performance metrics dashboard
4. Email/Telegram alerts for important events
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import smtplib
from email.mime.text import MIMEText


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Set up comprehensive logging with file and console handlers.
    
    Creates separate log files for:
    - trading.log: All trading activity
    - errors.log: Errors only
    - trades.log: Trade executions only
    """
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create main logger
    logger = logging.getLogger("trading_bot")
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Main log file (all messages)
    file_handler = logging.FileHandler(f"{log_dir}/trading.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Error log file (errors only)
    error_handler = logging.FileHandler(f"{log_dir}/errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    return logger


# ============================================================
# TRADE JOURNAL
# ============================================================

@dataclass
class TradeRecord:
    """Record of a single trade for the journal."""
    timestamp: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_id: str
    pnl: Optional[float] = None
    notes: str = ""


class TradeJournal:
    """
    Maintains a persistent journal of all trades.
    Saves to JSON for easy analysis.
    """
    
    def __init__(self, journal_path: str = "logs/trade_journal.json"):
        self.journal_path = journal_path
        self.trades: list[Dict[str, Any]] = []
        self._load()
        
    def _load(self):
        """Load existing journal from disk."""
        if os.path.exists(self.journal_path):
            with open(self.journal_path, 'r') as f:
                self.trades = json.load(f)
                
    def _save(self):
        """Persist journal to disk."""
        Path(os.path.dirname(self.journal_path)).mkdir(exist_ok=True)
        with open(self.journal_path, 'w') as f:
            json.dump(self.trades, f, indent=2)
            
    def record_trade(self, trade: TradeRecord):
        """Add a trade to the journal."""
        self.trades.append(asdict(trade))
        self._save()
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all trades."""
        if not self.trades:
            return {"total_trades": 0}
        
        total_pnl = sum(t.get('pnl', 0) or 0 for t in self.trades)
        winning = sum(1 for t in self.trades if (t.get('pnl') or 0) > 0)
        
        return {
            "total_trades": len(self.trades),
            "total_pnl": total_pnl,
            "winning_trades": winning,
            "losing_trades": len(self.trades) - winning,
            "win_rate": winning / len(self.trades) if self.trades else 0
        }


# ============================================================
# ALERTING SYSTEM
# ============================================================

class AlertManager:
    """
    Sends alerts for important trading events.
    Supports email and Telegram.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.email_enabled = config.get('email_enabled', False)
        self.smtp_server = config.get('smtp_server', "")
        self.smtp_port = config.get('smtp_port', 587)
        self.smtp_user = config.get('smtp_user', "")
        self.smtp_password = config.get('smtp_password', "")
        self.alert_email = config.get('alert_email', "")
        
        self.logger = logging.getLogger("trading_bot.alerts")
        
        # Telegram Setup
        tg_config = config.get('telegram', {})
        self.tg_enabled = tg_config.get('enabled', False)
        if self.tg_enabled:
            from utils.telegram_service import TelegramService
            self.telegram = TelegramService(
                bot_token=tg_config.get('bot_token', ""),
                chat_id=tg_config.get('chat_id', ""),
                enabled=True
            )
        else:
            self.telegram = None
        
    def send_alert(self, subject: str, message: str, level: str = "INFO"):
        """Send an alert through configured channels."""
        # Always log the alert
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"ALERT: {subject} - {message}")
        
        # Send email if enabled
        if self.email_enabled:
            self._send_email(subject, message)
            
        # Send Telegram if enabled
        if self.telegram:
            icon = "ℹ️"
            if level == "WARNING": icon = "⚠️"
            elif level == "ERROR": icon = "❌"
            elif level == "CRITICAL": icon = "🚨"
            
            tg_msg = f"{icon} *{subject}*\n{message}"
            self.telegram.send_message(tg_msg)
            
    def _send_email(self, subject: str, message: str):
        """Send email alert."""
        try:
            msg = MIMEText(message)
            msg['Subject'] = f"[Trading Bot] {subject}"
            msg['From'] = self.smtp_user
            msg['To'] = self.alert_email
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
                
            self.logger.info(f"Email alert sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            
    def alert_trade_executed(self, symbol: str, side: str, quantity: int, price: float):
        """Alert when a trade is executed."""
        self.send_alert(
            subject=f"Trade Executed: {side.upper()} {symbol}",
            message=f"Executed {side} order: {quantity} {symbol} @ ${price:.2f}",
            level="INFO"
        )
        # Using dedicated trade alert format for Telegram if available
        if self.telegram:
            self.telegram.send_trade_alert(symbol, side, price, quantity)
        
    def alert_stop_loss(self, symbol: str, loss_pct: float):
        """Alert when stop-loss is triggered."""
        self.send_alert(
            subject=f"STOP-LOSS: {symbol}",
            message=f"Stop-loss triggered for {symbol}. Loss: {loss_pct:.2%}",
            level="WARNING"
        )
        
    def alert_daily_limit(self, loss_pct: float):
        """Alert when daily loss limit is reached."""
        self.send_alert(
            subject="DAILY LOSS LIMIT REACHED",
            message=f"Trading halted. Daily loss: {loss_pct:.2%}",
            level="CRITICAL"
        )
        
    def alert_error(self, error_message: str):
        """Alert on system errors."""
        self.send_alert(
            subject="System Error",
            message=error_message,
            level="ERROR"
        )


# ============================================================
# PERFORMANCE MONITOR
# ============================================================

class PerformanceMonitor:
    """
    Tracks and reports bot performance metrics.
    """
    
    def __init__(self, metrics_path: str = "logs/metrics.json"):
        self.metrics_path = metrics_path
        self.metrics: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "total_cycles": 0,
            "errors": 0,
            "trades_executed": 0,
            "daily_pnl": {},
            "equity_history": []
        }
        self._load()
        
    def _load(self):
        """Load existing metrics."""
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                self.metrics.update(json.load(f))
                
    def _save(self):
        """Save metrics to disk."""
        Path(os.path.dirname(self.metrics_path)).mkdir(exist_ok=True)
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def record_cycle(self, equity: float):
        """Record a trading cycle completion."""
        self.metrics["total_cycles"] += 1
        self.metrics["equity_history"].append({
            "timestamp": datetime.now().isoformat(),
            "equity": equity
        })
        # Keep only last 1000 data points
        if len(self.metrics["equity_history"]) > 1000:
            self.metrics["equity_history"] = self.metrics["equity_history"][-1000:]
        self._save()
        
    def record_trade(self):
        """Record a trade execution."""
        self.metrics["trades_executed"] += 1
        self._save()
        
    def record_error(self):
        """Record an error occurrence."""
        self.metrics["errors"] += 1
        self._save()
        
    def get_report(self) -> str:
        """Generate a performance report."""
        report = f"""
╔══════════════════════════════════════════════════════╗
║           TRADING BOT PERFORMANCE REPORT             ║
╠══════════════════════════════════════════════════════╣
║ Start Time:      {self.metrics.get('start_time', 'N/A')[:19]}
║ Total Cycles:    {self.metrics.get('total_cycles', 0)}
║ Trades Executed: {self.metrics.get('trades_executed', 0)}
║ Errors:          {self.metrics.get('errors', 0)}
╚══════════════════════════════════════════════════════╝
"""
        return report


# ============================================================
# MAIN MONITORING CLASS
# ============================================================

class BotMonitor:
    """
    Central monitoring class that combines all monitoring features.
    """
    
    def __init__(self, log_dir: str = "logs", alert_config: dict = None):
        # Setup logging
        self.logger = setup_logging(log_dir)
        
        # Initialize components
        # self.journal = TradeJournal(f"{log_dir}/trade_journal.json") # Deprecated replaced by DB
        from database import Database
        self.db = Database()
        
        # Legacy support
        if alert_config is None: alert_config = {}
        
        self.alerts = AlertManager(alert_config)
        self.performance = PerformanceMonitor(f"{log_dir}/metrics.json")
        
        self.logger.info("Bot monitoring system initialized (SQLite Enabled)")
        
    def log_trade(self, trade: TradeRecord):
        """Log a trade execution."""
        # self.journal.record_trade(trade)
        self.db.log_trade(asdict(trade))
        
        self.performance.record_trade()
        self.alerts.alert_trade_executed(
            trade.symbol, trade.side, trade.quantity, trade.price
        )
        
    def get_recent_trades(self, limit=50):
        return self.db.get_trades(limit)
        
    def log_cycle(self, equity: float):
        """Log a trading cycle."""
        self.performance.record_cycle(equity)
        
    def log_error(self, error: str):
        """Log an error."""
        self.logger.error(error)
        self.performance.record_error()
        self.alerts.alert_error(error)
        
    def get_status(self) -> str:
        """Get current bot status."""
        stats = self.db.get_stats()
        report = self.performance.get_report()
        
        return report + f"""
Trade Summary:
  Total Trades: {stats.get('total_trades', 0)}
  Win Rate: {stats.get('win_rate', 0):.1%}
  Total P&L: ${stats.get('total_pnl', 0):.2f}
"""


if __name__ == "__main__":
    # Test the monitoring system
    print("Testing Monitoring System...")
    
    monitor = BotMonitor(log_dir="logs")
    
    # Log a test trade
    trade = TradeRecord(
        timestamp=datetime.now().isoformat(),
        symbol="AAPL",
        side="buy",
        quantity=10,
        price=150.0,
        order_id="TEST-001"
    )
    monitor.log_trade(trade)
    
    # Log a cycle
    monitor.log_cycle(equity=10150.0)
    
    # Get status
    print(monitor.get_status())
    
    print("Monitoring system test complete!")
