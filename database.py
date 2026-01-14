import sqlite3
import json
from datetime import datetime
import os

class Database:
    def __init__(self, db_path="trading_bot.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades Table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            side TEXT,
            quantity REAL,
            price REAL,
            order_id TEXT,
            pnl REAL,
            notes TEXT,
            strategy TEXT
        )
        ''')
        
        # Signals/Logs Table (Optional for now, but good for Phase 2)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            signal_data TEXT,
            confidence REAL
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def log_trade(self, trade_data: dict):
        """Log a trade to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO trades (timestamp, symbol, side, quantity, price, order_id, pnl, notes, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('timestamp'),
                trade_data.get('symbol'),
                trade_data.get('side'),
                trade_data.get('quantity'),
                trade_data.get('price'),
                trade_data.get('order_id'),
                trade_data.get('pnl'),
                trade_data.get('notes'),
                trade_data.get('strategy', 'AI')
            ))
            
            conn.commit()
            conn.close()
            print(f"Trade logged to DB: {trade_data.get('symbol')} {trade_data.get('side')}")
        except Exception as e:
            print(f"DB Error: {e}")

    def get_trades(self, limit=100):
        """Retrieve recent trades."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM trades ORDER BY id DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        
        trades = [dict(row) for row in rows]
        conn.close()
        return trades
        
    def get_stats(self):
        """Get trading statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*), SUM(pnl) FROM trades WHERE side="CLOSE"')
        close_stats = cursor.fetchone()
        
        total_pnl = close_stats[1] if close_stats[1] else 0.0
        
        cursor.execute('SELECT COUNT(*) FROM trades WHERE side="CLOSE" AND pnl > 0')
        wins = cursor.fetchone()[0]
        
        total_trades = close_stats[0]
        win_rate = (wins / total_trades) if total_trades > 0 else 0.0
        
        conn.close()
        
        return {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "win_rate": win_rate
        }

if __name__ == "__main__":
    db = Database()
    print("Database initialized.")
