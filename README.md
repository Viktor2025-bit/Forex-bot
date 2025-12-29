# AI Forex Trading Bot

An AI-powered trading bot using LSTM neural networks for price direction prediction.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python run_backtest.py
```
This will download data, train the model, and show backtest results.

### 3. Run Paper Trading
```bash
python trading_bot.py
```

## 📁 Project Structure

```
├── models/             # ML model definitions
│   └── lstm_model.py   # LSTM neural network
├── strategies/         # Trading strategies & execution
│   ├── backtester.py   # Backtesting engine
│   ├── order_executor.py  # Order execution (Paper/Live)
│   └── risk_manager.py    # Risk management
├── utils/              # Utilities
│   ├── data_loader.py  # Data fetching (yfinance)
│   ├── preprocessing.py   # Data preprocessing
│   └── monitoring.py   # Logging & alerts
├── logs/               # Log files
├── config.json         # Bot configuration
├── train.py            # Model training script
├── run_backtest.py     # Train + backtest pipeline
└── trading_bot.py      # Main trading bot
```

## ⚙️ Configuration

Edit `config.json` to customize:
- Trading symbol and capital
- Risk parameters (stop-loss, take-profit)
- Broker credentials (Alpaca/OANDA/MT5)
- Alert settings

## 📊 Features

- **LSTM Model**: Predicts price direction (up/down)
- **Backtesting**: Test strategies on historical data
- **Paper Trading**: Test with fake money
- **Risk Management**: Stop-loss, take-profit, position sizing
- **Monitoring**: Logging, trade journal, performance metrics

## 🔧 Supported Brokers

- **Paper Trading** (built-in)
- **Alpaca** (stocks/crypto)
- **OANDA** (forex) - coming soon
- **MT5** (forex) - coming soon

## ⚠️ Disclaimer

This bot is for educational purposes. Trading involves substantial risk of loss. Never trade with money you cannot afford to lose.
