import pandas as pd
from data_loader import download_data
from strategy_engine import MovingAverageCrossover
from risk_manager import RiskManager
from backtester import Backtester
import matplotlib.pyplot as plt

def run_bot():
    print("=== AI Forex Bot Starting ===")
    
    # 1. Get Data
    # Using EURUSD data for the last 2 years, 1-hour interval
    file_path = download_data(pair="EURUSD=X", period="2y", interval="1h")
    
    if not file_path:
        print("Failed to get data. Exiting.")
        return

    # Load data into DataFrame
    # yfinance creates a multi-row header, so we skip the first 2 rows
    # and manually assign column names
    df = pd.read_csv(file_path, skiprows=[0, 1], 
                     names=['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume'],
                     parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.index.name = 'Date'
    
    # 2. Feed Strategy
    # Initialize the strategy (e.g., Simple Moving Average Crossover)
    strategy = MovingAverageCrossover(df, short_window=50, long_window=200)
    
    # Execute Strategy to get signals
    processed_data = strategy.generate_signals()
    
    # 3. Backtest the Strategy
    print("\n" + "="*60)
    print("BACKTESTING STRATEGY PERFORMANCE")
    print("="*60)
    
    # Initialize Risk Manager
    risk_manager = RiskManager(
        risk_per_trade=2.0,        # Risk 2% of capital per trade
        stop_loss_pct=2.0,         # 2% stop-loss
        risk_reward_ratio=2.0,     # Target 2:1 reward-to-risk
        max_drawdown_pct=15.0      # Halt if 15% drawdown
    )
    
    print("\nRisk Management Active:")
    print(f"  • Risk Per Trade: {risk_manager.risk_per_trade}%")
    print(f"  • Stop-Loss:      {risk_manager.stop_loss_pct}%")
    print(f"  • Risk/Reward:    1:{risk_manager.risk_reward_ratio}")
    print(f"  • Max Drawdown:   {risk_manager.max_drawdown_pct}%")
    
    backtester = Backtester(processed_data, initial_capital=10000, risk_manager=risk_manager)
    results = backtester.run()
    backtester.print_summary()
    
    # 4. Analyze Signals
    print("\nStrategy Signal Analysis:")
    print(processed_data[['Close', 'Short_MA', 'Long_MA', 'Signal', 'Position']].tail(10))
    
    # Count crossovers (Position = 2 means bullish crossover, -2 means bearish crossover)
    buy_signals = processed_data[processed_data['Position'] > 1]
    sell_signals = processed_data[processed_data['Position'] < -1]
    
    print(f"\nTotal Buy Crossovers: {len(buy_signals)}")
    print(f"Total Sell Crossovers: {len(sell_signals)}")
    
    # 5. Visualize Strategy & Portfolio Performance
    print("\nGenerating visualizations...")
    
    # Create a 2-subplot figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Subplot 1: Price & Moving Averages with Signals
    ax1.plot(processed_data.index, processed_data['Close'], label='Close Price', alpha=0.5, linewidth=1.5)
    ax1.plot(processed_data.index, processed_data['Short_MA'], label='50 SMA', alpha=0.9, color='orange', linewidth=1.5)
    ax1.plot(processed_data.index, processed_data['Long_MA'], label='200 SMA', alpha=0.9, color='purple', linewidth=1.5)
    
    # Plot Buy signals (Green Triangle Up)
    if len(buy_signals) > 0:
        ax1.plot(buy_signals.index, 
                 processed_data.loc[buy_signals.index]['Short_MA'], 
                 '^', markersize=10, color='g', lw=0, label='Buy Signal')
    
    # Plot Sell signals (Red Triangle Down)
    if len(sell_signals) > 0:
        ax1.plot(sell_signals.index, 
                 processed_data.loc[sell_signals.index]['Short_MA'], 
                 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    
    ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('EUR/USD - Moving Average Crossover Strategy', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Portfolio Value Over Time
    ax2.plot(processed_data.index, processed_data['Portfolio_Value'], 
             label='Strategy Portfolio', color='blue', linewidth=2)
    ax2.axhline(y=backtester.initial_capital, color='gray', linestyle='--', 
                label=f'Initial Capital (${backtester.initial_capital:,.0f})', alpha=0.7)
    
    # Add buy-and-hold comparison
    buy_hold_line = backtester.initial_capital * processed_data['Cumulative_Market_Return']
    ax2.plot(processed_data.index, buy_hold_line, 
             label='Buy & Hold', color='orange', linewidth=2, alpha=0.7, linestyle='--')
    
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Portfolio Value (USD)', fontsize=12, fontweight='bold')
    ax2.set_title('Portfolio Performance vs Buy & Hold', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Add performance annotation
    final_value = results['final_portfolio_value']
    total_return = results['total_return_pct']
    color = 'green' if total_return > 0 else 'red'
    ax2.text(0.02, 0.98, f"Final Value: ${final_value:,.2f}\nReturn: {total_return:.2f}%", 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    
    plt.tight_layout()
    
    # Save the chart
    chart_path = 'strategy_results.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Chart saved to: {chart_path}")
    print("Open this file to see the complete strategy analysis!")

if __name__ == "__main__":
    run_bot()
