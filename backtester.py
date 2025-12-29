import pandas as pd
import numpy as np

class Backtester:
    """
    Backtesting engine to evaluate trading strategy performance.
    Calculates P&L, win rate, drawdown, and other key metrics.
    """
    
    def __init__(self, data, initial_capital=10000, risk_manager=None):
        """
        Initialize the backtester.
        
        Args:
            data (pd.DataFrame): DataFrame with 'Close', 'Signal', and 'Position' columns
            initial_capital (float): Starting capital in USD
            risk_manager (RiskManager, optional): Risk management system for position sizing and stop-loss/take-profit
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.risk_manager = risk_manager
        self.trades = []
        self.results = {}
        
        # Track risk management statistics
        self.stop_loss_hits = 0
        self.take_profit_hits = 0
        self.signal_exits = 0
        
    def run(self):
        """
        Execute the backtest and calculate all metrics.
        
        Returns:
            dict: Dictionary containing all performance metrics
        """
        print("\n=== Running Backtest ===")
        
        # Calculate Strategy Returns
        self._calculate_returns()
        
        # Track Individual Trades
        self._track_trades()
        
        # Calculate Performance Metrics
        self._calculate_metrics()
        
        # Calculate Buy-and-Hold Baseline
        self._calculate_baseline()
        
        return self.results
    
    def _calculate_returns(self):
        """Calculate daily returns and cumulative strategy returns."""
        # Daily returns based on close prices
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        
        # Strategy returns: Only earn returns when holding a position
        # Signal = 1 (long), -1 (short), 0 (no position)
        self.data['Strategy_Return'] = self.data['Daily_Return'] * self.data['Signal'].shift(1)
        
        # Cumulative returns
        self.data['Cumulative_Return'] = (1 + self.data['Strategy_Return']).cumprod()
        self.data['Cumulative_Market_Return'] = (1 + self.data['Daily_Return']).cumprod()
        
    def _track_trades(self):
        """Track individual trade entry/exit points and P&L with risk management."""
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        entry_date = None
        stop_loss = None
        take_profit = None
        position_size = 1.0  # Default to full position if no risk manager
        current_balance = self.initial_capital
        entry_balance = 0
        portfolio_values_tracked = []
        
        for i, (date, row) in enumerate(self.data.iterrows()):
            # Check if we have an open position and risk management is enabled
            if position != 0 and self.risk_manager is not None and stop_loss is not None:
                direction = 'LONG' if position == 1 else 'SHORT'
                
                # Check if stop-loss was hit (using Low for long, High for short)
                sl_price = row['Low'] if direction == 'LONG' else row['High']
                if self.risk_manager.check_stop_loss_hit(sl_price, stop_loss, direction):
                    # Exit at stop-loss
                    exit_price = stop_loss
                    pnl_pct = ((exit_price - entry_price) / entry_price) * position
                    pnl_dollar = entry_balance * pnl_pct * position_size
                    
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': direction,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_dollar': pnl_dollar,
                        'exit_reason': 'STOP_LOSS',
                        'position_size': position_size
                    })
                    
                    current_balance += pnl_dollar
                    position = 0
                    self.stop_loss_hits += 1
                    portfolio_values_tracked.append(current_balance)
                    continue
                
                # Check if take-profit was hit (using High for long, Low for short)
                tp_price = row['High'] if direction == 'LONG' else row['Low']
                if self.risk_manager.check_take_profit_hit(tp_price, take_profit, direction):
                    # Exit at take-profit
                    exit_price = take_profit
                    pnl_pct = ((exit_price - entry_price) / entry_price) * position
                    pnl_dollar = entry_balance * pnl_pct * position_size
                    
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': direction,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_dollar': pnl_dollar,
                        'exit_reason': 'TAKE_PROFIT',
                        'position_size': position_size
                    })
                    
                    current_balance += pnl_dollar
                    position = 0
                    self.take_profit_hits += 1
                    portfolio_values_tracked.append(current_balance)
                    continue
            
            # Check for position changes from strategy signals
            if pd.isna(row['Position']) or row['Position'] == 0:
                portfolio_values_tracked.append(current_balance)
                continue
                
            # Entry signal: Position changed from 0 or opposite direction
            if position == 0 or (position == 1 and row['Position'] < 0) or (position == -1 and row['Position'] > 0):
                # Close previous position if exists (signal-based exit)
                if position != 0 and entry_price != 0:
                    exit_price = row['Close']
                    pnl_pct = ((exit_price - entry_price) / entry_price) * position
                    pnl_dollar = entry_balance * pnl_pct * position_size
                    
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'LONG' if position == 1 else 'SHORT',
                        'pnl_pct': pnl_pct * 100,
                        'pnl_dollar': pnl_dollar,
                        'exit_reason': 'SIGNAL',
                        'position_size': position_size
                    })
                    
                    current_balance += pnl_dollar
                    self.signal_exits += 1
                    portfolio_values_tracked.append(current_balance)
                
                # Check if trading is still enabled (drawdown protection)
                if self.risk_manager is not None:
                    trading_enabled, _ = self.risk_manager.check_drawdown(current_balance, self.initial_capital)
                    if not trading_enabled:
                        print(f"\n⚠️ Maximum drawdown exceeded. Trading halted at {date}.")
                        position = 0
                        break
                    
                    # Calculate position size and risk levels
                    position_size = self.risk_manager.calculate_position_size(current_balance, row['Close'])
                else:
                    position_size = 1.0
                
                # Open new position
                position = 1 if row['Signal'] == 1 else -1
                entry_price = row['Close']
                entry_date = date
                entry_balance = current_balance
                
                # Set stop-loss and take-profit if risk manager exists
                if self.risk_manager is not None:
                    direction = 'LONG' if position == 1 else 'SHORT'
                    stop_loss = self.risk_manager.calculate_stop_loss(entry_price, direction)
                    take_profit = self.risk_manager.calculate_take_profit(entry_price, direction)
                else:
                    stop_loss = None
                    take_profit = None
        
        # Close final position if still open
        if position != 0 and entry_price != 0:
            exit_price = self.data['Close'].iloc[-1]
            pnl_pct = ((exit_price - entry_price) / entry_price) * position
            pnl_dollar = entry_balance * pnl_pct * position_size
            
            self.trades.append({
                'entry_date': entry_date,
                'exit_date': self.data.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'LONG' if position == 1 else 'SHORT',
                'pnl_pct': pnl_pct * 100,
                'pnl_dollar': pnl_dollar,
                'exit_reason': 'END_OF_DATA',
                'position_size': position_size
            })
        
        self.data['Portfolio_Value_Tracked'] = portfolio_values_tracked

    
    def _calculate_metrics(self):
        """Calculate key performance metrics."""
        if len(self.trades) == 0:
            print("⚠️ No trades executed during backtest period.")
            self.results = {
                'total_trades': 0,
                'total_return_pct': 0,
                'total_return_dollar': 0,
                'win_rate': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'final_portfolio_value': self.initial_capital
            }
            return
        
        # Convert trades to DataFrame for easier analysis
        trades_df = pd.DataFrame(self.trades)
        
        # Basic Stats
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl_dollar'] > 0]
        losing_trades = trades_df[trades_df['pnl_dollar'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Returns
        total_return_pct = ((self.data['Portfolio_Value_Tracked'].iloc[-1] - self.initial_capital) / self.initial_capital) * 100
        total_return_dollar = self.data['Portfolio_Value_Tracked'].iloc[-1] - self.initial_capital
        
        # Average Win/Loss
        avg_win_pct = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss_pct = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        
        # Maximum Drawdown
        cumulative = self.data['Portfolio_Value_Tracked']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (annualized, assuming 252 trading days per year)
        strategy_returns = self.data['Strategy_Return'].dropna()
        if len(strategy_returns) > 0 and strategy_returns.std() != 0:
            sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Store results
        self.results = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'total_return_pct': total_return_pct,
            'total_return_dollar': total_return_dollar,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_portfolio_value': self.data['Portfolio_Value_Tracked'].iloc[-1],
            'trades_list': trades_df
        }
    
    def _calculate_baseline(self):
        """Calculate buy-and-hold baseline for comparison."""
        buy_hold_return = ((self.data['Close'].iloc[-1] - self.data['Close'].iloc[0]) / 
                          self.data['Close'].iloc[0]) * 100
        buy_hold_value = self.initial_capital * (1 + buy_hold_return / 100)
        
        self.results['buy_hold_return_pct'] = buy_hold_return
        self.results['buy_hold_final_value'] = buy_hold_value
        self.results['outperformance'] = self.results['total_return_pct'] - buy_hold_return
    
    def print_summary(self):
        """Print a formatted summary of backtest results."""
        if not self.results:
            print("⚠️ No results to display. Run backtest first.")
            return
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nRETURNS:")
        print(f"  Initial Capital:        ${self.initial_capital:,.2f}")
        print(f"  Final Portfolio Value:  ${self.results['final_portfolio_value']:,.2f}")
        print(f"  Total Return:           {self.results['total_return_pct']:.2f}% (${self.results['total_return_dollar']:,.2f})")
        
        print(f"\nTRADING STATISTICS:")
        print(f"  Total Trades:           {self.results['total_trades']}")
        print(f"  Winning Trades:         {self.results['winning_trades']}")
        print(f"  Losing Trades:          {self.results['losing_trades']}")
        print(f"  Win Rate:               {self.results['win_rate']:.2f}%")
        print(f"  Avg Win:                {self.results['avg_win_pct']:.2f}%")
        print(f"  Avg Loss:               {self.results['avg_loss_pct']:.2f}%")
        
        print(f"\nRISK METRICS:")
        print(f"  Max Drawdown:           {self.results['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio:           {self.results['sharpe_ratio']:.2f}")
        
        # Show risk management statistics if enabled
        if self.risk_manager is not None:
            total_exits = self.stop_loss_hits + self.take_profit_hits + self.signal_exits
            print(f"\nRISK MANAGEMENT STATS:")
            print(f"  Stop-Loss Hits:         {self.stop_loss_hits} ({self.stop_loss_hits/total_exits*100:.1f}%)" if total_exits > 0 else "  Stop-Loss Hits:         0")
            print(f"  Take-Profit Hits:       {self.take_profit_hits} ({self.take_profit_hits/total_exits*100:.1f}%)" if total_exits > 0 else "  Take-Profit Hits:       0")
            print(f"  Signal-Based Exits:     {self.signal_exits} ({self.signal_exits/total_exits*100:.1f}%)" if total_exits > 0 else "  Signal-Based Exits:     0")
            rm_metrics = self.risk_manager.get_risk_metrics()
            print(f"  Risk Per Trade:         {rm_metrics['risk_per_trade_pct']}%")
            print(f"  Stop-Loss Distance:     {rm_metrics['stop_loss_pct']}%")
            print(f"  Take-Profit Distance:   {rm_metrics['take_profit_pct']}%")
            print(f"  Risk-Reward Ratio:      1:{rm_metrics['risk_reward_ratio']}")

        
        print(f"\nBASELINE COMPARISON:")
        print(f"  Buy & Hold Return:      {self.results['buy_hold_return_pct']:.2f}%")
        print(f"  Buy & Hold Final Value: ${self.results['buy_hold_final_value']:,.2f}")
        print(f"  Strategy Outperformance: {self.results['outperformance']:.2f}%")
        
        print("\n" + "="*60)
        
        # Show last 5 trades
        if len(self.results['trades_list']) > 0:
            print("\nLAST 5 TRADES:")
            print(self.results['trades_list'].tail(5).to_string(index=False))
        
        print("\n")
