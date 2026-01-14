import itertools
import pandas as pd
import numpy as np
from strategies.risk_manager import RiskManager
from backtester import Backtester
from utils.backtest_utils import simple_ma_strategy # Default strategy for now

class GridOptimizer:
    def __init__(self, data, initial_capital=200.0):
        self.data = data
        self.initial_capital = initial_capital
        
    def optimize(self, param_grid):
        """
        Run grid search over parameters.
        param_grid: dict of list, e.g. {'sl': [0.01, 0.02], 'tp': [0.02, 0.04]}
        """
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        results = []
        
        # Pre-calculate strategy signals ONCE if strategy params aren't being optimized
        # For now, we assume simple_ma_strategy is fixed or we optimize its params too?
        # Let's support SL/TP optimization primarily as requested.
        
        processed_data = simple_ma_strategy(self.data.copy())
        
        print(f"Starting optimization with {len(combinations)} combinations...")
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            
            # Setup Risk Manager with these params
            # We mock the configuration structure expected by RiskManager
            # Actually Backtester takes an optional RiskManager.
            
            # Create a mock RiskManager or just use Backtester's manual logic?
            # Backtester uses RiskManager if provided.
            
            # Let's create a dynamic config wrapper
            class MockRiskParams:
                def __init__(self, sl, tp, risk_per_trade):
                    self.stop_loss_pct = sl
                    self.take_profit_pct = tp
                    self.risk_per_trade_pct = risk_per_trade
                    self.max_daily_loss_pct = 0.10
                    self.max_position_size_pct = 1.0
                    self.trailing_stop_pct = 0.0 # Disable for basic grid search
                    self.trailing_activation_pct = 0.0
                    
                    # Forex params defaults
                    self.forex_risk = type('obj', (object,), {
                        'risk_per_trade_pct': risk_per_trade,
                        'stop_loss_pips': int(sl * 10000), # Approx
                        'risk_reward_ratio': tp/sl if sl > 0 else 1
                    })
                    
            # We need to monkey patch or modify RiskManager to accept this simple struct
            # Or better, we just manually inject the logic into a custom RiskManager
            # But simpler: The Backtester uses methods like `check_stop_loss_hit`.
            # Let's instantiate a real RiskManager but with a modified config dict.
            
            sim_config = {
                "risk": {
                    "max_position_size_pct": 1.0,
                    "max_daily_loss_pct": 0.10,
                    "risk_per_trade_pct": params.get('risk_per_trade', 0.01),
                    "stop_loss_pct": params.get('sl', 0.02),
                    "take_profit_pct": params.get('tp', 0.04),
                    "trailing_stop_pct": 0.0,
                    "trailing_activation_pct": 0.0
                },
                "bot": {
                     "initial_capital": self.initial_capital
                }
            }
            
            # Initialize RiskManager
            # RiskManager.__init__ expects "config" dict.
            rm = RiskManager(sim_config)
            
            # Run Backtest
            tester = Backtester(processed_data.copy(), initial_capital=self.initial_capital, risk_manager=rm)
            res = tester.run()
            
            # Collect Metric
            results.append({
                **params,
                "Total Return %": res['total_return_pct'],
                "Win Rate": res['win_rate'],
                "Max Drawdown": res['max_drawdown_pct'],
                "Sharpe": res['sharpe_ratio'],
                "Trades": res['total_trades']
            })
            
        return pd.DataFrame(results).sort_values(by="Total Return %", ascending=False)
