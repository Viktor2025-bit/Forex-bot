"""Test script to verify the Forex-specific functionalities."""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.risk_manager import RiskManager, RiskParameters, ForexRiskParameters

def test_forex_risk_manager():
    """Tests the Forex-specific methods of the RiskManager."""
    print("\n[1] Testing Forex Risk Manager...")
    
    # Setup
    forex_params = ForexRiskParameters(
        risk_per_trade_pct=0.01,  # Risk 1%
        stop_loss_pips=50,
        risk_reward_ratio=2.0
    )
    params = RiskParameters(forex_risk=forex_params)
    rm = RiskManager(params)

    portfolio_value = 10000  # $10,000 account
    symbol = "EUR/USD"
    
    # Test 1: Position Sizing
    # Expectation: Risking 1% of $10k is $100. With a 50 pip SL and ~$1/pip value,
    # we expect $100 / (50 * $1) = 2 mini lots, which is 20,000 units.
    expected_units = 20000
    units = rm.calculate_forex_position_size(portfolio_value, symbol)
    if units == expected_units:
        print(f"   ✓ Position Sizing: Correctly calculated {units} units.")
    else:
        print(f"   ✗ Position Sizing: Expected {expected_units}, but got {units}.")
        sys.exit(1)
        
    # Test 2: Stop-Loss
    entry_price = 1.10000
    # Price moves 50 pips against a long position
    bad_price = entry_price - (forex_params.stop_loss_pips * 0.0001)
    sl_triggered = rm.check_forex_stop_loss(entry_price, bad_price, symbol, 'long')
    if sl_triggered:
        print("   ✓ Stop-Loss: Correctly triggered at 50 pips.")
    else:
        print("   ✗ Stop-Loss: Failed to trigger.")
        sys.exit(1)
        
    # Test 3: Take-Profit
    # Expectation: TP should be 50 pips * 2.0 R:R = 100 pips
    good_price = entry_price + (forex_params.stop_loss_pips * forex_params.risk_reward_ratio * 0.0001)
    tp_triggered = rm.check_forex_take_profit(entry_price, good_price, symbol, 'long')
    if tp_triggered:
        print(f"   ✓ Take-Profit: Correctly triggered at {forex_params.stop_loss_pips * forex_params.risk_reward_ratio} pips.")
    else:
        print("   ✗ Take-Profit: Failed to trigger.")
        sys.exit(1)

    # Test 4: JPY pair pip size
    symbol_jpy = "USD/JPY"
    entry_price_jpy = 130.00
    bad_price_jpy = entry_price_jpy - (forex_params.stop_loss_pips * 0.01)
    sl_triggered_jpy = rm.check_forex_stop_loss(entry_price_jpy, bad_price_jpy, symbol_jpy, 'long')
    if sl_triggered_jpy:
        print("   ✓ JPY Stop-Loss: Correctly triggered for JPY pair.")
    else:
        print("   ✗ JPY Stop-Loss: Failed to trigger for JPY pair.")
        sys.exit(1)

    print("   ✓ All Forex risk manager tests passed!")


def main():
    print("=" * 50)
    print("Testing Forex Functionalities...")
    print("=" * 50)
    
    test_forex_risk_manager()
    
    print("\n" + "=" * 50)
    print("All Forex tests passed! ✓")
    print("=" * 50)

if __name__ == "__main__":
    main()
