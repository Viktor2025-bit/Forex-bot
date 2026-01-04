
import MetaTrader5 as mt5
import pandas as pd

def scan_indices():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return

    # Common search terms for indices
    search_terms = ["CRASH", "BOOM", "R_"]
    
    found_symbols = []
    all_symbols = mt5.symbols_get()
    
    if all_symbols:
        for s in all_symbols:
            for term in search_terms:
                if term in s.name.upper():
                    found_symbols.append(s.name)
                    break
    
    print("\n--- FOUND INDICES ---")
    for sym in sorted(list(set(found_symbols))):
        print(sym)
        
    mt5.shutdown()

if __name__ == "__main__":
    scan_indices()
