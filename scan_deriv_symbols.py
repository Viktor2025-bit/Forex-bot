
import asyncio
from deriv_api import DerivAPI
import json
import sys

# Windows Unicode fix
sys.stdout.reconfigure(encoding='utf-8')

async def scan():
    # Try common app_id
    api = DerivAPI(app_id=1089)
    try:
        # Fetch active symbols
        response = await api.active_symbols({"active_symbols": "brief", "product_type": "basic"})
        symbols = response['active_symbols']
        
        print("\n--- DERIV SYMBOLS (Crash/Boom/Vol) ---")
        for s in symbols:
            name = s['symbol']
            display = s['display_name']
            market = s['market']
            
            # Filter for Synthetics (often market='synthetic_index' or similar)
            # or name containing relevant keywords
            if "Crash" in display or "Boom" in display or "Range" in display or "Index" in display:
                 print(f"Code: {name:<15} | Name: {display}")
                 
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await api.clear()

if __name__ == "__main__":
    asyncio.run(scan())
