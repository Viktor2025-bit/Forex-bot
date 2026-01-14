try:
    import yfinance as yf
except Exception as e:
    print(f"Warning: Failed to import yfinance: {e}")
    yf = None
import pandas as pd
import os

def fetch_historical_data(ticker, start_date, end_date, interval="1d"):
    """
    Fetches historical data for a given ticker or list of tickers using yfinance.
    
    Args:
        ticker (str or list): Asset symbol or list of symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Data interval (e.g., "1d", "1h", "15m").
        
    Returns:
        pd.DataFrame or dict: DataFrame if single ticker, dict of DataFrames if list.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    
    if isinstance(ticker, list):
        data_dict = {}
        for t in ticker:
            d = fetch_historical_data(t, start_date, end_date, interval)
            if d is not None:
                data_dict[t] = d
        return data_dict
        
    if yf is None:
        print("Error: yfinance module not available.")
        return None

    # Ticker Mapping for Indices (Broker -> Yahoo)
    ticker_map = {
        "US500": "^GSPC",
        "US30": "^DJI",
        "NAS100": "^NDX",
        "GER40": "^GDAXI",
        "UK100": "^FTSE"
    }
    
    # Use mapped ticker if available, else original
    search_ticker = ticker_map.get(ticker, ticker)

    data = yf.download(search_ticker, start=start_date, end=end_date, interval=interval, progress=False)
    
    if data.empty:
        # Try finding futures if index fails (e.g., GC=F for Gold)
        print(f"Warning: No data found for {ticker} (mapped: {search_ticker}).")
        return None
        
    return data


def save_data(data_dict, folder="data/raw"):
    """
    Saves the dataframes to CSV files.
    """
    if not data_dict:
        return
        
    os.makedirs(folder, exist_ok=True)
    filenames = []
    for ticker, df in data_dict.items():
        if df is not None:
            filename = f"{folder}/{ticker.replace('=X', '')}_{df.index[0].date()}_to_{df.index[-1].date()}.csv"
            df.to_csv(filename)
            print(f"Data for {ticker} saved to {filename}")
            filenames.append(filename)
    return filenames

def fetch_data_mt5(tickers, n_candles=1000, timeframe="M5"):
    """
    Fetches real-time candle data from MetaTrader 5.
    Args:
        tickers (str or list): Symbols to fetch.
        n_candles (int): Number of candles.
        timeframe (str): "M1", "M5", "H1", "D1".
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 module not found.")
        return None

    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return None

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)

    data_dict = {}
    
    if isinstance(tickers, str):
        tickers = [tickers]
        
    for ticker in tickers:
        # Map Yahoo symbol to MT5 symbol (remove =X)
        mt5_symbol = ticker.replace("=X", "")
        
        # Ensure symbol is selected in Market Watch
        if not mt5.symbol_select(mt5_symbol, True):
             print(f"Failed to select {mt5_symbol} in MT5")
             continue

        rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_tf, 0, n_candles)
        
        if rates is None or len(rates) == 0:
            print(f"No data for {mt5_symbol}")
            continue
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename columns to match Yahoo Finance format
        df.rename(columns={
            'time': 'Date', 
            'open': 'Open', 
            'high': 'High', 
            'low': 'Low', 
            'close': 'Close', 
            'tick_volume': 'Volume'
        }, inplace=True)
        
        df.set_index('Date', inplace=True)
        
        # Drop potentially unused columns from MT5 (spread, real_volume)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        data_dict[ticker] = df # Keep original key (e.g. EURUSD=X) for compatibility
        
    return data_dict

if __name__ == "__main__":
    # Test execution
    stock_tickers = ["AAPL", "MSFT"]
    data = fetch_historical_data(stock_tickers, "2023-01-01", "2023-12-31")
    save_data(data)

    forex_tickers = ["EURUSD=X", "GBPUSD=X"]
    data = fetch_historical_data(forex_tickers, "2023-01-01", "2023-12-31")
    save_data(data)
