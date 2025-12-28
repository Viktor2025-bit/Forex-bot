import yfinance as yf
import pandas as pd
import os

def download_data(pair="EURUSD=X", period="1y", interval="1h"):
    """
    Downloads historical data from Yahoo Finance.
    
    Args:
        pair (str): The currency pair symbol (e.g., "EURUSD=X").
        period (str): The time period to download (e.g., "1y", "2y", "max").
        interval (str): The data interval (e.g., "1h", "1d", "1m").
        
    Returns:
        str: The path to the saved CSV file.
    """
    print(f"Downloading data for {pair} (Period: {period}, Interval: {interval})...")
    
    # Download data
    data = yf.download(pair, period=period, interval=interval)
    
    if not data.empty:
        # Create 'data' directory if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Save to CSV
        filename = f"data/{pair.replace('=X', '')}_{period}_{interval}.csv"
        data.to_csv(filename)
        
        print(f"Success! Data saved to: {filename}")
        print(f"Rows: {len(data)}")
        print(data.tail())
        return filename
    else:
        print("Error: No data found. Check your internet connection or the ticker symbol.")
        return None

if __name__ == "__main__":
    download_data()
