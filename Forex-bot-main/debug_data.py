import pandas as pd
from data_loader import download_data
from feature_engine import FeatureEngine

print("=== DEBUG DATA LOADING ===")
file_path = download_data(pair="EURUSD=X", period="1mo", interval="1h")
print(f"File path: {file_path}")

if file_path:
    df = pd.read_csv(file_path, skiprows=[0, 1], 
                     names=['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume'],
                     parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    print(f"Loaded DataFrame shape: {df.shape}")
    print("Head:")
    print(df.head())
    print("Tail:")
    print(df.tail())
    
    print("\n=== DEBUG FEATURE ENGINE ===")
    fe = FeatureEngine()
    
    # Check nulls BEFORE feature gen
    print("Nulls before features:")
    print(df.isnull().sum())
    
    # Generate features
    # (We copy the code manually here effectively to see where it breaks if needed, 
    # but let's call the class first)
    try:
        df_processed = fe.generate_features(df)
        print(f"Processed shape: {df_processed.shape}")
    except Exception as e:
        print(f"Feature generation failed: {e}")

else:
    print("Failed to download data.")
