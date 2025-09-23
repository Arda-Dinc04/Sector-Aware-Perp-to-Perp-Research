import pandas as pd
import os
from dotenv import load_dotenv

# Load .env in case you want to use paths or configs
load_dotenv()

# Example: folder where parquet files are saved
DATA_DIR = os.getenv("DATA_DIR", "data/binance_30m")

def inspect_parquet(symbol: str):
    """
    Load and inspect a parquet file for a given symbol
    """
    file_path = os.path.join(DATA_DIR, f"{symbol}.parquet")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Load parquet file into Pandas DataFrame
    df = pd.read_parquet(file_path)
    
    print(f"✅ Loaded {symbol} with {len(df)} rows and {len(df.columns)} columns")
    print("\n📌 Columns:", list(df.columns))
    
    # Show first 5 rows
    print("\n🔹 Head (first 5 rows):")
    print(df.head())
    
    # Show last 5 rows
    print("\n🔹 Tail (last 5 rows):")
    print(df.tail())
    
    # Summary statistics
    print("\n📊 Summary statistics:")
    print(df.describe(include="all"))

    # Check for missing values
    print("\n🧩 Missing values per column:")
    print(df.isna().sum())

if __name__ == "__main__":
    # Example: inspect BTCUSDT file
    inspect_parquet("BTCUSDT")
