#!/usr/bin/env python3
import pandas as pd
import pyarrow.parquet as pq
import sys
from pathlib import Path

def read_parquet_file(file_path):
    """
    Read and display basic information about a parquet file.
    
    Args:
        file_path (str): Path to the parquet file
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return
    
    if not path.suffix.lower() == '.parquet':
        print(f"Warning: '{file_path}' does not have a .parquet extension.")
    
    try:
        # Read with pandas
        df = pd.read_parquet(file_path)
        
        print(f"Successfully read: {path.name}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # Show first few rows
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"\nBasic statistics:")
            print(df[numeric_cols].describe())
        
        return df
        
    except Exception as e:
        print(f"Error reading parquet file: {str(e)}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python read_parquet.py <parquet_file_path>")
        print("Example: python read_parquet.py data/sample.parquet")
        sys.exit(1)
    
    file_path = sys.argv[1]
    read_parquet_file(file_path)

if __name__ == "__main__":
    main()