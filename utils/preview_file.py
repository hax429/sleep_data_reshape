import pandas as pd
import os

def preview_file(file_path, num_rows=10):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_ext == '.csv':
            df = pd.read_csv(file_path)
        else:
            print(f"Error: Unsupported file type '{file_ext}'. Only .parquet and .csv files are supported.")
            return
        
        total_rows = len(df)
        print(f"File: {os.path.basename(file_path)}")
        print(f"Total rows: {total_rows}")
        print(f"Total columns: {len(df.columns)}")
        print("\n" + "="*80)
        
        print(f"FIRST {num_rows} ROWS:")
        print("-" * 40)
        with pd.option_context('display.max_columns', None, 'display.width', None):
            print(df.head(num_rows))
        
        print("\n" + "="*80)
        print(f"LAST {num_rows} ROWS:")
        print("-" * 40)
        with pd.option_context('display.max_columns', None, 'display.width', None):
            print(df.tail(num_rows))
        
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    file_path = input("Enter the file path (.parquet or .csv): ").strip()
    
    num_rows = input("Number of rows to show (default 10): ").strip()
    if not num_rows:
        num_rows = 10
    else:
        try:
            num_rows = int(num_rows)
        except ValueError:
            print("Invalid number, using default 10 rows")
            num_rows = 10
    
    preview_file(file_path, num_rows)

if __name__ == "__main__":
    main()