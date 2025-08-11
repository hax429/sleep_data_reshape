import pandas as pd
import sys
import os

def log_info(message):
    print(message)

def display_rows(csv_file_path, start_row, num_rows):
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File '{csv_file_path}' does not exist.")
    
    log_info(f"Reading CSV file: {csv_file_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    log_info(f"Total rows in file: {len(df)}")
    log_info(f"Total columns in file: {len(df.columns)}")
    
    # Validate row range
    if start_row < 0 or start_row >= len(df):
        log_info(f"Error: Start row {start_row} is out of range. Valid range: 0 to {len(df)-1}")
        return
    
    end_row = min(start_row + num_rows, len(df))
    actual_rows_displayed = end_row - start_row
    
    log_info(f"Displaying rows {start_row} to {end_row-1} ({actual_rows_displayed} rows)")
    log_info("=" * 80)
    
    # Set pandas display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    # Extract and display the requested rows
    selected_rows = df.iloc[start_row:end_row]
    
    # Display each row with its index
    for idx, (row_idx, row_data) in enumerate(selected_rows.iterrows()):
        log_info(f"\nRow {row_idx} (Display #{idx+1}):")
        log_info("-" * 40)
        
        # Display each column and its value
        for col_name, value in row_data.items():
            log_info(f"{col_name}: {value}")
        
        log_info("-" * 40)

def main():
    log_info("CSV Row Viewer")
    log_info("=" * 40)
    
    # Get CSV file path
    csv_file_path = input("Enter the CSV file path: ").strip()
    
    # Get starting row number
    start_row_input = input("Enter the starting row number (0-based index): ").strip()
    try:
        start_row = int(start_row_input)
    except ValueError:
        log_info("Error: Please enter a valid integer for starting row number.")
        return
    
    # Get number of rows to display
    num_rows_input = input("Enter the number of rows to display: ").strip()
    try:
        num_rows = int(num_rows_input)
        if num_rows <= 0:
            log_info("Error: Number of rows must be greater than 0.")
            return
    except ValueError:
        log_info("Error: Please enter a valid integer for number of rows.")
        return
    
    try:
        log_info("\n" + "=" * 50)
        display_rows(csv_file_path, start_row, num_rows)
        
    except Exception as e:
        log_info(f"\nError processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()