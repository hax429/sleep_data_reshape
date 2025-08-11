import pandas as pd
import os
import sys
from pathlib import Path

def convert_excel_to_csv(excel_file_path, output_folder='output'):
    """
    Convert Excel file (xlsx or xls) to CSV format
    
    Args:
        excel_file_path (str): Path to the Excel file
        output_folder (str): Output directory for CSV file
    
    Returns:
        str: Path to the created CSV file
    """
    if not os.path.exists(excel_file_path):
        raise FileNotFoundError(f"File '{excel_file_path}' does not exist.")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Get file extension to determine Excel format
    file_extension = Path(excel_file_path).suffix.lower()
    if file_extension not in ['.xlsx', '.xls']:
        raise ValueError(f"Unsupported file format: {file_extension}. Only .xlsx and .xls are supported.")
    
    print(f"Converting Excel file: {os.path.basename(excel_file_path)}")
    
    # Read Excel file
    try:
        if file_extension == '.xlsx':
            df = pd.read_excel(excel_file_path, engine='openpyxl')
        else:  # .xls
            df = pd.read_excel(excel_file_path, engine='xlrd')
        
        print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        
    except Exception as e:
        raise Exception(f"Error reading Excel file: {e}")
    
    # Generate output file path
    base_filename = Path(excel_file_path).stem + '_converted.csv'
    output_file = os.path.join(output_folder, base_filename)
    
    # Write CSV file
    print(f"Writing CSV to: {output_file}")
    df.to_csv(output_file, index=False)
    
    output_size = os.path.getsize(output_file) / (1024**2)  # MB
    print(f"CSV file created successfully: {output_file} ({output_size:.2f} MB)")
    
    return output_file

def main():
    print("Excel to CSV Converter")
    print("=" * 30)
    
    excel_file_path = input("Enter the Excel file path (.xlsx or .xls): ").strip()
    
    output_folder = input("Enter output folder (default: 'output'): ").strip()
    if not output_folder:
        output_folder = 'output'
    
    try:
        print("\n" + "=" * 40)
        print("Starting conversion...")
        
        output_file = convert_excel_to_csv(excel_file_path, output_folder)
        
        print("\n" + "=" * 40)
        print("Conversion completed successfully!")
        print(f"Output file: {output_file}")
        
    except Exception as e:
        print(f"\nError converting file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()