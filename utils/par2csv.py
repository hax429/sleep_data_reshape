#!/usr/bin/env python3
import pandas as pd
import sys
from pathlib import Path


def convert_parquet_to_csv(parquet_path, csv_path=None):
    """
    Convert a parquet file to CSV format.

    Args:
        parquet_path (str): Path to the input parquet file
        csv_path (str): Path for the output CSV file (optional)
    """
    input_path = Path(parquet_path)

    if not input_path.exists():
        print(f"Error: File '{parquet_path}' does not exist.")
        return False

    if not input_path.suffix.lower() == '.parquet':
        print(f"Warning: '{parquet_path}' does not have a .parquet extension.")

    # Generate output path if not provided
    if csv_path is None:
        csv_path = input_path.with_suffix('.csv')
    else:
        csv_path = Path(csv_path)

    try:
        # Read parquet file with different engines if needed
        print(f"Reading parquet file: {input_path.name}")

        # Try pandas first
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e1:
            print(f"Pandas failed: {e1}")
            print("Trying with pyarrow engine...")
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(parquet_path)
                df = table.to_pandas()
            except Exception as e2:
                print(f"PyArrow failed: {e2}")
                print("Trying to read without compression...")
                df = pd.read_parquet(parquet_path, engine='fastparquet')

        print(f"Original file: {df.shape[0]} rows × {df.shape[1]} columns")

        # Limit to first 200 rows
        df = df.head(2000)
        print(f"Converting first 200 rows: {df.shape[0]} rows × {df.shape[1]} columns")

        # Write to CSV
        print(f"Converting to CSV: {csv_path.name}")
        df.to_csv(csv_path, index=False)

        # Display file sizes
        parquet_size = input_path.stat().st_size / (1024 * 1024)
        csv_size = csv_path.stat().st_size / (1024 * 1024)

        print(f"Conversion complete!")
        print(f"Original parquet: {parquet_size:.2f} MB")
        print(f"Output CSV: {csv_size:.2f} MB")
        print(f"Size ratio: {csv_size / parquet_size:.1f}x larger")

        return True

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python transform.py <parquet_file> [output_csv_file]")
        print("Example: python transform.py data/sample.parquet")
        print("Example: python transform.py data/sample.parquet output/sample.csv")
        sys.exit(1)

    parquet_file = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None

    convert_parquet_to_csv(parquet_file, csv_file)


if __name__ == "__main__":
    main()