#!/usr/bin/env python3
import pandas as pd
import pyarrow.parquet as pq
import sys
from pathlib import Path

def validate_parquet_file(file_path):
    """
    Validate a parquet file and provide detailed diagnostic information.
    
    Args:
        file_path (str): Path to the parquet file to validate
    """
    path = Path(file_path)
    
    print(f"Validating parquet file: {path.name}")
    print("=" * 60)
    
    # Check if file exists
    if not path.exists():
        print(f"❌ File does not exist: {file_path}")
        return False
    
    # Check file extension
    if not path.suffix.lower() == '.parquet':
        print(f"⚠️  Warning: File does not have .parquet extension")
    
    # Check file size
    file_size = path.stat().st_size
    print(f"📁 File size: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
    
    if file_size == 0:
        print("❌ File is empty")
        return False
    
    validation_results = {
        'pandas': False,
        'pyarrow': False,
        'fastparquet': False
    }
    
    # Test 1: Try reading with pandas (default engine)
    print("\n🔍 Testing with pandas (default engine)...")
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Success: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   Columns: {list(df.columns)}")
        validation_results['pandas'] = True
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
    
    # Test 2: Try reading with pyarrow directly
    print("\n🔍 Testing with pyarrow engine...")
    try:
        table = pq.read_table(file_path)
        df_arrow = table.to_pandas()
        print(f"✅ Success: {df_arrow.shape[0]} rows × {df_arrow.shape[1]} columns")
        validation_results['pyarrow'] = True
        
        # Get additional metadata
        parquet_file = pq.ParquetFile(file_path)
        print(f"   Schema: {parquet_file.schema_arrow}")
        print(f"   Metadata: {parquet_file.metadata.num_row_groups} row groups")
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
    
    # Test 3: Try reading with fastparquet
    print("\n🔍 Testing with fastparquet engine...")
    try:
        df_fast = pd.read_parquet(file_path, engine='fastparquet')
        print(f"✅ Success: {df_fast.shape[0]} rows × {df_fast.shape[1]} columns")
        validation_results['fastparquet'] = True
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
    
    # Test 4: Check for corruption by reading metadata only
    print("\n🔍 Testing metadata access...")
    try:
        pf = pq.ParquetFile(file_path)
        metadata = pf.metadata
        print(f"✅ Metadata accessible:")
        print(f"   Total rows: {metadata.num_rows:,}")
        print(f"   Total columns: {metadata.num_columns}")
        print(f"   Row groups: {metadata.num_row_groups}")
        print(f"   Created by: {metadata.created_by}")
    except Exception as e:
        print(f"❌ Metadata access failed: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    successful_engines = [engine for engine, success in validation_results.items() if success]
    
    if successful_engines:
        print(f"✅ File is valid and readable with: {', '.join(successful_engines)}")
        return True
    else:
        print("❌ File appears to be corrupted or unreadable with all tested engines")
        print("   Possible issues:")
        print("   - Corrupted compression (Snappy/GZIP/LZ4)")
        print("   - Incomplete file transfer")
        print("   - Wrong file format")
        print("   - Unsupported parquet version")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_parquet.py <parquet_file_path>")
        print("Example: python validate_parquet.py data/sample.parquet")
        sys.exit(1)
    
    file_path = sys.argv[1]
    is_valid = validate_parquet_file(file_path)
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()