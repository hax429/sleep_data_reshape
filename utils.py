"""
Utility Functions for Parquet to CSV Transform Process

This module contains utility functions for data processing, file operations,
and data transformations used throughout the transform pipeline.

Functions are organized by category:
- File and path operations
- Data type extraction and formatting
- Date/time processing
- Metadata operations
- Validation functions

Author: Transform Script
Date: 2024
"""

import os
import re
import pandas as pd
import pytz
from datetime import datetime
from logger import log_warning, log_info


def extract_participant_info(file_path):
    """
    Extract participant ID and data type from parquet file path.
    
    Args:
        file_path (str): Path to parquet file
        
    Returns:
        tuple: (participant_id, data_type, subject_id_raw, subject_id_padded, subject_id_int)
        
    Examples:
        >>> extract_participant_info("SP114_e4_ibi_left.parquet")
        ('SP114', 'ibi_left', '114', '114', 114)
    """
    filename = os.path.basename(file_path)
    
    # Extract raw subject ID (remove 'SP' prefix)
    subject_id_raw = filename.split('_')[0].lstrip('SP')
    subject_id_padded = subject_id_raw.zfill(3)
    subject_id_int = int(subject_id_raw)
    
    # Build participant ID
    participant_id = f"SP{subject_id_raw}"
    
    # Extract data type (e.g., "ibi_left", "acc_right")
    parts = filename.split('_')
    if len(parts) >= 4:
        data_type = f"{parts[2]}_{parts[3].replace('.parquet', '')}"
    else:
        data_type = "unknown"
    
    return participant_id, data_type, subject_id_raw, subject_id_padded, subject_id_int


def extract_participant_from_path(directory_path):
    """
    Extract participant ID from directory path using regex.
    
    Args:
        directory_path (str): Directory path that may contain participant ID
        
    Returns:
        str: Participant ID (e.g., 'SP114') or 'UNKNOWN' if not found
        
    Examples:
        >>> extract_participant_from_path("/data/SP114/files/")
        'SP114'
    """
    try:
        match = re.search(r'SP(\d+)', directory_path)
        return f"SP{match.group(1)}" if match else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def get_device_position(file_path):
    """
    Determine device position from file path.
    
    Args:
        file_path (str): Path to the parquet file
        
    Returns:
        int: 0 for left, 1 for right, '' for unknown
    """
    filename = os.path.basename(file_path).lower()
    if 'left' in filename:
        return 0
    elif 'right' in filename:
        return 1
    else:
        return ''


def format_device_timestamp(timestamp_series):
    """
    Format timestamp series to device timestamp format.
    
    Args:
        timestamp_series (pd.Series): Series of timestamps
        
    Returns:
        pd.Series: Formatted timestamps in 'YYYY-MM-DD HH:MM:SS.ffffff' format
    """
    if not timestamp_series.empty:
        return timestamp_series.dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    else:
        return pd.Series([''] * len(timestamp_series))


def get_timezone_abbreviation(timezone_series):
    """
    Get timezone abbreviation from timezone series.
    
    Args:
        timezone_series (pd.Series): Series containing timezone strings
        
    Returns:
        str: Timezone abbreviation (e.g., 'EST', 'PST') or empty string
    """
    if timezone_series.empty:
        return ''
    
    # Get the first timezone value (assuming consistent across dataset)
    timezone_value = timezone_series.iloc[0] if not timezone_series.empty else ''
    
    if not timezone_value:
        return ''
    
    try:
        # Create a timezone object from the timezone string
        tz = pytz.timezone(timezone_value)
        
        # Get the current time in that timezone to determine abbreviation
        now = datetime.now(tz)
        
        # Return the timezone abbreviation (e.g., EST, PST)
        return now.strftime('%Z')
    except Exception:
        # If timezone conversion fails, return original value
        return timezone_value


def get_utc_offset(timezone_series):
    """
    Calculate UTC offset from timezone series.
    
    Args:
        timezone_series (pd.Series): Series containing timezone strings
        
    Returns:
        int: UTC offset in hours (-11 to 12) or empty string if invalid
    """
    if timezone_series.empty:
        return ''
    
    # Get the first timezone value (assuming consistent across dataset)
    timezone_value = timezone_series.iloc[0] if not timezone_series.empty else ''
    
    if not timezone_value:
        return ''
    
    try:
        # Create a timezone object from the timezone string
        tz = pytz.timezone(timezone_value)
        
        # Get the current time in that timezone to determine offset
        now = datetime.now(tz)
        
        # Calculate UTC offset in hours
        utc_offset_seconds = now.utcoffset().total_seconds()
        utc_offset_hours = int(utc_offset_seconds / 3600)
        
        # Ensure offset is within expected range (-11 to 12)
        if -11 <= utc_offset_hours <= 12:
            return utc_offset_hours
        else:
            return ''
    except Exception:
        # If timezone conversion fails, return empty string
        return ''


def calculate_day_codes(timestamp_series):
    """
    Calculate day codes for timestamp series.
    
    Day code classification:
    - 0 = Weekday (Sunday-Thursday)
    - 1 = Weekend day (Friday-Saturday)
    
    Args:
        timestamp_series (pd.Series): Series of timestamps
        
    Returns:
        pd.Series: Series of day codes (0 or 1)
    """
    if timestamp_series.empty:
        return pd.Series([''] * len(timestamp_series))
    
    try:
        day_codes = []
        for timestamp in timestamp_series:
            if timestamp is None or pd.isna(timestamp):
                day_codes.append('')
            else:
                # Get day of week (0=Monday, 1=Tuesday, ..., 6=Sunday)
                daycode = timestamp.weekday()
                
                # Convert to classification:
                # 0 = Weekday (Sunday-Thursday) = weekday() 6,0,1,2,3
                # 1 = Weekend day (Friday-Saturday) = weekday() 4,5
                if daycode in [4, 5]:  # Friday, Saturday
                    day_codes.append(1)
                else:  # Sunday, Monday, Tuesday, Wednesday, Thursday
                    day_codes.append(0)
        
        return pd.Series(day_codes)
    except Exception:
        return pd.Series([''] * len(timestamp_series))


def calculate_actual_device_days(df):
    """
    Calculate actual device days based on device worn percentage.
    
    A day is considered valid if the device was worn more than 90% of the day.
    
    Args:
        df (pd.DataFrame): DataFrame with 'device_worn' and 'timestamp_local' columns
        
    Returns:
        int: Number of days device was worn > 90% of the time
    """
    if df.empty or 'device_worn' not in df.columns or 'timestamp_local' not in df.columns:
        return 0
    
    # Create a copy to avoid modifying original dataframe
    df_copy = df.copy()
    
    # Convert timestamp to date for grouping by day
    df_copy['date'] = df_copy['timestamp_local'].dt.date
    
    worn_days = []
    
    # Group by date and calculate device worn percentage for each day
    for date, day_data in df_copy.groupby('date'):
        total_records = len(day_data)
        worn_records = (day_data['device_worn'] == 1).sum()
        
        # Calculate percentage worn for this day
        worn_percentage = worn_records / total_records if total_records > 0 else 0
        
        # If device worn more than 90% of the day, add to worn days list
        if worn_percentage > 0.9:
            worn_days.append(date)
    
    log_info(f"Actual Device Days: {worn_days}")
    return len(worn_days)


def load_metadata(metadata_path, subject_id_int):
    """
    Load metadata for a specific subject from CSV file.
    
    Args:
        metadata_path (str): Path to metadata.csv file
        subject_id_int (int): Subject ID to look up
        
    Returns:
        tuple: (subjectkey, sex) or ('', '') if not found
    """
    if not os.path.exists(metadata_path):
        log_warning("metadata.csv not found")
        return '', ''
    
    try:
        metadata_df = pd.read_csv(metadata_path)
        subject_row = metadata_df[metadata_df['src_subject_id'] == subject_id_int]
        
        if not subject_row.empty:
            subjectkey = subject_row.iloc[0]['subjectkey']
            sex = subject_row.iloc[0]['sex']
            log_info(f"Found metadata: subjectkey={subjectkey}, sex={sex}")
            return subjectkey, sex
        else:
            log_warning(f"No metadata found for subject ID {subject_id_int}")
            return '', ''
            
    except Exception as e:
        log_warning(f"Could not read metadata.csv: {e}")
        return '', ''


def validate_directory(directory_path):
    """
    Validate that directory exists and is accessible.
    
    Args:
        directory_path (str): Path to directory
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If path is not a directory
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a directory.")


def find_required_parquet_files(directory_path):
    """
    Find and validate all required parquet files in directory.
    
    Expected files:
    - SP{id}_e4_acc_left.parquet
    - SP{id}_e4_eda_left.parquet  
    - SP{id}_e4_ibi_left.parquet
    - SP{id}_e4_temp_left.parquet
    
    Args:
        directory_path (str): Directory containing parquet files
        
    Returns:
        tuple: (found_files_dict, subject_id_raw)
        
    Raises:
        FileNotFoundError: If required files are missing
    """
    validate_directory(directory_path)
    
    # Find all parquet files in the directory
    all_files = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
    
    if not all_files:
        raise FileNotFoundError(f"No parquet files found in directory '{directory_path}'.")
    
    # Extract subject ID from the first file
    sample_file = all_files[0]
    subject_id_raw = sample_file.split('_')[0].lstrip('SP')
    
    # Define the four required file patterns
    required_files = [
        f"SP{subject_id_raw}_e4_acc_left.parquet",
        f"SP{subject_id_raw}_e4_eda_left.parquet",
        f"SP{subject_id_raw}_e4_ibi_left.parquet",
        f"SP{subject_id_raw}_e4_temp_left.parquet"
    ]
    
    # Check if all required files exist
    missing_files = []
    found_files = {}
    
    for required_file in required_files:
        file_path = os.path.join(directory_path, required_file)
        if os.path.exists(file_path):
            found_files[required_file] = file_path
        else:
            missing_files.append(required_file)
    
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    log_info(f"Found all required files for subject SP{subject_id_raw}:")
    for file_name in required_files:
        log_info(f"  - {file_name}")
    
    return found_files, subject_id_raw


def create_output_directory(output_folder):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_folder (str): Path to output directory
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        log_info(f"Created output folder: {output_folder}")


def get_file_size_gb(file_path):
    """
    Get file size in gigabytes.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        float: File size in GB
    """
    return os.path.getsize(file_path) / (1024**3)


def get_file_size_mb(file_path):
    """
    Get file size in megabytes.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        float: File size in MB
    """
    return os.path.getsize(file_path) / (1024**2)


def generate_log_filename(directory_path):
    """
    Generate timestamped log filename based on directory path.
    
    Args:
        directory_path (str): Base directory path
        
    Returns:
        str: Full path to log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.path.dirname(directory_path), f"transform_log_{timestamp}.txt")