#!/usr/bin/env python3
"""
Parquet to CSV Transform Pipeline
Multi-file Directory Mode with Enhanced Logging and Parallel Processing

Consolidates E4 wearable sensor data transformation from parquet to CSV format
with comprehensive configuration, logging, and processing capabilities.

Author: Transform Script
Date: 2024
"""

import pandas as pd
import os
import sys
import logging
import re
import pytz
import platform
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

# =============================================================================
# CONFIGURATION SECTION - All user inputs and defaults
# =============================================================================

# Default paths
DEFAULT_OUTPUT_FOLDER = os.path.join('data', 'output')
DEFAULT_METADATA_PATH = os.path.join('data', 'metadata.csv')
DEFAULT_LOGS_FOLDER = 'logs'
DEFAULT_PARTICIPANTS_FOLDER = os.path.join('..', 'final_processed_study_parquet')

# Default processing settings
DEFAULT_PROCESSING_MODE = 'single'  # Options: 'single', 'cpu', 'gpu'
DEFAULT_ROW_LIMITS = {'acc': -1, 'eda': -1, 'ibi': -1, 'temp': -1}  # -1 means all rows

# Processing modes configuration
PROCESSING_MODES = {
    'cpu': 'CPU parallel processing',
    'gpu': 'GPU/CUDA parallel processing', 
    'single': 'Single-threaded processing'
}

# User prompts and defaults
PROMPTS = {
    'processing_mode': f"Processing mode (cpu/gpu/single, default: {DEFAULT_PROCESSING_MODE}): ",
    'output_folder': f"Enter output folder (default: '{DEFAULT_OUTPUT_FOLDER}'): ",
    'metadata_path': f"Enter metadata CSV file path (default: '{DEFAULT_METADATA_PATH}'): ",
    'row_limits': {
        'acc': "ACC rows (default: all): ",
        'eda': "EDA rows (default: all): ",
        'ibi': "IBI rows (default: all): ",
        'temp': "TEMP rows (default: all): "
    }
}

# File patterns and requirements
REQUIRED_FILE_PATTERNS = ['acc', 'eda', 'ibi', 'temp']
FILE_PATTERN_TEMPLATE = "SP{subject_id}_e4_{file_type}_{side}.parquet"  # side can be 'left' or 'right'

# =============================================================================
# CUDA CONFIGURATION
# =============================================================================

# Optional imports for CUDA support
try:
    import cudf
    import cupy as cp
    # CUDA is not supported on macOS
    if platform.system() == 'Darwin':
        CUDA_AVAILABLE = False
        CUDA_REASON = "CUDA is not supported on macOS (requires NVIDIA GPU)"
    else:
        CUDA_AVAILABLE = True
        CUDA_REASON = None
except ImportError:
    cudf = None
    cp = None
    CUDA_AVAILABLE = False
    CUDA_REASON = "CUDA libraries not installed (pip install cudf-cu12 cupy-cuda12x)"

# =============================================================================
# LOGGING SYSTEM
# =============================================================================

class ProcessLogger:
    """Enhanced logging system for batch processing with participant tracking."""
    
    def __init__(self, log_file=None, participant_id=None):
        self.participant_id = participant_id
        self.log_file = log_file
        self.logger = logging.getLogger(f'transform_logger_{os.getpid()}')
        
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers = []
        self.logger.setLevel(logging.DEBUG)
        
        # Setup formatters and handlers
        self._setup_formatters()
        self._setup_console_handler()
        if log_file:
            self._setup_file_handler()
    
    def _setup_formatters(self):
        """Setup log formatters for console and file output."""
        self.console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(participant_id)s | %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _setup_console_handler(self):
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Setup file logging handler."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created logs directory: {log_dir}")
        
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)
    
    def _log(self, level, message):
        """Internal logging method with participant ID context."""
        extra = {'participant_id': self.participant_id or 'SYSTEM'}
        self.logger.log(level, message, extra=extra)
    
    def info(self, message):
        self._log(logging.INFO, message)
    
    def warning(self, message):
        self._log(logging.WARNING, message)
    
    def error(self, message):
        self._log(logging.ERROR, message)
    
    def debug(self, message):
        self._log(logging.DEBUG, message)
    
    def set_participant_id(self, participant_id):
        self.participant_id = participant_id
    
    def section_header(self, title):
        separator = "=" * 60
        self.info(separator)
        self.info(f" {title.upper()}")
        self.info(separator)
    
    def close(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


class LoggerManager:
    """Manager class for global logger instance."""
    _instance = None
    
    @classmethod
    def setup(cls, log_file=None, participant_id=None):
        cls._instance = ProcessLogger(log_file, participant_id)
        return cls._instance
    
    @classmethod
    def get_logger(cls):
        return cls._instance
    
    @classmethod
    def close(cls):
        if cls._instance:
            cls._instance.close()
            cls._instance = None


# Convenience functions for logging
def setup_logging(log_file=None, participant_id=None):
    return LoggerManager.setup(log_file, participant_id)

def log_info(message):
    logger = LoggerManager.get_logger()
    if logger:
        logger.info(message)
    else:
        print(message)

def log_warning(message):
    logger = LoggerManager.get_logger()
    if logger:
        logger.warning(message)
    else:
        print(f"WARNING: {message}")

def log_error(message):
    logger = LoggerManager.get_logger()
    if logger:
        logger.error(message)
    else:
        print(f"ERROR: {message}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_participant_info(file_path):
    """Extract participant ID and data type from parquet file path."""
    filename = os.path.basename(file_path)
    
    subject_id_raw = filename.split('_')[0].lstrip('SP')
    subject_id_padded = subject_id_raw.zfill(3)
    subject_id_int = int(subject_id_raw)
    
    participant_id = f"SP{subject_id_raw}"
    
    parts = filename.split('_')
    if len(parts) >= 4:
        data_type = f"{parts[2]}_{parts[3].replace('.parquet', '')}"
    else:
        data_type = "unknown"
    
    return participant_id, data_type, subject_id_raw, subject_id_padded, subject_id_int

def extract_participant_from_path(directory_path):
    """Extract participant ID from directory path using regex."""
    try:
        match = re.search(r'SP(\d+)', directory_path)
        return f"SP{match.group(1)}" if match else "UNKNOWN"
    except Exception:
        return "UNKNOWN"

def get_device_position(file_path):
    """Determine device position from file path."""
    filename = os.path.basename(file_path).lower()
    if 'left' in filename:
        return 0
    elif 'right' in filename:
        return 1
    else:
        return ''

def format_device_timestamp(timestamp_series):
    """Format timestamp series to device timestamp format."""
    if not timestamp_series.empty:
        return timestamp_series.dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    else:
        return pd.Series([''] * len(timestamp_series))

def get_timezone_abbreviation(timezone_series):
    """Get timezone abbreviation from timezone series."""
    if timezone_series.empty:
        return ''
    
    timezone_value = timezone_series.iloc[0] if not timezone_series.empty else ''
    if not timezone_value:
        return ''
    
    try:
        tz = pytz.timezone(timezone_value)
        now = datetime.now(tz)
        return now.strftime('%Z')
    except Exception:
        return timezone_value

def get_utc_offset(timezone_series):
    """Calculate UTC offset from timezone series."""
    if timezone_series.empty:
        return ''
    
    timezone_value = timezone_series.iloc[0] if not timezone_series.empty else ''
    if not timezone_value:
        return ''
    
    try:
        tz = pytz.timezone(timezone_value)
        now = datetime.now(tz)
        utc_offset_seconds = now.utcoffset().total_seconds()
        utc_offset_hours = int(utc_offset_seconds / 3600)
        
        if -11 <= utc_offset_hours <= 12:
            return utc_offset_hours
        else:
            return ''
    except Exception:
        return ''

# Global birth date cache to avoid recalculating for the same participant
_birth_date_cache = {}

def clear_birth_date_cache():
    """Clear the birth date cache."""
    global _birth_date_cache
    _birth_date_cache.clear()
    log_info("Birth date cache cleared")

def get_cache_info():
    """Get information about the current cache state."""
    return f"Birth date cache contains {len(_birth_date_cache)} entries: {list(_birth_date_cache.keys())}"

def discover_participants(base_directory=None):
    """Discover all participant folders (SP*) in the participants directory."""
    if base_directory is None:
        base_directory = DEFAULT_PARTICIPANTS_FOLDER
    if not os.path.exists(base_directory):
        return []
    
    participants = []
    try:
        for item in os.listdir(base_directory):
            full_path = os.path.join(base_directory, item)
            if os.path.isdir(full_path) and item.startswith('SP') and len(item) >= 3:
                # Extract numeric part for sorting
                try:
                    participant_num = int(item[2:])  # Extract number after "SP"
                    participants.append((participant_num, item, full_path))
                except ValueError:
                    # Skip folders that don't have valid numeric suffix
                    continue
        
        # Sort by participant number
        participants.sort(key=lambda x: x[0])
        return [(folder, path) for _, folder, path in participants]
        
    except Exception as e:
        log_warning(f"Error discovering participants: {e}")
        return []

def display_participants(participants):
    """Display available participants in a formatted way."""
    if not participants:
        print("No participant folders found.")
        return
    
    print(f"\nFound {len(participants)} participants:")
    print("=" * 60)
    
    # Display in columns for better readability
    cols = 4
    for i in range(0, len(participants), cols):
        row_participants = participants[i:i+cols]
        row_text = ""
        for j, (folder, _) in enumerate(row_participants):
            row_text += f"{i+j+1:2d}. {folder:<8}"
        print(row_text)
    
    print("=" * 60)

def get_participant_selection(participants):
    """Get user selection of participants to process."""
    if not participants:
        return []
    
    display_participants(participants)
    print("\nSelection options:")
    print("  -1 or 'all': Process ALL participants")
    print("  Single number: Process one participant (e.g., 1)")
    print("  Multiple numbers: Process multiple participants (e.g., 1,3,5)")
    print("  Range: Process range of participants (e.g., 1-5)")
    print("  Mixed: Combine options (e.g., 1,3-5,7)")
    
    while True:
        try:
            selection = input(f"\nSelect participants to process (1-{len(participants)}, -1 for all): ").strip()
            
            if selection.lower() in ['-1', 'all']:
                return participants
            
            if not selection:
                print("Error: Please enter a selection.")
                continue
            
            selected_indices = set()
            
            # Parse different selection formats
            parts = selection.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part and not part.startswith('-'):
                    # Range selection (e.g., 1-5)
                    try:
                        start, end = part.split('-', 1)
                        start_idx = int(start.strip()) - 1
                        end_idx = int(end.strip()) - 1
                        if 0 <= start_idx < len(participants) and 0 <= end_idx < len(participants):
                            for i in range(start_idx, end_idx + 1):
                                selected_indices.add(i)
                        else:
                            print(f"Error: Range {part} is out of bounds (1-{len(participants)})")
                            raise ValueError("Invalid range")
                    except ValueError:
                        print(f"Error: Invalid range format '{part}'. Use format like '1-5'")
                        break
                else:
                    # Single number
                    try:
                        idx = int(part) - 1
                        if 0 <= idx < len(participants):
                            selected_indices.add(idx)
                        else:
                            print(f"Error: {int(part)} is out of bounds (1-{len(participants)})")
                            raise ValueError("Out of bounds")
                    except ValueError:
                        print(f"Error: Invalid number '{part}'")
                        break
            else:
                # If we didn't break out of the loop, selection is valid
                if selected_indices:
                    selected_participants = [participants[i] for i in sorted(selected_indices)]
                    
                    # Confirm selection
                    print(f"\nSelected {len(selected_participants)} participants:")
                    for folder, _ in selected_participants:
                        print(f"  - {folder}")
                    
                    confirm = input("\nProceed with this selection? (y/n, default: y): ").strip().lower()
                    if confirm in ['', 'y', 'yes']:
                        return selected_participants
                    else:
                        print("Selection cancelled. Please choose again.")
                        continue
                else:
                    print("Error: No valid participants selected.")
                    continue
                    
        except KeyboardInterrupt:
            print("\n\nSelection cancelled by user.")
            return []
        except Exception as e:
            print(f"Error: {e}. Please try again.")
            continue

def load_visit_data_and_calculate_birth_date(visit_csv_path, subject_id_int):
    """Load visit data and calculate possible date of birth for a subject with verification."""
    # Check cache first
    if subject_id_int in _birth_date_cache:
        cached_birth_date = _birth_date_cache[subject_id_int]
        log_info(f"Using cached birth date for subject {subject_id_int}: {cached_birth_date.strftime('%Y-%m-%d')}")
        return cached_birth_date, []
    
    if not os.path.exists(visit_csv_path):
        log_warning(f"visit_converted.csv not found at {visit_csv_path}")
        return None, []
    
    try:
        visit_df = pd.read_csv(visit_csv_path)
        subject_rows = visit_df[visit_df['src_subject_id'] == f"{subject_id_int:03d}--1"]
        
        if subject_rows.empty:
            log_warning(f"No visit data found for subject ID {subject_id_int} (padded: {subject_id_int:03d}--1)")
            return None, []
        
        # Convert interview_date to datetime
        subject_rows = subject_rows.copy()
        subject_rows['interview_date'] = pd.to_datetime(subject_rows['interview_date'])
        
        # Sort by interview date
        subject_rows = subject_rows.sort_values('interview_date')
        
        log_info(f"Found {len(subject_rows)} interview entries for subject {subject_id_int}:")
        for _, row in subject_rows.iterrows():
            log_info(f"  Date: {row['interview_date'].strftime('%Y-%m-%d')}, Age: {row['interview_age']} months")
        
        # Calculate possible birth date using multiple data points for better accuracy
        birth_dates = []
        for _, row in subject_rows.iterrows():
            # Calculate birth date from this interview
            interview_date = row['interview_date']
            age_months = row['interview_age']
            
            # Subtract age in months from interview date
            birth_date = interview_date - relativedelta(months=age_months)
            birth_dates.append(birth_date)
            log_info(f"  Calculated birth date from {interview_date.strftime('%Y-%m-%d')}: {birth_date.strftime('%Y-%m-%d')}")
        
        # Use the most common birth date (or average if they're close)
        if birth_dates:
            # Calculate the median birth date for consistency
            sorted_dates = sorted(birth_dates)
            median_date = sorted_dates[len(sorted_dates) // 2]
            
            log_info(f"Initial inferred birth date for subject {subject_id_int}: {median_date.strftime('%Y-%m-%d')}")
            
            # Verify the birth date against all entries
            verified_birth_date = verify_birth_date_against_interviews(median_date, subject_rows)
            
            # Cache the verified birth date
            _birth_date_cache[subject_id_int] = verified_birth_date
            
            return verified_birth_date, subject_rows.to_dict('records')
        
        return None, []
        
    except Exception as e:
        log_warning(f"Could not process visit_converted.csv: {e}")
        return None, []

def verify_birth_date_against_interviews(birth_date, interview_data):
    """Verify that the calculated birth date is consistent with all interview entries."""
    log_info("Verifying birth date against all interview entries:")
    
    max_deviation = 0
    total_deviation = 0
    acceptable_entries = 0
    
    for _, row in interview_data.iterrows():
        interview_date = row['interview_date']
        reported_age = row['interview_age']
        
        # Calculate age from birth date to interview date
        calculated_age_delta = relativedelta(interview_date.date(), birth_date.date())
        calculated_age_months = calculated_age_delta.years * 12 + calculated_age_delta.months
        
        deviation = abs(calculated_age_months - reported_age)
        max_deviation = max(max_deviation, deviation)
        total_deviation += deviation
        
        status = "✓" if deviation <= 1 else "⚠" if deviation <= 2 else "✗"
        log_info(f"  {status} {interview_date.strftime('%Y-%m-%d')}: Reported={reported_age}, Calculated={calculated_age_months}, Deviation={deviation} months")
        
        if deviation <= 1:  # Accept within 1 month tolerance
            acceptable_entries += 1
    
    avg_deviation = total_deviation / len(interview_data) if len(interview_data) > 0 else 0
    acceptance_rate = acceptable_entries / len(interview_data) if len(interview_data) > 0 else 0
    
    log_info(f"Birth date verification summary:")
    log_info(f"  - Maximum deviation: {max_deviation} months")
    log_info(f"  - Average deviation: {avg_deviation:.1f} months") 
    log_info(f"  - Entries within 1 month: {acceptable_entries}/{len(interview_data)} ({acceptance_rate:.1%})")
    
    if acceptance_rate >= 0.8:  # At least 80% of entries should be within 1 month
        log_info(f"✓ Birth date verification PASSED: {birth_date.strftime('%Y-%m-%d')}")
        return birth_date
    else:
        log_warning(f"⚠ Birth date verification shows high deviation. Using calculated date with caution: {birth_date.strftime('%Y-%m-%d')}")
        return birth_date

def get_device_type_from_participant_id(subject_id_int):
    """Determine device type based on participant ID range."""
    if 1 <= subject_id_int < 84:
        return "Empatica E4"
    else:
        # For participants 84 and above, return empty or could be set to other device types
        return ""

def calculate_interview_age_from_timestamp(timestamp_series, birth_date):
    """Calculate interview age in months from timestamp_local and birth date for each row."""
    if birth_date is None or timestamp_series.empty:
        return pd.Series([''] * len(timestamp_series))
    
    try:
        ages = []
        valid_timestamps = 0
        for timestamp in timestamp_series:
            if timestamp is None or pd.isna(timestamp):
                ages.append('')
            else:
                # Calculate age in months for this specific timestamp
                age_delta = relativedelta(timestamp.date(), birth_date.date())
                age_months = age_delta.years * 12 + age_delta.months
                ages.append(int(round(age_months)))
                valid_timestamps += 1
        
        # Log summary of age calculation
        if valid_timestamps > 0:
            valid_ages = [age for age in ages if age != '']
            if valid_ages:
                min_age = min(valid_ages)
                max_age = max(valid_ages)
                log_info(f"Calculated interview_age for {len(valid_ages):,} entries: range {min_age}-{max_age} months")
                
                # Log first and last few entries for verification
                first_timestamps = timestamp_series.dropna().head(3)
                last_timestamps = timestamp_series.dropna().tail(3)
                
                log_info("Sample age calculations:")
                for ts in first_timestamps:
                    age_delta = relativedelta(ts.date(), birth_date.date())
                    age_months = age_delta.years * 12 + age_delta.months
                    log_info(f"  {ts.strftime('%Y-%m-%d')}: {age_months} months")
                
                if len(timestamp_series.dropna()) > 6:  # Only show "last" if we have more than 6 entries
                    log_info("  ...")
                    for ts in last_timestamps:
                        age_delta = relativedelta(ts.date(), birth_date.date())
                        age_months = age_delta.years * 12 + age_delta.months
                        log_info(f"  {ts.strftime('%Y-%m-%d')}: {age_months} months")
        
        return pd.Series(ages)
    except Exception as e:
        log_warning(f"Error calculating interview ages: {e}")
        return pd.Series([''] * len(timestamp_series))

def calculate_day_codes(timestamp_series):
    """Calculate day codes for timestamp series."""
    if timestamp_series.empty:
        return pd.Series([''] * len(timestamp_series))
    
    try:
        day_codes = []
        for timestamp in timestamp_series:
            if timestamp is None or pd.isna(timestamp):
                day_codes.append('')
            else:
                daycode = timestamp.weekday()
                if daycode in [4, 5]:  # Friday, Saturday
                    day_codes.append(1)
                else:  # Sunday, Monday, Tuesday, Wednesday, Thursday
                    day_codes.append(0)
        
        return pd.Series(day_codes)
    except Exception:
        return pd.Series([''] * len(timestamp_series))

def calculate_actual_device_days(df):
    """Calculate actual device days based on device worn percentage."""
    if df.empty or 'device_worn' not in df.columns or 'timestamp_local' not in df.columns:
        return 0
    
    df_copy = df.copy()
    df_copy['date'] = df_copy['timestamp_local'].dt.date
    
    worn_days = []
    
    for date, day_data in df_copy.groupby('date'):
        total_records = len(day_data)
        worn_records = (day_data['device_worn'] == 1).sum()
        
        worn_percentage = worn_records / total_records if total_records > 0 else 0
        
        if worn_percentage > 0.9:
            worn_days.append(date)
    
    log_info(f"Actual Device Days: {worn_days}")
    return len(worn_days)

def load_metadata(metadata_path, subject_id_int):
    """Load metadata for a specific subject from CSV file."""
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
    """Validate that directory exists and is accessible."""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a directory.")

def find_required_parquet_files(directory_path):
    """Find and validate all required parquet files in directory (processes ALL available files)."""
    validate_directory(directory_path)
    
    all_files = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
    
    if not all_files:
        raise FileNotFoundError(f"No parquet files found in directory '{directory_path}'.")
    
    # Extract subject ID from the first file
    sample_file = all_files[0]
    subject_id_raw = sample_file.split('_')[0].lstrip('SP')
    
    # Look for ALL files of each type (both left AND right if they exist)
    file_types = ['acc', 'eda', 'ibi', 'temp']
    found_files = {}
    missing_types = []
    
    for file_type in file_types:
        left_file = f"SP{subject_id_raw}_e4_{file_type}_left.parquet"
        right_file = f"SP{subject_id_raw}_e4_{file_type}_right.parquet"
        
        left_path = os.path.join(directory_path, left_file)
        right_path = os.path.join(directory_path, right_file)
        
        found_any = False
        
        # Add both files if they exist
        if os.path.exists(left_path):
            found_files[left_file] = left_path
            log_info(f"Found {file_type.upper()} file (left side): {left_file}")
            found_any = True
            
        if os.path.exists(right_path):
            found_files[right_file] = right_path
            log_info(f"Found {file_type.upper()} file (right side): {right_file}")
            found_any = True
        
        # Only mark as missing if neither left nor right exists
        if not found_any:
            missing_types.append(file_type)
    
    if missing_types:
        missing_list = []
        for file_type in missing_types:
            missing_list.extend([
                f"SP{subject_id_raw}_e4_{file_type}_left.parquet",
                f"SP{subject_id_raw}_e4_{file_type}_right.parquet"
            ])
        raise FileNotFoundError(
            f"Missing required file types: {missing_types}. "
            f"Could not find any of these files: {missing_list}"
        )
    
    log_info(f"Found {len(found_files)} files for subject SP{subject_id_raw}:")
    for file_name in sorted(found_files.keys()):
        side = 'left' if 'left' in file_name else 'right'
        file_type = file_name.split('_')[2].upper()
        log_info(f"  - {file_type}: {file_name} ({side} side)")
    
    return found_files, subject_id_raw

def create_output_directory(output_folder):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        log_info(f"Created output folder: {output_folder}")

def get_file_size_gb(file_path):
    """Get file size in gigabytes."""
    return os.path.getsize(file_path) / (1024**3)

def get_file_size_mb(file_path):
    """Get file size in megabytes."""
    return os.path.getsize(file_path) / (1024**2)

def generate_log_filename(directory_path):
    """Generate timestamped log filename based on directory path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.path.dirname(directory_path), f"transform_log_{timestamp}.txt")

# =============================================================================
# DATA PROCESSING CLASSES
# =============================================================================

class DataFrameBuilder:
    """Handles construction of output dataframes with proper column mapping."""
    
    def __init__(self, metadata_path=None):
        self.logger = LoggerManager.get_logger()
        self.metadata_path = metadata_path
        self.visit_csv_path = os.path.join('data', 'visit_converted.csv')
    
    def create_output_dataframe(self, df, parquet_file_path):
        """Create output dataframe with proper column mapping and transformations."""
        participant_id, data_type, subject_id_raw, subject_id_padded, subject_id_int = \
            extract_participant_info(parquet_file_path)
        
        if self.logger:
            self.logger.set_participant_id(participant_id)
        
        log_info(f"Creating output dataframe structure for {participant_id} {data_type.upper()}...")
        log_info(f"Processing subject ID: {subject_id_int} (padded: {subject_id_padded})")
        
        if self.metadata_path:
            metadata_path = self.metadata_path
        else:
            metadata_path = DEFAULT_METADATA_PATH
        subject_key_value, sex_value = load_metadata(metadata_path, subject_id_int)
        
        # Load visit data and calculate birth date
        birth_date, visit_entries = load_visit_data_and_calculate_birth_date(self.visit_csv_path, subject_id_int)
        
        output_df = self._build_dataframe(
            df, participant_id, data_type, subject_id_padded,
            subject_key_value, sex_value, parquet_file_path, birth_date
        )
        
        log_info(f"Dataframe structure created successfully for {participant_id} {data_type.upper()} ({len(output_df):,} rows)")
        
        return output_df
    
    def _build_dataframe(self, df, participant_id, data_type, subject_id_padded,
                        subject_key_value, sex_value, parquet_file_path, birth_date=None):
        """Internal method to build the output dataframe structure."""
        subject_key_data = pd.Series([subject_key_value] * len(df))
        sex_data = pd.Series([sex_value] * len(df))
        
        interview_date_data = self._get_interview_date(df)
        
        # Calculate interview age based on birth date and timestamp_local
        interview_age_data = calculate_interview_age_from_timestamp(
            df.get('timestamp_local', pd.Series()), birth_date
        )
        
        # Determine device type based on participant ID
        subject_id_int = int(participant_id)  # Convert padded ID to integer
        device_type = get_device_type_from_participant_id(subject_id_int)
        log_info(f"Device type for participant {subject_id_int}: {device_type}")
        
        timezone_data = get_timezone_abbreviation(df.get('timezone', pd.Series()))
        utc_offset_data = get_utc_offset(df.get('timezone', pd.Series()))
        day_code_data = calculate_day_codes(df.get('timestamp_local', pd.Series()))
        
        length_per = self._get_length_per(df)
        actual_dev_days = calculate_actual_device_days(df)
        
        dc_start_date_data = self._get_dc_start_date(df)
        dc_start_time_data = self._get_dc_start_time(df)
        dc_end_date_data = self._get_dc_end_date(df)
        dc_end_time_data = self._get_dc_end_time(df)
        
        device_position = get_device_position(parquet_file_path)
        device_timestamp_data = format_device_timestamp(df.get('timestamp_utc', pd.Series()))
        
        log_info(f"Building column mappings for {participant_id} {data_type.upper()}...")
        
        output_data = {
            'subjectkey': subject_key_data,
            'src_subject_id': subject_id_padded,
            'interview_date': interview_date_data,
            'interview_age': interview_age_data,
            'sex': sex_data,
            'mt_pilot_vib_device': pd.Series([device_type] * len(df)),
            'length_per': length_per,
            'max_devdays': length_per,
            'actual_devdays': actual_dev_days,
            'tlfb_daynum': df.get('day_of_study', pd.Series([0] * len(df))) + 1,
            'day_code': day_code_data,
            'site_time_zone': timezone_data,
            'utc_offset': utc_offset_data,
            'dc_start_date': dc_start_date_data,
            'dc_start_time': dc_start_time_data,
            'dc_end_date': dc_end_date_data,
            'dc_end_time': dc_end_time_data,
            'watch_device_type': pd.Series([device_type] * len(df)),
            'device_position': device_position,
            'device_timestamp': device_timestamp_data
        }
        
        # Add sensor-specific columns based on data type
        if 'acc' in data_type:
            output_data['accel_x'] = df.get('x_g', pd.Series([''] * len(df)))
            output_data['accel_y'] = df.get('y_g', pd.Series([''] * len(df)))
            output_data['accel_z'] = df.get('z_g', pd.Series([''] * len(df)))
        elif 'eda' in data_type:
            output_data['eda_us'] = df.get('eda', pd.Series([''] * len(df)))
        elif 'ibi' in data_type:
            output_data['ibi'] = df.get('ibi', pd.Series([''] * len(df)))
        elif 'temp' in data_type:
            output_data['skin_temp'] = df.get('temp', pd.Series([''] * len(df)))
        
        return pd.DataFrame(output_data)
    
    def _get_interview_date(self, df):
        timestamp_series = df.get('timestamp_local', pd.Series())
        if timestamp_series.empty:
            return ''
        interview_date = timestamp_series.dt.strftime('%m/%d/%Y')
        return interview_date
    
    def _get_length_per(self, df):
        length_per = df.get('day_of_study', pd.Series())
        length = length_per.max() + 1 if not length_per.empty else 0
        return length
    
    def _get_dc_start_date(self, df):
        timestamp_series = df.get('timestamp_utc', pd.Series())
        if timestamp_series.empty:
            return ''
        first_timestamp = timestamp_series.iloc[0]
        if first_timestamp is None or pd.isna(first_timestamp):
            return ''
        return first_timestamp.strftime('%m/%d/%Y')
    
    def _get_dc_start_time(self, df):
        timestamp_series = df.get('timestamp_utc', pd.Series())
        if timestamp_series.empty:
            return ''
        first_timestamp = timestamp_series.iloc[0]
        if first_timestamp is None or pd.isna(first_timestamp):
            return ''
        return first_timestamp.strftime('%H:%M')
    
    def _get_dc_end_date(self, df):
        timestamp_series = df.get('timestamp_utc', pd.Series())
        if timestamp_series.empty:
            return ''
        last_timestamp = timestamp_series.iloc[-1]
        if last_timestamp is None or pd.isna(last_timestamp):
            return ''
        return last_timestamp.strftime('%m/%d/%Y')
    
    def _get_dc_end_time(self, df):
        timestamp_series = df.get('timestamp_utc', pd.Series())
        if timestamp_series.empty:
            return ''
        last_timestamp = timestamp_series.iloc[-1]
        if last_timestamp is None or pd.isna(last_timestamp):
            return ''
        return last_timestamp.strftime('%H:%M')


class ParallelProcessor:
    """Handles parallel processing operations for both CPU and GPU modes."""
    
    @staticmethod
    def is_cuda_available():
        return CUDA_AVAILABLE
    
    @staticmethod
    def get_cuda_unavailable_reason():
        return CUDA_REASON
    
    @staticmethod
    def process_with_cuda(parquet_file_path, num_rows=-1):
        """Process parquet file using CUDA/GPU acceleration."""
        if not CUDA_AVAILABLE:
            raise ImportError(f"CUDA libraries not available: {CUDA_REASON}")
        
        participant_id, data_type, _, _, _ = extract_participant_info(parquet_file_path)
        log_info(f"Using CUDA/GPU processing for {participant_id} {data_type.upper()}")
        
        if num_rows != -1:
            df_gpu = cudf.read_parquet(parquet_file_path, nrows=num_rows)
        else:
            df_gpu = cudf.read_parquet(parquet_file_path)
        
        log_info(f"Loaded {len(df_gpu):,} rows on GPU")
        
        df = df_gpu.to_pandas()
        return df
    
    @staticmethod
    def process_file_worker(args):
        """Worker function for CPU parallel processing."""
        parquet_file_path, num_rows, output_folder = args
        try:
            processor = ParquetProcessor()
            return processor.process_single_file(parquet_file_path, num_rows, output_folder, 'single')
        except Exception as e:
            participant_id, data_type, _, _, _ = extract_participant_info(parquet_file_path)
            log_error(f"Error in worker process for {participant_id} {data_type.upper()}: {e}")
            raise


class ParquetProcessor:
    """Main processor class that orchestrates the entire parquet to CSV pipeline."""
    
    def __init__(self, metadata_path=None):
        self.dataframe_builder = DataFrameBuilder(metadata_path)
        self.parallel_processor = ParallelProcessor()
        self.metadata_path = metadata_path
    
    def process_single_file(self, parquet_file_path, num_rows=-1, output_folder=DEFAULT_OUTPUT_FOLDER, processing_mode='single'):
        """Process a single parquet file with optional parallel processing."""
        if not os.path.exists(parquet_file_path):
            raise FileNotFoundError(f"File '{parquet_file_path}' does not exist.")
        
        participant_id, data_type, _, _, _ = extract_participant_info(parquet_file_path)
        filename = os.path.basename(parquet_file_path)
        file_size = get_file_size_gb(parquet_file_path)
        
        log_info(f"Processing {participant_id} {data_type.upper()} file: {filename} ({file_size:.2f} GB)")
        log_info(f"Using {PROCESSING_MODES.get(processing_mode, 'single-threaded')} mode...")
        
        df = self._load_data(parquet_file_path, num_rows, processing_mode)
        output_df = self.dataframe_builder.create_output_dataframe(df, parquet_file_path)
        output_file = self._write_csv_output(output_df, parquet_file_path, output_folder, 
                                           participant_id, data_type)
        
        return output_file, len(output_df)
    
    def _load_data(self, parquet_file_path, num_rows, processing_mode):
        """Load data from parquet file based on processing mode."""
        if processing_mode == 'gpu' and CUDA_AVAILABLE:
            return self.parallel_processor.process_with_cuda(parquet_file_path, num_rows)
        
        if processing_mode == 'gpu' and not CUDA_AVAILABLE:
            log_warning(f"CUDA not available: {CUDA_REASON}, falling back to single-threaded processing")
        
        return self._load_with_pandas(parquet_file_path, num_rows)
    
    def _load_with_pandas(self, parquet_file_path, num_rows):
        """Load data using pandas with progress bar."""
        file_size = get_file_size_gb(parquet_file_path)
        
        log_info("Reading parquet file...")
        with tqdm(desc="Loading data", unit="MB") as pbar:
            df = pd.read_parquet(parquet_file_path, engine='pyarrow')
            pbar.update(file_size * 1024)
        
        log_info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        
        if num_rows != -1:
            log_info(f"Limiting to first {num_rows:,} rows...")
            df = df.head(num_rows)
        
        return df
    
    def _write_csv_output(self, output_df, parquet_file_path, output_folder,
                         participant_id, data_type):
        """Write output dataframe to CSV file."""
        base_filename = os.path.splitext(os.path.basename(parquet_file_path))[0] + '_converted.csv'
        output_file = os.path.join(output_folder, base_filename)
        
        log_info(f"Writing {participant_id} {data_type.upper()} CSV to: {output_file}")
        
        output_df.to_csv(output_file, index=False, chunksize=10000)
        
        output_size = get_file_size_mb(output_file)
        log_info(f"{participant_id} {data_type.upper()} CSV file created successfully: {output_file} ({output_size:.2f} MB)")
        
        return output_file
    
    def process_directory(self, directory_path, file_row_limits=None, 
                         output_folder=DEFAULT_OUTPUT_FOLDER, processing_mode='single'):
        """Process all four parquet files in a directory."""
        create_output_directory(output_folder)
        
        found_files, subject_id = find_required_parquet_files(directory_path)
        
        if file_row_limits is None:
            file_row_limits = DEFAULT_ROW_LIMITS.copy()
        
        processing_tasks = self._prepare_processing_tasks(found_files, file_row_limits, output_folder)
        
        if processing_mode == 'cpu' and len(processing_tasks) > 1:
            return self._process_parallel_cpu(processing_tasks)
        else:
            return self._process_sequential(processing_tasks, processing_mode)
    
    def _prepare_processing_tasks(self, found_files, file_row_limits, output_folder):
        """Prepare list of processing tasks for ALL found files."""
        processing_tasks = []
        
        # Process each found file individually (not grouped by file type)
        for file_name, file_path in found_files.items():
            # Extract file type from filename
            parts = file_name.split('_')
            if len(parts) >= 3:
                file_type = parts[2]  # 'acc', 'eda', 'ibi', 'temp'
                num_rows = file_row_limits.get(file_type, -1)
                processing_tasks.append((file_path, num_rows, output_folder, file_type))
        
        return processing_tasks
    
    def _process_parallel_cpu(self, processing_tasks):
        """Process files in parallel using CPU multiprocessing."""
        log_info(f"Using CPU parallel processing with {min(cpu_count(), len(processing_tasks))} processes")
        
        worker_args = [(task[0], task[1], task[2]) for task in processing_tasks]
        
        try:
            with Pool(processes=min(cpu_count(), len(processing_tasks))) as pool:
                results = pool.map(self.parallel_processor.process_file_worker, worker_args)
                
            processed_files = []
            total_processed_rows = 0
            
            for i, (output_file, processed_rows) in enumerate(results):
                file_type = processing_tasks[i][3]
                processed_files.append(output_file)
                total_processed_rows += processed_rows
                log_info(f"Successfully processed {file_type.upper()}: {processed_rows:,} rows")
                
            return processed_files, total_processed_rows
                
        except Exception as e:
            log_error(f"Error in parallel processing: {e}")
            raise
    
    def _process_sequential(self, processing_tasks, processing_mode):
        """Process files sequentially."""
        processed_files = []
        total_processed_rows = 0
        
        for matching_file, num_rows, output_folder, file_type in processing_tasks:
            log_info(f"\n{'='*50}")
            log_info(f"Processing {file_type.upper()} file with {num_rows if num_rows != -1 else 'all'} rows...")
            
            try:
                output_file, processed_rows = self.process_single_file(
                    matching_file, num_rows, output_folder, processing_mode
                )
                processed_files.append(output_file)
                total_processed_rows += processed_rows
                log_info(f"Successfully processed {file_type.upper()}: {processed_rows:,} rows")
            except Exception as e:
                log_error(f"Error processing {file_type.upper()} file: {e}")
                raise
        
        return processed_files, total_processed_rows


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class TransformApplication:
    """Main application class for the parquet to CSV transformation pipeline."""
    
    def __init__(self):
        self.processor = None
        self.logger = None
    
    def run(self):
        """Main application entry point."""
        try:
            self._print_banner()
            config = self._get_user_configuration()
            self._setup_logging(config)
            self._process_files(config)
            self._report_success(config)
            
        except KeyboardInterrupt:
            print("\n\nProcess interrupted by user. Exiting...")
            sys.exit(0)
        except Exception as e:
            self._handle_error(e, config if 'config' in locals() else None)
            sys.exit(1)
        finally:
            self._cleanup()
    
    def _print_banner(self):
        """Print application banner and version information."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Parquet to CSV Transform Pipeline                        ║
║                        Multi-file Directory Mode v2.0                        ║
║                           (Consolidated Version)                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print("Processing E4 wearable sensor data with enhanced logging and parallel processing")
        print("=" * 80)
    
    def _get_user_configuration(self):
        """Collect all user configuration through interactive prompts."""
        config = {}
        
        # First, discover and select participants
        config['selected_participants'] = self._get_participant_selection()
        if not config['selected_participants']:
            print("No participants selected. Exiting...")
            sys.exit(0)
        
        config['processing_mode'] = self._get_processing_mode()
        config['file_row_limits'] = self._get_row_limits()
        config['output_folder'] = self._get_output_folder()
        config['metadata_path'] = self._get_metadata_path()
        
        # Generate log file name based on batch processing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(config['selected_participants']) == 1:
            participant_folder = config['selected_participants'][0][0]
            config['log_file'] = os.path.join(DEFAULT_LOGS_FOLDER, f"transform_log_{participant_folder}_{timestamp}.txt")
            config['participant_id'] = participant_folder
        else:
            config['log_file'] = os.path.join(DEFAULT_LOGS_FOLDER, f"transform_log_batch_{len(config['selected_participants'])}participants_{timestamp}.txt")
            config['participant_id'] = "BATCH"
        
        return config
    
    def _get_participant_selection(self):
        """Get participant selection from user."""
        print(f"\nDiscovering available participants in {DEFAULT_PARTICIPANTS_FOLDER}...")
        participants = discover_participants()
        
        if not participants:
            print(f"No participant folders found in {DEFAULT_PARTICIPANTS_FOLDER}.")
            print("Please ensure the participants directory exists and contains SP* folders.")
            return []
        
        return get_participant_selection(participants)
    
    
    def _get_processing_mode(self):
        """Get and validate processing mode from user."""
        print("\nSelect processing mode:")
        for mode, description in PROCESSING_MODES.items():
            print(f"  '{mode}' - {description}")
        
        if not CUDA_AVAILABLE:
            print(f"\nNote: GPU mode unavailable - {CUDA_REASON}")
        
        while True:
            processing_mode = input(f"\n{PROMPTS['processing_mode']}").strip().lower()
            if not processing_mode:
                processing_mode = DEFAULT_PROCESSING_MODE
            
            if processing_mode not in PROCESSING_MODES:
                print(f"Invalid mode. Please enter one of: {', '.join(PROCESSING_MODES.keys())}")
                continue
                
            if processing_mode == 'gpu' and not CUDA_AVAILABLE:
                print(f"\nWarning: {CUDA_REASON}")
                fallback = input("Fall back to single-threaded processing? (y/n, default: y): ").strip().lower()
                if fallback in ['', 'y', 'yes']:
                    processing_mode = 'single'
                    print("Falling back to single-threaded processing")
                    break
                else:
                    print("Please select a different processing mode.")
                    continue
            
            return processing_mode
    
    def _get_row_limits(self):
        """Get row limits for each file type from user."""
        print("\nEnter number of rows to process for each file type (-1 for all rows):")
        print("Press Enter for default (-1 = all rows)")
        
        file_row_limits = {}
        file_types = ['acc', 'eda', 'ibi', 'temp']
        
        for file_type in file_types:
            while True:
                try:
                    rows_input = input(f"  {PROMPTS['row_limits'][file_type]}").strip()
                    if not rows_input:
                        file_row_limits[file_type] = -1
                        break
                    
                    rows = int(rows_input)
                    if rows < -1 or rows == 0:
                        print("    Error: Please enter -1 (all rows) or a positive integer.")
                        continue
                    
                    file_row_limits[file_type] = rows
                    break
                except ValueError:
                    print("    Error: Please enter a valid integer or press Enter for all rows.")
        
        return file_row_limits
    
    def _get_output_folder(self):
        """Get output folder from user."""
        output_folder = input(f"\n{PROMPTS['output_folder']}").strip()
        if not output_folder:
            output_folder = DEFAULT_OUTPUT_FOLDER
        
        return output_folder
    
    def _get_metadata_path(self):
        """Get metadata CSV file path from user."""
        metadata_path = input(f"\n{PROMPTS['metadata_path']}").strip()
        
        if not metadata_path:
            metadata_path = DEFAULT_METADATA_PATH
        
        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata file not found at '{metadata_path}'")
            print("Processing will continue but metadata fields may be empty")
        
        return metadata_path
    
    def _setup_logging(self, config):
        """Setup the logging system with configuration."""
        self.logger = setup_logging(
            config['log_file'], 
            config['participant_id']
        )
        
        self.logger.section_header("Transform Process Started")
        log_info(f"Selected participants: {len(config['selected_participants'])}")
        for folder, path in config['selected_participants']:
            log_info(f"  - {folder}: {path}")
        log_info(f"Log file: {config['log_file']}")
        log_info(f"Processing mode: {config['processing_mode']}")
        log_info(f"Row limits: {config['file_row_limits']}")
        log_info(f"Output folder: {config['output_folder']}")
        log_info(f"Metadata path: {config['metadata_path']}")
        log_info(get_cache_info())
    
    def _process_files(self, config):
        """Execute the main file processing pipeline."""
        self.logger.section_header("Starting File Processing")
        
        self.processor = ParquetProcessor(config['metadata_path'])
        
        all_output_files = []
        total_processed_rows = 0
        
        # Process each selected participant
        for i, (participant_folder, participant_path) in enumerate(config['selected_participants']):
            participant_num = i + 1
            total_participants = len(config['selected_participants'])
            
            self.logger.section_header(f"Processing Participant {participant_num}/{total_participants}: {participant_folder}")
            log_info(f"Participant directory: {participant_path}")
            
            try:
                output_files, processed_rows = self.processor.process_directory(
                    participant_path,
                    config['file_row_limits'],
                    config['output_folder'],
                    config['processing_mode']
                )
                
                all_output_files.extend(output_files)
                total_processed_rows += processed_rows
                
                log_info(f"✓ Completed {participant_folder}: {len(output_files)} files, {processed_rows:,} rows")
                
            except Exception as e:
                log_error(f"✗ Failed to process {participant_folder}: {e}")
                print(f"ERROR: Failed to process {participant_folder}: {e}")
                continue
        
        config['output_files'] = all_output_files
        config['total_processed_rows'] = total_processed_rows
    
    def _report_success(self, config):
        """Report successful completion with summary."""
        self.logger.section_header("Processing Completed Successfully")
        
        log_info(f"Processed {len(config['output_files'])} files:")
        for output_file in config['output_files']:
            log_info(f"  ✓ {os.path.basename(output_file)}")
        
        log_info(f"Total processed rows: {config['total_processed_rows']:,}")
        log_info(f"Output directory: {os.path.abspath(config['output_folder'])}")
        log_info(f"Log file saved: {config['log_file']}")
        log_info(get_cache_info())
        
        print(f"\n🎉 Processing completed successfully!")
        print(f"📁 Output files: {len(config['output_files'])}")
        print(f"📊 Total rows: {config['total_processed_rows']:,}")
        print(f"📝 Log file: {config['log_file']}")
    
    def _handle_error(self, error, config):
        """Handle application errors with proper logging."""
        error_msg = f"Application error: {error}"
        
        if self.logger:
            log_error(error_msg)
            if config and 'log_file' in config:
                log_error(f"Check log file for details: {config['log_file']}")
        else:
            print(f"ERROR: {error_msg}")
        
        print(f"\n❌ Processing failed: {error}")
        if config and 'log_file' in config:
            print(f"📝 Check log file for details: {config['log_file']}")
    
    def _cleanup(self):
        """Cleanup resources and close logging."""
        if self.logger:
            LoggerManager.close()


def main():
    """Main entry point for the application."""
    app = TransformApplication()
    app.run()


if __name__ == "__main__":
    main()