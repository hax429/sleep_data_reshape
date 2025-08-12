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
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# =============================================================================
# CONFIGURATION SECTION - All user inputs and defaults
# =============================================================================

# Default paths
DEFAULT_OUTPUT_FOLDER = '/Users/hax429/Developer/Internship/reshape/data/output'
DEFAULT_METADATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'metadata.csv')

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
    'directory_path': "Enter the directory path containing the four parquet files: ",
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
        
        output_df = self._build_dataframe(
            df, participant_id, data_type, subject_id_padded,
            subject_key_value, sex_value, parquet_file_path
        )
        
        log_info(f"Dataframe structure created successfully for {participant_id} {data_type.upper()} ({len(output_df):,} rows)")
        
        return output_df
    
    def _build_dataframe(self, df, participant_id, data_type, subject_id_padded,
                        subject_key_value, sex_value, parquet_file_path):
        """Internal method to build the output dataframe structure."""
        subject_key_data = pd.Series([subject_key_value] * len(df))
        sex_data = pd.Series([sex_value] * len(df))
        
        interview_date_data = self._get_interview_date(df)
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
            'interview_age': df.get('interview_age', pd.Series([''] * len(df))),
            'sex': sex_data,
            'mt_pilot_vib_device': df.get('mt_pilot_vib_device', pd.Series([''] * len(df))),
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
        
        config['directory_path'] = self._get_directory_path()
        config['processing_mode'] = self._get_processing_mode()
        config['file_row_limits'] = self._get_row_limits()
        config['output_folder'] = self._get_output_folder()
        config['metadata_path'] = self._get_metadata_path()
        
        config['log_file'] = generate_log_filename(config['directory_path'])
        config['participant_id'] = extract_participant_from_path(config['directory_path'])
        
        return config
    
    def _get_directory_path(self):
        """Get and validate directory path from user."""
        while True:
            directory_path = input(f"\n{PROMPTS['directory_path']}").strip()
            
            if not directory_path:
                print("Error: Directory path cannot be empty. Please try again.")
                continue
                
            try:
                validate_directory(directory_path)
                find_required_parquet_files(directory_path)
                return directory_path
            except (FileNotFoundError, ValueError) as e:
                print(f"Error: {e}")
                retry = input("Would you like to try another path? (y/n, default: y): ").strip().lower()
                if retry in ['n', 'no']:
                    print("Exiting...")
                    sys.exit(0)
    
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
        log_info(f"Processing directory: {config['directory_path']}")
        log_info(f"Log file: {config['log_file']}")
        log_info(f"Participant ID: {config['participant_id']}")
        log_info(f"Processing mode: {config['processing_mode']}")
        log_info(f"Row limits: {config['file_row_limits']}")
        log_info(f"Output folder: {config['output_folder']}")
        log_info(f"Metadata path: {config['metadata_path']}")
    
    def _process_files(self, config):
        """Execute the main file processing pipeline."""
        self.logger.section_header("Starting File Processing")
        
        self.processor = ParquetProcessor(config['metadata_path'])
        
        output_files, total_processed_rows = self.processor.process_directory(
            config['directory_path'],
            config['file_row_limits'],
            config['output_folder'],
            config['processing_mode']
        )
        
        config['output_files'] = output_files
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