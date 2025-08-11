import pandas as pd
import os
from tqdm import tqdm
import sys
from datetime import datetime
import pytz
from multiprocessing import Pool, cpu_count
import logging

# Optional imports for CUDA support
try:
    import cudf
    import cupy as cp
    import platform
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

# Processing modes
PROCESSING_MODES = {
    'cpu': 'CPU parallel processing',
    'gpu': 'GPU/CUDA parallel processing', 
    'single': 'Single-threaded processing'
}


def getInterviewDate(df):
    timestamp_series = df.get('timestamp_local', pd.Series())
    if timestamp_series.empty:
        return ''
    interview_date = timestamp_series.dt.strftime('%m/%d/%Y')
    return interview_date

def getLengthPer(df):
    length_per = df.get('day_of_study', pd.Series())
    length = length_per.max() + 1 if not length_per.empty else 0
    return length


def getTimeZone(df): #static for all rows
    timezone_series = df.get('timezone', pd.Series())
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
    except:
        # If timezone conversion fails, return original value
        return timezone_value


def getUTCOffset(df): #static for all rows
    timezone_series = df.get('timezone', pd.Series())
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
    except:
        # If timezone conversion fails, return empty string
        return ''


def getDevicePosition(parquet_file_path):
    filename = os.path.basename(parquet_file_path).lower()
    if 'left' in filename:
        return 0
    elif 'right' in filename:
        return 1
    else:
        return ''


def getDayCode(df):
    timestamp_series = df.get('timestamp_local', pd.Series())
    if timestamp_series.empty:
        return pd.Series([''] * len(df))
    
    try:
        # Calculate day code for each timestamp
        day_codes = []
        for timestamp in timestamp_series:
            if timestamp is None or pd.isna(timestamp):
                day_codes.append('')
            else:
                # Get day of week (0=Monday, 1=Tuesday, ..., 6=Sunday)
                daycode = timestamp.weekday()
                
                # Convert to your classification:
                # 0 = Weekday (Sunday-Thursday) = weekday() 6,0,1,2,3
                # 1 = Weekend day (Friday-Saturday) = weekday() 4,5
                if daycode in [4, 5]:  # Friday, Saturday
                    day_codes.append(1)
                else:  # Sunday, Monday, Tuesday, Wednesday, Thursday
                    day_codes.append(0)
        
        return pd.Series(day_codes)
    except:
        return pd.Series([''] * len(df))


def getActualDevDays(df): #static for all rows
    if df.empty or 'device_worn' not in df.columns or 'timestamp_local' not in df.columns:
        return 0
    
    # Convert timestamp to date for grouping by day
    df['date'] = df['timestamp_local'].dt.date
    
    worn_days = []
    
    # Group by date and calculate device worn percentage for each day
    for date, day_data in df.groupby('date'):
        total_records = len(day_data)
        worn_records = (day_data['device_worn'] == 1).sum()
        
        # Calculate percentage worn for this day
        worn_percentage = worn_records / total_records if total_records > 0 else 0
        
        # If device worn more than 90% of the day, add to worn days list
        if worn_percentage > 0.9:
            worn_days.append(date)
    log_info("Actual Dev Days: {}".format(worn_days))
    return len(worn_days)


def getDCStartDate(df):
    timestamp_series = df.get('timestamp_utc', pd.Series())
    if timestamp_series.empty:
        return ''
    first_timestamp = timestamp_series.iloc[0]
    if first_timestamp is None or pd.isna(first_timestamp):
        return ''
    return first_timestamp.strftime('%m/%d/%Y')


def getDCStartTime(df):
    timestamp_series = df.get('timestamp_utc', pd.Series())
    if timestamp_series.empty:
        return ''
    first_timestamp = timestamp_series.iloc[0]
    if first_timestamp is None or pd.isna(first_timestamp):
        return ''
    return first_timestamp.strftime('%H:%M')


def getDCEndDate(df):
    timestamp_series = df.get('timestamp_utc', pd.Series())
    if timestamp_series.empty:
        return ''
    last_timestamp = timestamp_series.iloc[-1]
    if last_timestamp is None or pd.isna(last_timestamp):
        return ''
    return last_timestamp.strftime('%m/%d/%Y')


def getDCEndTime(df):
    timestamp_series = df.get('timestamp_utc', pd.Series())
    if timestamp_series.empty:
        return ''
    last_timestamp = timestamp_series.iloc[-1]
    if last_timestamp is None or pd.isna(last_timestamp):
        return ''
    return last_timestamp.strftime('%H:%M')

def getDeviceTimestamp(df):
    timestamp_utc_series = df.get('timestamp_utc', pd.Series())
    if not timestamp_utc_series.empty:
        device_timestamp_data = timestamp_utc_series.dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    else:
        device_timestamp_data = pd.Series([''] * len(df))
    return device_timestamp_data



class ProcessLogger:
    """Enhanced logging system for batch processing with participant tracking."""
    
    def __init__(self, log_file=None, participant_id=None):
        self.participant_id = participant_id
        self.log_file = log_file
        self.logger = logging.getLogger(f'transform_logger_{os.getpid()}')
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Set logger level
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        console_formatter = logging.Formatter('%(asctime)s | %(levelname)-5s | %(message)s', 
                                            datefmt='%Y-%m-%d %H:%M:%S')
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)-5s | %(participant_id)s | %(message)s', 
                                         datefmt='%Y-%m-%d %H:%M:%S')
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _log(self, level, message):
        """Internal logging method with participant ID context."""
        extra = {'participant_id': self.participant_id or 'SYSTEM'}
        self.logger.log(level, message, extra=extra)
    
    def info(self, message):
        """Log info message."""
        self._log(logging.INFO, message)
    
    def warning(self, message):
        """Log warning message."""
        self._log(logging.WARNING, message)
    
    def error(self, message):
        """Log error message."""
        self._log(logging.ERROR, message)
    
    def debug(self, message):
        """Log debug message."""
        self._log(logging.DEBUG, message)
    
    def set_participant_id(self, participant_id):
        """Update participant ID for context."""
        self.participant_id = participant_id
    
    def section_header(self, title):
        """Log a section header for better organization."""
        separator = "=" * 60
        self.info(separator)
        self.info(f" {title.upper()}")
        self.info(separator)
    
    def file_progress(self, file_type, current, total):
        """Log file processing progress."""
        self.info(f"[{file_type.upper()}] Processing file {current}/{total}")


# Global logger instance
process_logger = None


def setup_logging(log_file=None, participant_id=None):
    """Initialize the global logging system."""
    global process_logger
    process_logger = ProcessLogger(log_file, participant_id)
    return process_logger


def log_info(message):
    """Backward compatibility function."""
    if process_logger:
        process_logger.info(message)
    else:
        print(message)


def log_warning(message):
    """Log warning message."""
    if process_logger:
        process_logger.warning(message)
    else:
        print(f"WARNING: {message}")


def log_error(message):
    """Log error message."""
    if process_logger:
        process_logger.error(message)
    else:
        print(f"ERROR: {message}")


def process_parquet_with_cuda(parquet_file_path, num_rows=-1):
    """Process parquet file using CUDA/GPU acceleration."""
    if not CUDA_AVAILABLE:
        raise ImportError("CUDA libraries (cudf, cupy) not available. Install using: pip install cudf-cu12 cupy-cuda12x")
    
    log_info(f"Using CUDA/GPU processing for: {os.path.basename(parquet_file_path)}")
    
    # Read parquet with cudf for GPU acceleration
    if num_rows != -1:
        df_gpu = cudf.read_parquet(parquet_file_path, nrows=num_rows)
    else:
        df_gpu = cudf.read_parquet(parquet_file_path)
    
    log_info(f"Loaded {len(df_gpu):,} rows on GPU")
    
    # Convert back to pandas for compatibility with existing functions
    df = df_gpu.to_pandas()
    
    return df


def process_single_file_worker(args):
    """Worker function for CPU parallel processing."""
    parquet_file_path, num_rows, output_folder = args
    try:
        return process_single_parquet_file(parquet_file_path, num_rows, output_folder, 'single')
    except Exception as e:
        log_info(f"Error in worker process for {parquet_file_path}: {e}")
        raise

def create_output_dataframe(df, parquet_file_path):
    # Extract participant ID and data type from file path
    filename = os.path.basename(parquet_file_path)
    src_subject_id_raw = filename.split('_')[0].lstrip('SP')
    src_subject_id_padded = src_subject_id_raw.zfill(3)
    src_subject_id_int = int(src_subject_id_raw)
    
    # Extract data type info (e.g., "ibi_left", "acc_right", etc.)
    parts = filename.split('_')
    if len(parts) >= 4:
        data_type = f"{parts[2]}_{parts[3].replace('.parquet', '')}"  # e.g., "ibi_left"
    else:
        data_type = "unknown"
    
    # Update logger with participant ID if available
    participant_id = f"SP{src_subject_id_raw}"
    if process_logger:
        process_logger.set_participant_id(participant_id)
    
    log_info(f"Creating output dataframe structure for {participant_id} {data_type.upper()}...")
    log_info(f"Processing subject ID: {src_subject_id_int} (padded: {src_subject_id_padded})")
    
    # Read metadata.csv to get subjectkey and sex for this subject
    metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metadata.csv')
    subject_key_value = ''
    sex_value = ''
    
    if os.path.exists(metadata_path):
        try:
            metadata_df = pd.read_csv(metadata_path)
            subject_row = metadata_df[metadata_df['src_subject_id'] == src_subject_id_int]
            if not subject_row.empty:
                subject_key_value = subject_row.iloc[0]['subjectkey']
                sex_value = subject_row.iloc[0]['sex']
                log_info(f"Found metadata: subjectkey={subject_key_value}, sex={sex_value}")
            else:
                log_warning(f"No metadata found for subject ID {src_subject_id_int}")
        except Exception as e:
            log_warning(f"Could not read metadata.csv: {e}")
    else:
        log_warning("metadata.csv not found")
    
    # Pre-calculate values that are used for all rows
    subject_key_data = pd.Series([subject_key_value] * len(df))
    sex_data = pd.Series([sex_value] * len(df))

    interview_date_data = getInterviewDate(df)
    length_per = getLengthPer(df)
    timezone_data = getTimeZone(df)
    utc_offset_data = getUTCOffset(df)
    actual_dev_days = getActualDevDays(df)
    day_code_data = getDayCode(df)
    dc_start_date_data = getDCStartDate(df)
    dc_start_time_data = getDCStartTime(df)
    dc_end_date_data = getDCEndDate(df)
    dc_end_time_data = getDCEndTime(df)
    device_position = getDevicePosition(parquet_file_path)
    device_timestamp_data = getDeviceTimestamp(df)
    
    # Create output dataframe efficiently using dictionary
    log_info(f"Building column mappings for {participant_id} {data_type.upper()}...")
    output_data = {
        'subjectkey': subject_key_data,
        'src_subject_id': src_subject_id_padded,
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
        'device_timestamp': device_timestamp_data,
        'skin_temp': df.get('temp', pd.Series([''] * len(df)))

    }
    
    output_df = pd.DataFrame(output_data)
    log_info(f"Dataframe structure created successfully for {participant_id} {data_type.upper()} ({len(output_df):,} rows)")
    
    return output_df


def find_parquet_files(directory_path):
    """Find the four required parquet files in the directory."""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a directory.")
    
    # Find all parquet files in the directory
    all_files = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
    
    if not all_files:
        raise FileNotFoundError(f"No parquet files found in directory '{directory_path}'.")
    
    # Extract subject ID from the first file
    sample_file = all_files[0]
    src_subject_id_raw = sample_file.split('_')[0].lstrip('SP')
    
    # Define the four required file patterns
    required_files = [
        f"SP{src_subject_id_raw}_e4_acc_left.parquet",
        f"SP{src_subject_id_raw}_e4_eda_left.parquet",
        f"SP{src_subject_id_raw}_e4_ibi_left.parquet",
        f"SP{src_subject_id_raw}_e4_temp_left.parquet"
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
    
    log_info(f"Found all required files for subject SP{src_subject_id_raw}:")
    for file_name in required_files:
        log_info(f"  - {file_name}")
    
    return found_files, src_subject_id_raw


def process_single_parquet_file(parquet_file_path, num_rows=-1, output_folder='output', processing_mode='single'):
    """Process a single parquet file with optional parallel processing."""
    if not os.path.exists(parquet_file_path):
        raise FileNotFoundError(f"File '{parquet_file_path}' does not exist.")
    
    # Extract participant ID and data type for logging
    filename = os.path.basename(parquet_file_path)
    src_subject_id_raw = filename.split('_')[0].lstrip('SP')
    participant_id = f"SP{src_subject_id_raw}"
    
    parts = filename.split('_')
    if len(parts) >= 4:
        data_type = f"{parts[2]}_{parts[3].replace('.parquet', '')}"  # e.g., "ibi_left"
    else:
        data_type = "unknown"
    
    file_size = os.path.getsize(parquet_file_path) / (1024**3)  # GB
    log_info(f"Processing {participant_id} {data_type.upper()} file: {filename} ({file_size:.2f} GB)")

    log_info(f"Using {PROCESSING_MODES.get(processing_mode, 'single-threaded')} mode...")
    
    # Load data based on processing mode
    if processing_mode == 'gpu' and CUDA_AVAILABLE:
        df = process_parquet_with_cuda(parquet_file_path, num_rows)
    else:
        if processing_mode == 'gpu' and not CUDA_AVAILABLE:
            log_info("Warning: CUDA not available, falling back to single-threaded processing")
        
        log_info("Reading parquet file...")
        with tqdm(desc="Loading data", unit="MB") as pbar:
            # Use efficient parquet reading with pyarrow engine
            df = pd.read_parquet(parquet_file_path, engine='pyarrow')
            pbar.update(file_size * 1024)  # Convert GB to MB for progress bar
        
        log_info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        
        if num_rows != -1:
            log_info(f"Limiting to first {num_rows:,} rows...")
            df = df.head(num_rows)
    
    output_df = create_output_dataframe(df, parquet_file_path)
    
    # Generate output file path in the output folder
    base_filename = os.path.splitext(os.path.basename(parquet_file_path))[0] + '_converted.csv'
    output_file = os.path.join(output_folder, base_filename)
    
    log_info(f"Writing {participant_id} {data_type.upper()} CSV to: {output_file}")
    output_df.to_csv(output_file, index=False, chunksize=10000)
    
    output_size = os.path.getsize(output_file) / (1024**2)  # MB
    log_info(f"{participant_id} {data_type.upper()} CSV file created successfully: {output_file} ({output_size:.2f} MB)")
    
    return output_file, len(output_df)


def process_parquet_directory(directory_path, file_row_limits=None, output_folder='output', processing_mode='single'):
    """Process all four parquet files in a directory.
    
    Args:
        directory_path: Path to directory containing the four parquet files
        file_row_limits: Dict mapping file types to row limits, e.g. {'acc': 100, 'eda': 200, 'ibi': -1, 'temp': 50}
        output_folder: Output directory for converted CSV files
        processing_mode: 'cpu', 'gpu', or 'single' for parallel processing mode
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        log_info(f"Created output folder: {output_folder}")
    
    # Find all required parquet files
    found_files, subject_id = find_parquet_files(directory_path)
    
    # Set default row limits if not provided
    if file_row_limits is None:
        file_row_limits = {'acc': -1, 'eda': -1, 'ibi': -1, 'temp': -1}
    
    processed_files = []
    total_processed_rows = 0
    
    # Process each file type
    file_types = ['acc', 'eda', 'ibi', 'temp']
    
    # Prepare file processing tasks
    processing_tasks = []
    for file_type in file_types:
        # Find the matching file
        matching_file = None
        for file_name, file_path in found_files.items():
            if f"_e4_{file_type}_left.parquet" in file_name:
                matching_file = file_path
                break
        
        if matching_file:
            num_rows = file_row_limits.get(file_type, -1)
            processing_tasks.append((matching_file, num_rows, output_folder, file_type))
    
    if processing_mode == 'cpu' and len(processing_tasks) > 1:
        # CPU parallel processing
        log_info(f"Using CPU parallel processing with {min(cpu_count(), len(processing_tasks))} processes")
        
        # Prepare arguments for multiprocessing
        worker_args = [(task[0], task[1], task[2]) for task in processing_tasks]
        
        try:
            with Pool(processes=min(cpu_count(), len(processing_tasks))) as pool:
                results = pool.map(process_single_file_worker, worker_args)
                
            for i, (output_file, processed_rows) in enumerate(results):
                file_type = processing_tasks[i][3]
                processed_files.append(output_file)
                total_processed_rows += processed_rows
                log_info(f"Successfully processed {file_type.upper()}: {processed_rows:,} rows")
                
        except Exception as e:
            log_error(f"Error in parallel processing: {e}")
            raise
    else:
        # Sequential processing (single or gpu mode)
        for matching_file, num_rows, _, file_type in processing_tasks:
            log_info(f"\n{'='*50}")
            log_info(f"Processing {file_type.upper()} file with {num_rows if num_rows != -1 else 'all'} rows...")
            
            try:
                output_file, processed_rows = process_single_parquet_file(
                    matching_file, num_rows, output_folder, processing_mode
                )
                processed_files.append(output_file)
                total_processed_rows += processed_rows
                log_info(f"Successfully processed {file_type.upper()}: {processed_rows:,} rows")
            except Exception as e:
                log_error(f"Error processing {file_type.upper()} file: {e}")
                raise
    
    return processed_files, total_processed_rows


def main():
    # Initialize basic logging first
    print("Parquet to CSV Converter - Multi-file Directory Mode")
    print("=" * 60)
    
    directory_path = input("Enter the directory path containing the four parquet files: ").strip()
    
    # Set up enhanced logging system
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(os.path.dirname(directory_path), f"transform_log_{timestamp}.txt")
    
    # Try to extract participant ID from directory path for logging context
    try:
        # Look for SP followed by digits in the directory path
        import re
        match = re.search(r'SP(\d+)', directory_path)
        participant_id = f"SP{match.group(1)}" if match else "UNKNOWN"
    except:
        participant_id = "UNKNOWN"
    
    # Initialize the logging system
    logger = setup_logging(log_file, participant_id)
    logger.section_header("Transform Process Started")
    log_info(f"Processing directory: {directory_path}")
    log_info(f"Log file: {log_file}")
    log_info(f"Participant ID: {participant_id}")
    
    # Get processing mode
    log_info("\nSelect processing mode:")
    log_info("  'cpu' - CPU parallel processing")
    log_info("  'gpu' - GPU/CUDA parallel processing")  
    log_info("  'single' - Single-threaded processing")
    
    while True:
        processing_mode = input("Processing mode (cpu/gpu/single, default: single): ").strip().lower()
        if not processing_mode:
            processing_mode = 'single'
        if processing_mode in PROCESSING_MODES:
            break
        log_info(f"Invalid mode. Please enter one of: {', '.join(PROCESSING_MODES.keys())}")
    
    # Check CUDA availability for GPU mode
    if processing_mode == 'gpu' and not CUDA_AVAILABLE:
        log_warning(f"CUDA not available: {CUDA_REASON}")
        use_fallback = input("Fall back to single-threaded processing? (y/n, default: y): ").strip().lower()
        if use_fallback in ['', 'y', 'yes']:
            processing_mode = 'single'
            log_info("Falling back to single-threaded processing")
        else:
            log_error("User chose to exit due to CUDA unavailability")
            sys.exit(1)
    
    # Get row limits for each file type
    log_info("\nEnter number of rows to process for each file type (-1 for all rows):")
    file_row_limits = {}
    
    file_types = ['acc', 'eda', 'ibi', 'temp']
    for file_type in file_types:
        while True:
            try:
                rows_input = input(f"  {file_type.upper()} rows: ").strip()
                if not rows_input:
                    file_row_limits[file_type] = -1
                    break
                file_row_limits[file_type] = int(rows_input)
                break
            except ValueError:
                log_info("    Error: Please enter a valid integer or press Enter for all rows.")
    
    output_folder = input("Enter output folder (default: 'output'): ").strip()
    if not output_folder:
        output_folder = 'output'

    try:
        log_info("\n" + "=" * 60)
        log_info("Starting processing...")
        log_info(f"Processing mode: {processing_mode}")
        log_info(f"Row limits: {file_row_limits}")
        
        output_files, total_processed_rows = process_parquet_directory(
            directory_path, file_row_limits, output_folder, processing_mode
        )
        
        logger.section_header("Processing completed successfully")
        log_info(f"Processed {len(output_files)} files:")
        for output_file in output_files:
            log_info(f"  - {output_file}")
        log_info(f"Total processed rows: {total_processed_rows:,}")
        log_info(f"Log file saved: {log_file}")
        
    except Exception as e:
        log_error(f"Error processing directory: {e}")
        log_error(f"Check log file for details: {log_file}")
        sys.exit(1)




if __name__ == "__main__":
    main()
