"""
Data Processor Classes for Parquet to CSV Transform

This module contains the main data processing classes that handle:
- DataFrame creation and transformation
- Parallel processing (CPU and GPU)
- File I/O operations
- Data validation and processing

Classes:
- DataFrameBuilder: Handles dataframe construction and column mapping
- ParallelProcessor: Manages CPU and GPU parallel processing
- ParquetProcessor: Main processing class that orchestrates the pipeline

Author: Transform Script
Date: 2024
"""

import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import platform

from logger import LoggerManager, log_info, log_warning, log_error
from utils import (
    extract_participant_info, get_device_position, format_device_timestamp,
    get_timezone_abbreviation, get_utc_offset, calculate_day_codes,
    calculate_actual_device_days, load_metadata, create_output_directory,
    get_file_size_gb, get_file_size_mb
)

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

# Processing modes configuration
PROCESSING_MODES = {
    'cpu': 'CPU parallel processing',
    'gpu': 'GPU/CUDA parallel processing', 
    'single': 'Single-threaded processing'
}


class DataFrameBuilder:
    """
    Handles construction of output dataframes with proper column mapping.
    
    This class is responsible for:
    - Creating output dataframe structure
    - Mapping input columns to output format
    - Calculating derived fields
    - Loading and applying metadata
    """
    
    def __init__(self, metadata_path=None):
        """Initialize the DataFrameBuilder."""
        self.logger = LoggerManager.get_logger()
        self.metadata_path = metadata_path
    
    def create_output_dataframe(self, df, parquet_file_path):
        """
        Create output dataframe with proper column mapping and transformations.
        
        Args:
            df (pd.DataFrame): Input dataframe from parquet file
            parquet_file_path (str): Path to source parquet file
            
        Returns:
            pd.DataFrame: Transformed output dataframe ready for CSV export
        """
        # Extract file information for logging and processing
        participant_id, data_type, subject_id_raw, subject_id_padded, subject_id_int = \
            extract_participant_info(parquet_file_path)
        
        # Update logger context with participant ID
        if self.logger:
            self.logger.set_participant_id(participant_id)
        
        log_info(f"Creating output dataframe structure for {participant_id} {data_type.upper()}...")
        log_info(f"Processing subject ID: {subject_id_int} (padded: {subject_id_padded})")
        
        # Load metadata for this subject
        if self.metadata_path:
            metadata_path = self.metadata_path
        else:
            metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'metadata.csv')
        subject_key_value, sex_value = load_metadata(metadata_path, subject_id_int)
        
        # Build output dataframe
        output_df = self._build_dataframe(
            df, participant_id, data_type, subject_id_padded,
            subject_key_value, sex_value, parquet_file_path
        )
        
        log_info(f"Dataframe structure created successfully for {participant_id} {data_type.upper()} ({len(output_df):,} rows)")
        
        return output_df
    
    def _build_dataframe(self, df, participant_id, data_type, subject_id_padded,
                        subject_key_value, sex_value, parquet_file_path):
        """
        Internal method to build the output dataframe structure.
        
        Args:
            df (pd.DataFrame): Input dataframe
            participant_id (str): Participant ID (e.g., 'SP114')
            data_type (str): Data type (e.g., 'ibi_left')
            subject_id_padded (str): Zero-padded subject ID
            subject_key_value (str): Subject key from metadata
            sex_value (str): Sex from metadata
            parquet_file_path (str): Path to source file
            
        Returns:
            pd.DataFrame: Built output dataframe
        """
        # Pre-calculate values that are used for all rows
        subject_key_data = pd.Series([subject_key_value] * len(df))
        sex_data = pd.Series([sex_value] * len(df))
        
        # Calculate time-based fields
        interview_date_data = self._get_interview_date(df)
        timezone_data = get_timezone_abbreviation(df.get('timezone', pd.Series()))
        utc_offset_data = get_utc_offset(df.get('timezone', pd.Series()))
        day_code_data = calculate_day_codes(df.get('timestamp_local', pd.Series()))
        
        # Calculate study period fields
        length_per = self._get_length_per(df)
        actual_dev_days = calculate_actual_device_days(df)
        
        # Calculate data collection time fields
        dc_start_date_data = self._get_dc_start_date(df)
        dc_start_time_data = self._get_dc_start_time(df)
        dc_end_date_data = self._get_dc_end_date(df)
        dc_end_time_data = self._get_dc_end_time(df)
        
        # Device and timestamp fields
        device_position = get_device_position(parquet_file_path)
        device_timestamp_data = format_device_timestamp(df.get('timestamp_utc', pd.Series()))
        
        log_info(f"Building column mappings for {participant_id} {data_type.upper()}...")
        
        # Create output dataframe efficiently using dictionary
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
            'device_timestamp': device_timestamp_data,
            'skin_temp': df.get('temp', pd.Series([''] * len(df)))
        }
        
        return pd.DataFrame(output_data)
    
    def _get_interview_date(self, df):
        """Get interview date from timestamp_local column."""
        timestamp_series = df.get('timestamp_local', pd.Series())
        if timestamp_series.empty:
            return ''
        interview_date = timestamp_series.dt.strftime('%m/%d/%Y')
        return interview_date
    
    def _get_length_per(self, df):
        """Calculate length per from day_of_study column."""
        length_per = df.get('day_of_study', pd.Series())
        length = length_per.max() + 1 if not length_per.empty else 0
        return length
    
    def _get_dc_start_date(self, df):
        """Get data collection start date."""
        timestamp_series = df.get('timestamp_utc', pd.Series())
        if timestamp_series.empty:
            return ''
        first_timestamp = timestamp_series.iloc[0]
        if first_timestamp is None or pd.isna(first_timestamp):
            return ''
        return first_timestamp.strftime('%m/%d/%Y')
    
    def _get_dc_start_time(self, df):
        """Get data collection start time."""
        timestamp_series = df.get('timestamp_utc', pd.Series())
        if timestamp_series.empty:
            return ''
        first_timestamp = timestamp_series.iloc[0]
        if first_timestamp is None or pd.isna(first_timestamp):
            return ''
        return first_timestamp.strftime('%H:%M')
    
    def _get_dc_end_date(self, df):
        """Get data collection end date."""
        timestamp_series = df.get('timestamp_utc', pd.Series())
        if timestamp_series.empty:
            return ''
        last_timestamp = timestamp_series.iloc[-1]
        if last_timestamp is None or pd.isna(last_timestamp):
            return ''
        return last_timestamp.strftime('%m/%d/%Y')
    
    def _get_dc_end_time(self, df):
        """Get data collection end time."""
        timestamp_series = df.get('timestamp_utc', pd.Series())
        if timestamp_series.empty:
            return ''
        last_timestamp = timestamp_series.iloc[-1]
        if last_timestamp is None or pd.isna(last_timestamp):
            return ''
        return last_timestamp.strftime('%H:%M')


class ParallelProcessor:
    """
    Handles parallel processing operations for both CPU and GPU modes.
    
    This class manages:
    - CUDA/GPU acceleration when available
    - CPU multiprocessing for parallel file processing
    - Processing mode validation and fallback
    """
    
    @staticmethod
    def is_cuda_available():
        """Check if CUDA processing is available."""
        return CUDA_AVAILABLE
    
    @staticmethod
    def get_cuda_unavailable_reason():
        """Get reason why CUDA is unavailable."""
        return CUDA_REASON
    
    @staticmethod
    def process_with_cuda(parquet_file_path, num_rows=-1):
        """
        Process parquet file using CUDA/GPU acceleration.
        
        Args:
            parquet_file_path (str): Path to parquet file
            num_rows (int): Number of rows to read (-1 for all)
            
        Returns:
            pd.DataFrame: Loaded dataframe converted from GPU
            
        Raises:
            ImportError: If CUDA libraries are not available
        """
        if not CUDA_AVAILABLE:
            raise ImportError(f"CUDA libraries not available: {CUDA_REASON}")
        
        participant_id, data_type, _, _, _ = extract_participant_info(parquet_file_path)
        log_info(f"Using CUDA/GPU processing for {participant_id} {data_type.upper()}")
        
        # Read parquet with cudf for GPU acceleration
        if num_rows != -1:
            df_gpu = cudf.read_parquet(parquet_file_path, nrows=num_rows)
        else:
            df_gpu = cudf.read_parquet(parquet_file_path)
        
        log_info(f"Loaded {len(df_gpu):,} rows on GPU")
        
        # Convert back to pandas for compatibility with existing functions
        df = df_gpu.to_pandas()
        
        return df
    
    @staticmethod
    def process_file_worker(args):
        """
        Worker function for CPU parallel processing.
        
        Args:
            args (tuple): (parquet_file_path, num_rows, output_folder)
            
        Returns:
            tuple: (output_file_path, processed_rows_count)
        """
        parquet_file_path, num_rows, output_folder = args
        try:
            # Import here to avoid circular imports in multiprocessing
            from processor import ParquetProcessor
            processor = ParquetProcessor()
            return processor.process_single_file(parquet_file_path, num_rows, output_folder, 'single')
        except Exception as e:
            participant_id, data_type, _, _, _ = extract_participant_info(parquet_file_path)
            log_error(f"Error in worker process for {participant_id} {data_type.upper()}: {e}")
            raise


class ParquetProcessor:
    """
    Main processor class that orchestrates the entire parquet to CSV pipeline.
    
    This class coordinates:
    - File reading and validation
    - Processing mode selection
    - DataFrame transformation
    - CSV output generation
    - Progress tracking and logging
    """
    
    def __init__(self, metadata_path=None):
        """Initialize the ParquetProcessor."""
        self.dataframe_builder = DataFrameBuilder(metadata_path)
        self.parallel_processor = ParallelProcessor()
        self.metadata_path = metadata_path
    
    def process_single_file(self, parquet_file_path, num_rows=-1, output_folder='output', processing_mode='single'):
        """
        Process a single parquet file with optional parallel processing.
        
        Args:
            parquet_file_path (str): Path to parquet file
            num_rows (int): Number of rows to process (-1 for all)
            output_folder (str): Output directory for CSV files
            processing_mode (str): Processing mode ('single', 'cpu', 'gpu')
            
        Returns:
            tuple: (output_file_path, processed_rows_count)
            
        Raises:
            FileNotFoundError: If input file doesn't exist
        """
        if not os.path.exists(parquet_file_path):
            raise FileNotFoundError(f"File '{parquet_file_path}' does not exist.")
        
        # Extract file information for logging
        participant_id, data_type, _, _, _ = extract_participant_info(parquet_file_path)
        filename = os.path.basename(parquet_file_path)
        file_size = get_file_size_gb(parquet_file_path)
        
        log_info(f"Processing {participant_id} {data_type.upper()} file: {filename} ({file_size:.2f} GB)")
        log_info(f"Using {PROCESSING_MODES.get(processing_mode, 'single-threaded')} mode...")
        
        # Load data based on processing mode
        df = self._load_data(parquet_file_path, num_rows, processing_mode)
        
        # Transform data
        output_df = self.dataframe_builder.create_output_dataframe(df, parquet_file_path)
        
        # Write CSV output
        output_file = self._write_csv_output(output_df, parquet_file_path, output_folder, 
                                           participant_id, data_type)
        
        return output_file, len(output_df)
    
    def _load_data(self, parquet_file_path, num_rows, processing_mode):
        """
        Load data from parquet file based on processing mode.
        
        Args:
            parquet_file_path (str): Path to parquet file
            num_rows (int): Number of rows to load
            processing_mode (str): Processing mode
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        # Use GPU processing if available and requested
        if processing_mode == 'gpu' and CUDA_AVAILABLE:
            return self.parallel_processor.process_with_cuda(parquet_file_path, num_rows)
        
        # Fall back to pandas for single/cpu mode or if GPU unavailable
        if processing_mode == 'gpu' and not CUDA_AVAILABLE:
            log_warning(f"CUDA not available: {CUDA_REASON}, falling back to single-threaded processing")
        
        return self._load_with_pandas(parquet_file_path, num_rows)
    
    def _load_with_pandas(self, parquet_file_path, num_rows):
        """
        Load data using pandas with progress bar.
        
        Args:
            parquet_file_path (str): Path to parquet file
            num_rows (int): Number of rows to load
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        file_size = get_file_size_gb(parquet_file_path)
        
        log_info("Reading parquet file...")
        with tqdm(desc="Loading data", unit="MB") as pbar:
            # Use efficient parquet reading with pyarrow engine
            df = pd.read_parquet(parquet_file_path, engine='pyarrow')
            pbar.update(file_size * 1024)  # Convert GB to MB for progress bar
        
        log_info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        
        # Limit rows if specified
        if num_rows != -1:
            log_info(f"Limiting to first {num_rows:,} rows...")
            df = df.head(num_rows)
        
        return df
    
    def _write_csv_output(self, output_df, parquet_file_path, output_folder,
                         participant_id, data_type):
        """
        Write output dataframe to CSV file.
        
        Args:
            output_df (pd.DataFrame): Output dataframe
            parquet_file_path (str): Original parquet file path
            output_folder (str): Output directory
            participant_id (str): Participant ID for logging
            data_type (str): Data type for logging
            
        Returns:
            str: Path to created CSV file
        """
        # Generate output file path
        base_filename = os.path.splitext(os.path.basename(parquet_file_path))[0] + '_converted.csv'
        output_file = os.path.join(output_folder, base_filename)
        
        log_info(f"Writing {participant_id} {data_type.upper()} CSV to: {output_file}")
        
        # Write CSV with chunking for large files
        output_df.to_csv(output_file, index=False, chunksize=10000)
        
        # Log completion with file size
        output_size = get_file_size_mb(output_file)
        log_info(f"{participant_id} {data_type.upper()} CSV file created successfully: {output_file} ({output_size:.2f} MB)")
        
        return output_file
    
    def process_directory(self, directory_path, file_row_limits=None, 
                         output_folder='output', processing_mode='single'):
        """
        Process all four parquet files in a directory.
        
        Args:
            directory_path (str): Directory containing parquet files
            file_row_limits (dict): Row limits per file type
            output_folder (str): Output directory
            processing_mode (str): Processing mode ('single', 'cpu', 'gpu')
            
        Returns:
            tuple: (list_of_output_files, total_processed_rows)
        """
        from utils import find_required_parquet_files
        
        # Create output directory
        create_output_directory(output_folder)
        
        # Find all required parquet files
        found_files, subject_id = find_required_parquet_files(directory_path)
        
        # Set default row limits if not provided
        if file_row_limits is None:
            file_row_limits = {'acc': -1, 'eda': -1, 'ibi': -1, 'temp': -1}
        
        # Prepare processing tasks
        processing_tasks = self._prepare_processing_tasks(found_files, file_row_limits, output_folder)
        
        # Execute processing based on mode
        if processing_mode == 'cpu' and len(processing_tasks) > 1:
            return self._process_parallel_cpu(processing_tasks)
        else:
            return self._process_sequential(processing_tasks, processing_mode)
    
    def _prepare_processing_tasks(self, found_files, file_row_limits, output_folder):
        """Prepare list of processing tasks."""
        file_types = ['acc', 'eda', 'ibi', 'temp']
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
        
        return processing_tasks
    
    def _process_parallel_cpu(self, processing_tasks):
        """Process files in parallel using CPU multiprocessing."""
        log_info(f"Using CPU parallel processing with {min(cpu_count(), len(processing_tasks))} processes")
        
        # Prepare arguments for multiprocessing
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