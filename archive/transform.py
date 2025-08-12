
import sys
import os
from datetime import datetime

# Import our custom modules
from logger import setup_logging, LoggerManager, log_info, log_warning, log_error
from processor import ParquetProcessor, PROCESSING_MODES, CUDA_AVAILABLE, CUDA_REASON
from utils import (
    extract_participant_from_path, generate_log_filename,
    validate_directory, find_required_parquet_files
)


class TransformApplication:
    """
    Main application class for the parquet to CSV transformation pipeline.
    
    This class handles:
    - User interaction and input validation
    - Application initialization and cleanup
    - Processing mode selection and validation
    - Error handling and reporting
    """
    
    def __init__(self):
        """Initialize the Transform Application."""
        self.processor = None
        self.logger = None
    
    def run(self):
        """
        Main application entry point.
        
        Orchestrates the entire transformation process from user input
        to final CSV output with comprehensive logging and error handling.
        """
        try:
            self._print_banner()
            
            # Get user input
            config = self._get_user_configuration()
            
            # Setup logging system
            self._setup_logging(config)
            
            # Process files
            self._process_files(config)
            
            # Report success
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
║                           (Modular Architecture)                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print("Processing E4 wearable sensor data with enhanced logging and parallel processing")
        print("=" * 80)
    
    def _get_user_configuration(self):
        """
        Collect all user configuration through interactive prompts.
        
        Returns:
            dict: Configuration dictionary with all user settings
        """
        config = {}
        
        # Get directory path
        config['directory_path'] = self._get_directory_path()
        
        # Get processing mode
        config['processing_mode'] = self._get_processing_mode()
        
        # Get row limits for each file type
        config['file_row_limits'] = self._get_row_limits()
        
        # Get output folder
        config['output_folder'] = self._get_output_folder()
        
        # Get metadata path
        config['metadata_path'] = self._get_metadata_path()
        
        # Generate log file path
        config['log_file'] = generate_log_filename(config['directory_path'])
        
        # Extract participant ID for logging context
        config['participant_id'] = extract_participant_from_path(config['directory_path'])
        
        return config
    
    def _get_directory_path(self):
        """Get and validate directory path from user."""
        while True:
            directory_path = input("\nEnter the directory path containing the four parquet files: ").strip()
            
            if not directory_path:
                print("Error: Directory path cannot be empty. Please try again.")
                continue
                
            try:
                validate_directory(directory_path)
                # Test if we can find the required files
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
            processing_mode = input("\nProcessing mode (cpu/gpu/single, default: single): ").strip().lower()
            if not processing_mode:
                processing_mode = 'single'
            
            if processing_mode not in PROCESSING_MODES:
                print(f"Invalid mode. Please enter one of: {', '.join(PROCESSING_MODES.keys())}")
                continue
                
            # Handle GPU mode validation
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
                    rows_input = input(f"  {file_type.upper()} rows (default: all): ").strip()
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
        output_folder = input("\nEnter output folder (default: 'output'): ").strip()
        if not output_folder:
            output_folder = 'output'
        
        return output_folder
    
    def _get_metadata_path(self):
        """Get metadata CSV file path from user."""
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'metadata.csv')
        metadata_path = input(f"\nEnter metadata CSV file path (default: '{default_path}'): ").strip()
        
        if not metadata_path:
            metadata_path = default_path
        
        # Validate that the file exists
        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata file not found at '{metadata_path}'")
            print("Processing will continue but metadata fields may be empty")
        
        return metadata_path
    
    def _setup_logging(self, config):
        """
        Setup the logging system with configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.logger = setup_logging(
            config['log_file'], 
            config['participant_id']
        )
        
        # Log application start and configuration
        self.logger.section_header("Transform Process Started")
        log_info(f"Processing directory: {config['directory_path']}")
        log_info(f"Log file: {config['log_file']}")
        log_info(f"Participant ID: {config['participant_id']}")
        log_info(f"Processing mode: {config['processing_mode']}")
        log_info(f"Row limits: {config['file_row_limits']}")
        log_info(f"Output folder: {config['output_folder']}")
        log_info(f"Metadata path: {config['metadata_path']}")
    
    def _process_files(self, config):
        """
        Execute the main file processing pipeline.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.logger.section_header("Starting File Processing")
        
        # Initialize processor with metadata path
        self.processor = ParquetProcessor(config['metadata_path'])
        
        # Process all files in the directory
        output_files, total_processed_rows = self.processor.process_directory(
            config['directory_path'],
            config['file_row_limits'],
            config['output_folder'],
            config['processing_mode']
        )
        
        # Store results in config for reporting
        config['output_files'] = output_files
        config['total_processed_rows'] = total_processed_rows
    
    def _report_success(self, config):
        """
        Report successful completion with summary.
        
        Args:
            config (dict): Configuration dictionary with results
        """
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
        """
        Handle application errors with proper logging.
        
        Args:
            error (Exception): The error that occurred
            config (dict, optional): Configuration dictionary if available
        """
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
    """
    Main entry point for the application.
    
    Creates and runs the TransformApplication instance.
    """
    app = TransformApplication()
    app.run()


if __name__ == "__main__":
    main()