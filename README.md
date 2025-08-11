# Parquet to CSV Transform Pipeline

## Overview

A modular, production-ready pipeline for converting E4 wearable sensor data from Parquet files to CSV format with comprehensive logging, parallel processing, and metadata integration.

## Features

- 🔄 **Multi-file batch processing** (4 sensor files per participant)
- ⚡ **Multiple processing modes** (single-threaded, CPU parallel, GPU/CUDA)
- 📝 **Comprehensive logging** with participant tracking
- 📊 **Progress monitoring** and error handling
- 🔗 **Metadata integration** from CSV files
- 🏗️ **Modular architecture** for maintainability

## Architecture

### File Structure

```
transform-pipeline/
├── transform_new.py    # Main application entry point
├── logger.py          # Logging system with dual output
├── processor.py       # Data processing classes and pipeline logic
├── utils.py           # Utility functions for file operations
├── README.md          # This documentation
└── metadata.csv       # Participant metadata (optional)
```

### Module Breakdown

#### `transform_new.py` - Main Application
- **TransformApplication**: Main application orchestrator
- User interaction and input validation
- Configuration management
- Error handling and cleanup

#### `logger.py` - Logging System
- **ProcessLogger**: Enhanced logging with participant tracking
- **LoggerManager**: Global logger management
- Dual output (console + file) with different detail levels
- Structured logging for batch processing analysis

#### `processor.py` - Data Processing Pipeline
- **DataFrameBuilder**: Handles dataframe construction and column mapping
- **ParallelProcessor**: Manages CPU and GPU parallel processing
- **ParquetProcessor**: Main processing orchestrator
- CUDA support with automatic fallback

#### `utils.py` - Utility Functions
- File operations and validation
- Data type extraction and formatting
- Date/time processing utilities
- Metadata loading and management

## Usage

### Basic Usage

```bash
python transform.py
```

The application will prompt for:
1. Directory path containing parquet files
2. Processing mode (single/cpu/gpu)
3. Row limits for each file type
4. Output directory

### Expected Input Files (per participant)

```
SP{id}_e4_acc_left.parquet    # Accelerometer data
SP{id}_e4_eda_left.parquet    # Electrodermal activity
SP{id}_e4_ibi_left.parquet    # Inter-beat interval
SP{id}_e4_temp_left.parquet   # Temperature data
```

### Output Files

```
SP{id}_e4_acc_left_converted.csv
SP{id}_e4_eda_left_converted.csv
SP{id}_e4_ibi_left_converted.csv
SP{id}_e4_temp_left_converted.csv
transform_log_{timestamp}.txt
```

## Processing Modes

### 1. Single-threaded (`single`)
- Sequential file processing
- Lowest memory usage
- Most compatible
- **Best for**: Small datasets, debugging

### 2. CPU Parallel (`cpu`)
- Parallel processing across multiple CPU cores
- Processes multiple files simultaneously
- **Best for**: Multiple files, CPU-bound operations

### 3. GPU/CUDA (`gpu`)
- GPU-accelerated parquet reading with cuDF
- Requires NVIDIA GPU and CUDA libraries
- **Best for**: Large files, GPU-enabled systems

## Logging System

### Console Output
Clean, real-time monitoring:
```
2024-08-11 14:30:25 | INFO  | Processing SP114 IBI_LEFT file...
2024-08-11 14:30:26 | WARN  | No metadata found for subject ID 114
2024-08-11 14:30:30 | INFO  | CSV file created successfully (2.35 MB)
```

### File Output
Comprehensive logging with participant context:
```
2024-08-11 14:30:25 | INFO  | SP114 | Building column mappings for SP114 IBI_LEFT...
2024-08-11 14:30:26 | WARN  | SP114 | No metadata found for subject ID 114
2024-08-11 14:30:30 | INFO  | SP114 | Writing SP114 IBI_LEFT CSV to: output/SP114_e4_ibi_left_converted.csv
```

### Log Analysis

Search by participant:
```bash
grep "SP114" transform_log_20240811_143025.txt
```

Search by error level:
```bash
grep "ERROR" transform_log_20240811_143025.txt
```

Search by data type:
```bash
grep "IBI_LEFT" transform_log_20240811_143025.txt
```

## Column Mapping

### Input Columns (from Parquet)
- `timestamp_utc`, `timestamp_local`
- `device_worn`, `day_of_study`
- `temp`, `timezone`
- `interview_age`
- Various sensor-specific columns

### Output Columns (CSV)
- `subjectkey`, `src_subject_id`
- `interview_date`, `interview_age`, `sex`
- `length_per`, `max_devdays`, `actual_devdays`
- `tlfb_daynum`, `day_code`
- `site_time_zone`, `utc_offset`
- `dc_start_date`, `dc_start_time`
- `dc_end_date`, `dc_end_time`
- `device_position`, `device_timestamp`
- `skin_temp`

## Installation

### Basic Requirements
```bash
pip install pandas pyarrow tqdm pytz
```

### GPU/CUDA Support (Optional)
```bash
# For NVIDIA GPU systems only
pip install cudf-cu12 cupy-cuda12x
```

Note: CUDA is not supported on macOS (requires NVIDIA GPU)

## Metadata Integration

Place a `metadata.csv` file in the same directory as the script with columns:
- `src_subject_id`: Integer subject ID
- `subjectkey`: Subject identifier key
- `sex`: Subject sex

Example:
```csv
src_subject_id,subjectkey,sex
114,NDAR_INV123ABC,M
115,NDAR_INV456DEF,F
```

## Error Handling

The application includes comprehensive error handling:

- **File validation**: Checks for required parquet files
- **Directory validation**: Ensures paths exist and are accessible
- **Processing errors**: Graceful handling with detailed logging
- **CUDA fallback**: Automatic fallback if GPU unavailable
- **User interruption**: Clean exit on Ctrl+C

## Performance Considerations

### Memory Usage
- Single mode: ~2x file size in memory
- CPU parallel: ~2x file size per process
- GPU mode: Additional GPU memory usage

### Processing Speed
- **Single**: Baseline performance
- **CPU parallel**: ~2-4x faster (depends on CPU cores and I/O)
- **GPU**: ~5-10x faster (depends on GPU and file size)

### Recommendations
- **Small files (<100MB)**: Use single mode
- **Multiple files**: Use CPU parallel mode
- **Large files (>1GB)**: Use GPU mode if available
- **Batch processing**: Use CPU parallel with logging analysis

## Troubleshooting

### Common Issues

1. **"No parquet files found"**
   - Ensure directory contains SP{id}_e4_*_left.parquet files
   - Check file naming convention

2. **"CUDA not available"**
   - Install CUDA libraries or use CPU/single mode
   - CUDA not supported on macOS

3. **"Memory error"**
   - Reduce row limits per file type
   - Use single-threaded mode
   - Process smaller batches

4. **"Permission denied"**
   - Check directory write permissions
   - Ensure output folder is accessible

### Debug Mode

For detailed debugging, check the log file which includes:
- Complete stack traces for errors
- Processing timestamps and duration
- Memory usage information
- File size statistics

## Development

### Code Structure
- **Classes**: Object-oriented design with clear responsibilities
- **Documentation**: Comprehensive docstrings and comments
- **Error handling**: Proper exception handling throughout
- **Logging**: Structured logging for debugging and monitoring

### Extending the Pipeline

1. **Add new data types**: Update file type lists in `utils.py`
2. **Custom column mapping**: Modify `DataFrameBuilder.create_output_dataframe()`
3. **New processing modes**: Extend `ParallelProcessor` class
4. **Additional validation**: Add checks in `utils.py`

### Testing

Run with small datasets first:
```bash
# Process only 100 rows per file type for testing
# Select this during the row limits prompt
```

## Version History

### v2.0 (Current)
- Modular architecture with separate files
- Enhanced logging with participant tracking
- Comprehensive error handling
- Improved documentation

### v1.0 (Legacy)
- Single file implementation
- Basic logging
- Limited error handling

## License

This project is for research and educational purposes.