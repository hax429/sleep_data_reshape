# Setup Instructions for New Environment

## Directory Structure Setup

Your project directory should be organized as follows:

```
project_root/
├── main_script.py                    # Main transformation script
├── data/
│   ├── metadata.csv                  # Subject metadata file
│   ├── visit_converted.csv           # Interview visit data
│   └── output/                       # Default output directory (auto-created)
├── logs/                             # Log files directory (auto-created)
│   ├── transform_log_SP1_20250812_143022.txt
│   └── transform_log_batch_5participants_20250812_144533.txt
└── ../final_processed_study_parquet/ # Participant folders (auto-discovered)
    ├── SP1/
    │   ├── SP1_e4_acc_left.parquet
    │   ├── SP1_e4_acc_right.parquet
    │   ├── SP1_e4_eda_left.parquet
    │   ├── SP1_e4_eda_right.parquet
    │   ├── SP1_e4_ibi_left.parquet
    │   ├── SP1_e4_ibi_right.parquet
    │   ├── SP1_e4_temp_left.parquet
    │   └── SP1_e4_temp_right.parquet
    ├── SP2/                          # More participant folders...
    ├── SP100/
    ├── SP104/
    ├── SP108/
    ├── SP111/
    ├── SP114/
    ├── SP119/
    └── SP121/
```

## Required Python Packages

Install the following packages:

```bash
pip install pandas
pip install pyarrow          # For parquet file support
pip install python-dateutil  # For date calculations
pip install pytz            # For timezone handling
pip install tqdm            # For progress bars
```

## Required Data Files

### 1. metadata.csv
Location: `./data/metadata.csv`

Required columns:
- `src_subject_id`: Integer subject ID (e.g., 1, 2, 3...)
- `subjectkey`: Subject key identifier
- `sex`: Subject sex (M/F)

Example:
```csv
src_subject_id,subjectkey,sex
1,NDARRV595LH6,F
2,NDARZD508AB7,F
3,NDARFW588UUE,F
```

### 2. visit_converted.csv
Location: `./data/visit_converted.csv`

Required columns:
- `src_subject_id`: Padded subject ID format (e.g., "001--1", "002--1")
- `interview_date`: Date in YYYY-MM-DD format
- `interview_age`: Age in months at time of interview
- `subjectkey`: Subject key identifier
- `sex`: Subject sex (M/F)

Example:
```csv
subjectkey,src_subject_id,interview_date,interview_age,sex
NDARRV595LH6,001--1,2020-01-28,294,F
NDARRV595LH6,001--1,2020-02-04,294,F
NDARRV595LH6,001--1,2020-02-11,294,F
```

### 3. Parquet Files
Location: `../final_processed_study_parquet/[SUBJECT_FOLDER]/`

Required naming convention:
- `SP[XXX]_e4_[TYPE]_[SIDE].parquet`
- Where:
  - `[XXX]` = Subject ID (e.g., 001, 002, 114)
  - `[TYPE]` = Data type (acc, eda, ibi, temp)
  - `[SIDE]` = Device position (left, right)

Required columns in parquet files:
- `timestamp_local`: Local timestamp for each measurement
- `timestamp_utc`: UTC timestamp for each measurement
- Data columns based on type:
  - **ACC**: `x_g`, `y_g`, `z_g` (acceleration values)
  - **EDA**: `eda` (electrodermal activity)
  - **IBI**: `ibi` (inter-beat interval)
  - **TEMP**: `temp` (temperature)

## Running the Script

1. **Navigate to project directory**:
   ```bash
   cd /path/to/your/project
   ```

2. **Run the script**:
   ```bash
   python main_script.py
   ```

3. **Follow the interactive prompts**:
   - **Select participants**: Choose which participants to process
     - `-1` or `all`: Process ALL participants
     - Single number: `1` (process SP1 only)
     - Multiple: `1,3,5` (process SP1, SP3, SP5)
     - Range: `1-5` (process SP1 through SP5)
     - Mixed: `1,3-5,7` (process SP1, SP3-SP5, SP7)
   - Choose processing mode (single/cpu/gpu)
   - Set row limits (optional)
   - Specify output folder (default: `data/output`)
   - Specify metadata path (default: `data/metadata.csv`)

## Expected Output

### CSV Files
Generated in the output directory with columns:
- `subjectkey`: Subject identifier
- `src_subject_id`: Padded subject ID
- `interview_date`: Date of measurement
- `interview_age`: **Calculated age in months for each timestamp**
- `sex`: Subject sex
- `mt_pilot_vib_device`: Device type (e.g., "Empatica E4" for participants 1-83)
- `watch_device_type`: Device type (same as mt_pilot_vib_device)
- Additional sensor-specific columns (accel_x, eda_us, ibi, skin_temp, etc.)

### Log Files
Detailed processing logs including:
- Birth date calculation and verification
- Interview age calculation details
- Sample calculations showing age progression
- Cache usage information
- Processing statistics

## Key Features

### 1. Automatic Participant Discovery and Selection
- **Auto-discovery**: Automatically finds all SP* folders in current directory
- **Flexible selection**: Multiple selection methods:
  - Process all participants: `-1` or `all`
  - Single participant: `1` (for SP1)
  - Multiple participants: `1,3,5` (for SP1, SP3, SP5)
  - Range selection: `1-5` (for SP1 through SP5)
  - Mixed selection: `1,3-5,7` (combines individual and range)
- **Smart sorting**: Participants sorted numerically (SP1, SP2, SP10, SP100)
- **Batch processing**: Processes selected participants sequentially
- **Individual logging**: Each participant gets detailed processing logs

### 2. Interview Age Calculation
- **Per-row calculation**: Each timestamp gets its own interview_age
- **Birth date inference**: Calculated from visit_converted.csv data
- **Birth date verification**: Validates against all interview entries
- **Caching**: Birth date calculated once per participant, reused across files
- **Age progression**: Shows how age changes throughout the experiment

### 3. Device Type Assignment
- **Automatic device detection**: Based on participant ID ranges
- **Empatica E4**: Participants 1-83 get "Empatica E4" for both `mt_pilot_vib_device` and `watch_device_type`
- **Other devices**: Participants 84+ get empty values (can be customized for other device types)
- **Consistent assignment**: All data entries for a participant get the same device type

### Example Participant Selection
```
Discovering available participants in ../final_processed_study_parquet...

Found 68 participants:
============================================================
 1. SP1      2. SP2      3. SP3      4. SP5    
 5. SP6     6. SP11     7. SP12     8. SP15   
 9. SP16    10. SP21    11. SP22    12. SP24   
13. SP27    14. SP28    15. SP30    16. SP31   
17. SP32    18. SP33    19. SP34    20. SP36   
21. SP100   22. SP104   23. SP108   24. SP111  
25. SP114   26. SP119   27. SP121
...
============================================================

Selection options:
  -1 or 'all': Process ALL participants
  Single number: Process one participant (e.g., 1)
  Multiple numbers: Process multiple participants (e.g., 1,3,5)
  Range: Process range of participants (e.g., 1-5)
  Mixed: Combine options (e.g., 1,3-5,7)

Select participants to process (1-68, -1 for all): 1,25,26

Selected 3 participants:
  - SP1
  - SP114
  - SP119

Proceed with this selection? (y/n, default: y): y
```

### Example Log Output
```
Found 13 interview entries for subject 1:
  Date: 2020-01-28, Age: 294 months
  Date: 2020-02-04, Age: 294 months
  ...
Initial inferred birth date for subject 1: 1995-07-30

Verifying birth date against all interview entries:
  ✓ 2020-01-28: Reported=294, Calculated=294, Deviation=0 months
  ...
✓ Birth date verification PASSED: 1995-07-30

Device type for participant 1: Empatica E4

Calculated interview_age for 1,000,000 entries: range 294-296 months
Sample age calculations:
  2020-01-28: 294 months
  2020-01-29: 294 months
  2020-01-30: 294 months
  ...
  2020-03-28: 296 months
  2020-03-29: 296 months
  2020-03-30: 296 months
```

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure `./data/metadata.csv` and `./data/visit_converted.csv` exist
2. **Participant folder not found**: Ensure `../final_processed_study_parquet/` directory exists and contains SP* folders
3. **Subject ID mismatch**: Check that subject IDs match between files
4. **Parquet file naming**: Verify naming convention matches expected format (SP[XXX]_e4_[TYPE]_[SIDE].parquet)
5. **Missing timestamps**: Ensure parquet files contain `timestamp_local` column
6. **Birth date verification fails**: Check interview data quality in visit_converted.csv
7. **Logs directory**: The `./logs/` directory will be created automatically if it doesn't exist

### Performance Notes

- **Single mode**: Good for small datasets or testing
- **CPU mode**: Use for multiple files when you have multiple CPU cores
- **GPU mode**: Requires CUDA libraries (not available on macOS)

## Migration from Previous Versions

If migrating from an environment with different directory structure:
1. Ensure your working directory is the project root (where main_script.py is located)
2. Create the required directory structure:
   - `./data/` containing `metadata.csv` and `visit_converted.csv`
   - `../final_processed_study_parquet/` containing all SP* participant folders
3. The `./logs/` directory will be created automatically
4. Update any custom paths in your environment to match the new structure

### Quick Setup Commands
```bash
# Create required directories
mkdir -p ./data
mkdir -p ./logs

# Move data files to correct locations (if needed)
# mv metadata.csv ./data/
# mv visit_converted.csv ./data/

# Verify participant folders exist
ls ../final_processed_study_parquet/
```