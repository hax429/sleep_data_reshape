"""
Enhanced Logging System for Parquet to CSV Transform Process

This module provides comprehensive logging functionality for batch processing
of parquet files with participant tracking and multiple output formats.

Features:
- Dual output (console + file)
- Log levels (DEBUG, INFO, WARN, ERROR)
- Participant ID context tracking
- Timestamp formatting
- Section headers for process organization
- Batch processing friendly output format

Author: Transform Script
Date: 2024
"""

import logging
import sys
import os
from datetime import datetime


class ProcessLogger:
    """
    Enhanced logging system for batch processing with participant tracking.
    
    Provides structured logging with both console and file output, including
    participant context for easy searching and analysis of batch processes.
    
    Attributes:
        participant_id (str): Current participant being processed
        log_file (str): Path to the log file
        logger (logging.Logger): Python logger instance
    """
    
    def __init__(self, log_file=None, participant_id=None):
        """
        Initialize the process logger.
        
        Args:
            log_file (str, optional): Path to log file. If None, only console logging.
            participant_id (str, optional): Initial participant ID for context.
        """
        self.participant_id = participant_id
        self.log_file = log_file
        self.logger = logging.getLogger(f'transform_logger_{os.getpid()}')
        
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # Set logger level to capture all messages
        self.logger.setLevel(logging.DEBUG)
        
        # Setup formatters for different outputs
        self._setup_formatters()
        
        # Setup handlers
        self._setup_console_handler()
        if log_file:
            self._setup_file_handler()
    
    def _setup_formatters(self):
        """Setup log formatters for console and file output."""
        # Console formatter: cleaner output for real-time monitoring
        self.console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File formatter: includes participant ID for batch processing analysis
        self.file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(participant_id)s | %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _setup_console_handler(self):
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Only show INFO and above on console
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Setup file logging handler."""
        # Ensure log directory exists
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create file handler with append mode for batch processing
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)
    
    def _log(self, level, message):
        """
        Internal logging method with participant ID context.
        
        Args:
            level (int): Logging level (DEBUG, INFO, WARNING, ERROR)
            message (str): Log message
        """
        extra = {'participant_id': self.participant_id or 'SYSTEM'}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message):
        """
        Log debug message (file only).
        
        Args:
            message (str): Debug message
        """
        self._log(logging.DEBUG, message)
    
    def info(self, message):
        """
        Log info message (console + file).
        
        Args:
            message (str): Info message
        """
        self._log(logging.INFO, message)
    
    def warning(self, message):
        """
        Log warning message (console + file).
        
        Args:
            message (str): Warning message
        """
        self._log(logging.WARNING, message)
    
    def error(self, message):
        """
        Log error message (console + file).
        
        Args:
            message (str): Error message
        """
        self._log(logging.ERROR, message)
    
    def set_participant_id(self, participant_id):
        """
        Update participant ID for logging context.
        
        Args:
            participant_id (str): New participant ID
        """
        self.participant_id = participant_id
    
    def section_header(self, title):
        """
        Log a section header for better process organization.
        
        Args:
            title (str): Section title
        """
        separator = "=" * 60
        self.info(separator)
        self.info(f" {title.upper()}")
        self.info(separator)
    
    def file_progress(self, file_type, current, total):
        """
        Log file processing progress.
        
        Args:
            file_type (str): Type of file being processed (e.g., 'acc', 'eda')
            current (int): Current file number
            total (int): Total number of files
        """
        self.info(f"[{file_type.upper()}] Processing file {current}/{total}")
    
    def close(self):
        """Close all handlers and cleanup."""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


class LoggerManager:
    """
    Manager class for global logger instance.
    
    Provides singleton-like behavior for the process logger to ensure
    consistent logging across all modules.
    """
    
    _instance = None
    
    @classmethod
    def setup(cls, log_file=None, participant_id=None):
        """
        Setup the global logger instance.
        
        Args:
            log_file (str, optional): Path to log file
            participant_id (str, optional): Initial participant ID
            
        Returns:
            ProcessLogger: The configured logger instance
        """
        cls._instance = ProcessLogger(log_file, participant_id)
        return cls._instance
    
    @classmethod
    def get_logger(cls):
        """
        Get the current logger instance.
        
        Returns:
            ProcessLogger or None: Current logger instance
        """
        return cls._instance
    
    @classmethod
    def close(cls):
        """Close the current logger and cleanup."""
        if cls._instance:
            cls._instance.close()
            cls._instance = None


# Convenience functions for backward compatibility
def setup_logging(log_file=None, participant_id=None):
    """
    Initialize the global logging system.
    
    Args:
        log_file (str, optional): Path to log file
        participant_id (str, optional): Initial participant ID
        
    Returns:
        ProcessLogger: The configured logger instance
    """
    return LoggerManager.setup(log_file, participant_id)


def log_info(message):
    """Log info message using global logger."""
    logger = LoggerManager.get_logger()
    if logger:
        logger.info(message)
    else:
        print(message)


def log_warning(message):
    """Log warning message using global logger."""
    logger = LoggerManager.get_logger()
    if logger:
        logger.warning(message)
    else:
        print(f"WARNING: {message}")


def log_error(message):
    """Log error message using global logger."""
    logger = LoggerManager.get_logger()
    if logger:
        logger.error(message)
    else:
        print(f"ERROR: {message}")


def log_debug(message):
    """Log debug message using global logger."""
    logger = LoggerManager.get_logger()
    if logger:
        logger.debug(message)
    else:
        # Debug messages not shown if no logger setup
        pass