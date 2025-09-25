#!/usr/bin/env python3
"""
Excel Reader for Sentiment Analysis System

This module provides functionality to read and parse Excel files containing
survey responses, with robust error handling and data validation.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging


class ExcelReaderError(Exception):
    """Custom exception for Excel reading related errors."""
    pass


class ExcelReader:
    """
    Handles Excel file operations and data extraction for sentiment analysis.
    
    This class provides methods to load survey responses from Excel files,
    validate file structure, and handle various error conditions gracefully.
    """
    
    def __init__(self):
        """Initialize the Excel reader."""
        self.logger = logging.getLogger(__name__)
    
    def load_responses(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and parse Excel data from the specified file.
        
        Args:
            file_path: Path to the Excel file to load
            
        Returns:
            List of dictionaries containing response data with metadata
            
        Raises:
            ExcelReaderError: If file cannot be read or parsed
        """
        try:
            # Validate file exists and is accessible
            self._validate_file_path(file_path)
            
            # Load Excel file
            self.logger.info(f"Loading Excel file: {file_path}")
            df = pd.read_excel(file_path, engine='openpyxl')
            
            # Validate file structure
            self._validate_file_structure(df)
            
            # Process and clean data
            responses = self._process_data(df, file_path)
            
            self.logger.info(f"Successfully loaded {len(responses)} responses from {file_path}")
            return responses
            
        except FileNotFoundError:
            raise ExcelReaderError(f"Excel file not found: {file_path}")
        except PermissionError:
            raise ExcelReaderError(f"Permission denied accessing file: {file_path}")
        except pd.errors.EmptyDataError:
            raise ExcelReaderError(f"Excel file is empty: {file_path}")
        except Exception as e:
            raise ExcelReaderError(f"Failed to load Excel file {file_path}: {str(e)}")
    
    def _validate_file_path(self, file_path: str) -> None:
        """
        Validate that the file path exists and is accessible.
        
        Args:
            file_path: Path to validate
            
        Raises:
            ExcelReaderError: If file is not accessible
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ExcelReaderError(f"File does not exist: {file_path}")
        
        if not path.is_file():
            raise ExcelReaderError(f"Path is not a file: {file_path}")
        
        if not path.suffix.lower() in ['.xlsx', '.xls']:
            raise ExcelReaderError(f"File is not an Excel file: {file_path}")
        
        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                f.read(1)  # Try to read one byte
        except PermissionError:
            raise ExcelReaderError(f"Permission denied reading file: {file_path}")
        except Exception as e:
            raise ExcelReaderError(f"Cannot access file {file_path}: {str(e)}")
    
    def _validate_file_structure(self, df: pd.DataFrame) -> None:
        """
        Validate that the Excel file has the expected structure.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ExcelReaderError: If file structure is invalid
        """
        if df.empty:
            raise ExcelReaderError(
                "Excel file contains no data. Please ensure the file has both headers and data rows."
            )
        
        if len(df.columns) == 0:
            raise ExcelReaderError(
                "Excel file has no columns. Please check that the file is properly formatted."
            )
        
        # Check if we have at least one row of data (excluding header)
        if len(df) == 0:
            raise ExcelReaderError(
                "Excel file has no data rows (only headers). Please add response data to analyze."
            )
        
        # Enhanced validation: Check for suspicious column patterns
        self._validate_column_structure(df)
        
        # Enhanced validation: Check for minimum viable data
        self._validate_data_quality(df)
        
        self.logger.info(f"File structure validated: {len(df)} rows, {len(df.columns)} columns")
    
    def _validate_column_structure(self, df: pd.DataFrame) -> None:
        """
        Validate column structure for common issues.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ExcelReaderError: If column structure has issues
        """
        # Check for unnamed columns (often indicates formatting issues)
        unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
        if unnamed_cols:
            self.logger.warning(f"Found {len(unnamed_cols)} unnamed columns, which may indicate formatting issues")
        
        # Check for duplicate column names
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            raise ExcelReaderError(
                f"Excel file contains duplicate column names: {duplicate_cols}. "
                "Please ensure all columns have unique names."
            )
        
        # Check for extremely wide files (might indicate transposed data)
        if len(df.columns) > 50:
            self.logger.warning(
                f"Excel file has {len(df.columns)} columns, which is unusually wide. "
                "Please verify the data is oriented correctly."
            )
        
        # Check for extremely narrow files (might miss important data)
        if len(df.columns) == 1:
            self.logger.warning(
                "Excel file has only 1 column. This may limit the quality of sentiment analysis. "
                "Consider including additional context columns if available."
            )
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """
        Validate data quality for sentiment analysis suitability.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ExcelReaderError: If data quality is insufficient
        """
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            self.logger.warning(f"Found {empty_rows} completely empty rows that will be skipped")
        
        # Check if most data is missing
        total_cells = len(df) * len(df.columns)
        empty_cells = df.isnull().sum().sum()
        empty_percentage = (empty_cells / total_cells) * 100
        
        if empty_percentage > 80:
            raise ExcelReaderError(
                f"Excel file is {empty_percentage:.1f}% empty. "
                "Please provide a file with more complete data for meaningful analysis."
            )
        elif empty_percentage > 50:
            self.logger.warning(
                f"Excel file is {empty_percentage:.1f}% empty. "
                "Results may be limited due to sparse data."
            )
        
        # Check for minimum text content
        text_columns = self._identify_text_columns(df)
        if not text_columns:
            raise ExcelReaderError(
                "No text columns suitable for sentiment analysis were found. "
                "Please ensure the file contains columns with textual responses."
            )
        
        self.logger.info(f"Identified {len(text_columns)} potential text columns for analysis")
    
    def _identify_text_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identify columns that likely contain text suitable for sentiment analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names that appear to contain text data
        """
        text_columns = []
        
        for col in df.columns:
            # Skip obviously non-text columns
            if self._is_likely_id_or_number_column(col, ""):
                continue
            
            # Check if column contains meaningful text
            non_null_values = df[col].dropna()
            if len(non_null_values) == 0:
                continue
            
            # Sample some values to check if they're text-like
            sample_values = non_null_values.head(min(10, len(non_null_values)))
            text_like_count = 0
            
            for value in sample_values:
                str_value = str(value).strip()
                # Consider it text-like if it has multiple words or is reasonably long
                if len(str_value) > 10 or len(str_value.split()) > 1:
                    text_like_count += 1
            
            # If most sampled values look like text, consider this a text column
            if text_like_count >= len(sample_values) * 0.5:
                text_columns.append(col)
        
        return text_columns
    
    def _process_data(self, df: pd.DataFrame, file_path: str) -> List[Dict[str, Any]]:
        """
        Process and clean the DataFrame data.
        
        Args:
            df: DataFrame to process
            file_path: Original file path for metadata
            
        Returns:
            List of processed response dictionaries
        """
        responses = []
        headers = df.columns.tolist()
        
        self.logger.info(f"Processing data with headers: {headers}")
        
        for index, row in df.iterrows():
            # Create response dictionary with metadata
            response_data = {
                'row_index': index,
                'source_file': file_path,
                'headers': headers,
                'data': {}
            }
            
            # Process each column
            for col in headers:
                cell_value = row[col]
                
                # Handle empty cells and clean data
                cleaned_value = self._clean_cell_value(cell_value)
                response_data['data'][col] = cleaned_value
            
            # Extract text content for analysis
            text_content = self._extract_text_content(response_data['data'])
            response_data['text_content'] = text_content
            
            # Only include responses that have meaningful text content
            if text_content and text_content.strip():
                responses.append(response_data)
            else:
                self.logger.warning(f"Skipping row {index}: no meaningful text content")
        
        if not responses:
            raise ExcelReaderError("No valid responses found in Excel file")
        
        return responses
    
    def _clean_cell_value(self, value: Any) -> Optional[str]:
        """
        Clean and normalize cell values.
        
        Args:
            value: Raw cell value
            
        Returns:
            Cleaned string value or None if empty
        """
        if pd.isna(value):
            return None
        
        if isinstance(value, str):
            # Strip whitespace and normalize
            cleaned = value.strip()
            return cleaned if cleaned else None
        
        # Convert non-string values to string
        return str(value).strip() if str(value).strip() else None
    
    def _extract_text_content(self, data: Dict[str, Any]) -> str:
        """
        Extract meaningful text content from response data.
        
        Args:
            data: Dictionary containing response data
            
        Returns:
            Combined text content for analysis
        """
        text_parts = []
        
        for key, value in data.items():
            if value is not None and isinstance(value, str) and value.strip():
                # Skip obviously non-textual columns (IDs, numbers, etc.)
                if not self._is_likely_id_or_number_column(key, value):
                    text_parts.append(value.strip())
        
        return ' '.join(text_parts)
    
    def _is_likely_id_or_number_column(self, column_name: str, value: str) -> bool:
        """
        Determine if a column is likely an ID or number field rather than text content.
        
        Args:
            column_name: Name of the column
            value: Value in the column
            
        Returns:
            True if likely an ID or number field
        """
        # Check column name patterns - only exact matches or very specific patterns
        id_patterns = ['id', 'number', 'num', 'index', 'seq']
        column_lower = column_name.lower().strip()
        
        # Only flag as ID if column name is exactly one of these patterns or starts with them
        if (column_lower in id_patterns or 
            column_lower.startswith('id_') or 
            column_lower.startswith('num_') or
            column_lower.endswith('_id') or
            column_lower.endswith('_number')):
            return True
        
        # Check if value looks like a simple ID or number (but be more restrictive)
        if (len(value.strip()) <= 6 and 
            value.strip().isdigit() and 
            len(value.strip()) <= 4):  # Only very short numeric values
            return True
        
        return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about an Excel file without fully loading it.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary with file information
            
        Raises:
            ExcelReaderError: If file cannot be accessed
        """
        try:
            self._validate_file_path(file_path)
            
            # Get basic file stats
            path = Path(file_path)
            stat = path.stat()
            
            # Try to read just the header and first few rows
            df_sample = pd.read_excel(file_path, engine='openpyxl', nrows=5)
            
            return {
                'file_path': str(path.absolute()),
                'file_size': stat.st_size,
                'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified_time': stat.st_mtime,
                'columns': df_sample.columns.tolist(),
                'column_count': len(df_sample.columns),
                'sample_rows': len(df_sample),
                'estimated_total_rows': 'unknown'  # Would need full load to determine
            }
            
        except Exception as e:
            raise ExcelReaderError(f"Failed to get file info for {file_path}: {str(e)}")