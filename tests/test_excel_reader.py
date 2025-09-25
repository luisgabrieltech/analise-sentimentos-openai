#!/usr/bin/env python3
"""
Unit tests for Excel Reader module.

This module contains comprehensive tests for the ExcelReader class,
including tests with sample Excel files and various error conditions.
"""

import unittest
import tempfile
import os
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

from src.excel_reader import ExcelReader, ExcelReaderError


class TestExcelReader(unittest.TestCase):
    """Test cases for ExcelReader class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.reader = ExcelReader()
        self.temp_dir = tempfile.mkdtemp()
        
        # Suppress logging during tests
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up after each test method."""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    def _create_sample_excel(self, filename: str, data: dict, empty: bool = False) -> str:
        """
        Create a sample Excel file for testing.
        
        Args:
            filename: Name of the file to create
            data: Dictionary with data to write
            empty: Whether to create an empty file
            
        Returns:
            Path to the created file
        """
        file_path = os.path.join(self.temp_dir, filename)
        
        if empty:
            # Create empty Excel file
            pd.DataFrame().to_excel(file_path, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_excel(file_path, index=False)
        
        return file_path
    
    def test_load_responses_success(self):
        """Test successful loading of responses from Excel file."""
        # Create sample data
        sample_data = {
            'ID': [1, 2, 3],
            'Question': ['How was your experience?', 'Any suggestions?', 'Overall rating?'],
            'Response': ['Great service!', 'Could be better', 'Excellent work'],
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03']
        }
        
        file_path = self._create_sample_excel('test_responses.xlsx', sample_data)
        
        # Load responses
        responses = self.reader.load_responses(file_path)
        
        # Verify results
        self.assertEqual(len(responses), 3)
        
        # Check first response structure
        first_response = responses[0]
        self.assertIn('row_index', first_response)
        self.assertIn('source_file', first_response)
        self.assertIn('headers', first_response)
        self.assertIn('data', first_response)
        self.assertIn('text_content', first_response)
        
        # Check data content
        self.assertEqual(first_response['row_index'], 0)
        self.assertEqual(first_response['source_file'], file_path)
        self.assertEqual(len(first_response['headers']), 4)
        self.assertIn('Great service!', first_response['text_content'])
    
    def test_load_responses_with_empty_cells(self):
        """Test loading responses with empty cells."""
        sample_data = {
            'ID': [1, 2, 3],
            'Response': ['Good', None, 'Bad'],
            'Comment': [None, 'Some comment', '']
        }
        
        file_path = self._create_sample_excel('test_empty_cells.xlsx', sample_data)
        
        responses = self.reader.load_responses(file_path)
        
        # Should have 2 responses (one with None response should be filtered out if no text)
        self.assertGreaterEqual(len(responses), 1)
        
        # Check that empty values are handled properly
        for response in responses:
            self.assertIsInstance(response['text_content'], str)
    
    def test_load_responses_file_not_found(self):
        """Test error handling when file doesn't exist."""
        with self.assertRaises(ExcelReaderError) as context:
            self.reader.load_responses('nonexistent_file.xlsx')
        
        self.assertIn('does not exist', str(context.exception).lower())
    
    def test_load_responses_empty_file(self):
        """Test error handling with empty Excel file."""
        file_path = self._create_sample_excel('empty.xlsx', {}, empty=True)
        
        with self.assertRaises(ExcelReaderError) as context:
            self.reader.load_responses(file_path)
        
        self.assertIn('no data', str(context.exception).lower())
    
    def test_load_responses_invalid_file_extension(self):
        """Test error handling with invalid file extension."""
        # Create a text file with .txt extension
        file_path = os.path.join(self.temp_dir, 'test.txt')
        with open(file_path, 'w') as f:
            f.write('This is not an Excel file')
        
        with self.assertRaises(ExcelReaderError) as context:
            self.reader.load_responses(file_path)
        
        self.assertIn('not an Excel file', str(context.exception))
    
    def test_validate_file_path_success(self):
        """Test successful file path validation."""
        sample_data = {'col1': [1, 2], 'col2': ['a', 'b']}
        file_path = self._create_sample_excel('valid.xlsx', sample_data)
        
        # Should not raise exception
        self.reader._validate_file_path(file_path)
    
    def test_validate_file_path_nonexistent(self):
        """Test file path validation with nonexistent file."""
        with self.assertRaises(ExcelReaderError):
            self.reader._validate_file_path('nonexistent.xlsx')
    
    def test_validate_file_structure_success(self):
        """Test successful file structure validation."""
        df = pd.DataFrame({
            'ID': [1, 2, 3], 
            'Response': ['This is a meaningful response', 'Another good response', 'Great feedback here']
        })
        
        # Should not raise exception
        self.reader._validate_file_structure(df)
    
    def test_validate_file_structure_empty_dataframe(self):
        """Test file structure validation with empty DataFrame."""
        df = pd.DataFrame()
        
        with self.assertRaises(ExcelReaderError) as context:
            self.reader._validate_file_structure(df)
        
        self.assertIn('no data', str(context.exception).lower())
    
    def test_validate_file_structure_no_data_rows(self):
        """Test file structure validation with no data rows."""
        # Create DataFrame with columns but no data
        df = pd.DataFrame(columns=['col1', 'col2'])
        
        with self.assertRaises(ExcelReaderError) as context:
            self.reader._validate_file_structure(df)
        
        self.assertIn('no data', str(context.exception).lower())
    
    def test_clean_cell_value_string(self):
        """Test cell value cleaning with string input."""
        # Test normal string
        result = self.reader._clean_cell_value('  Hello World  ')
        self.assertEqual(result, 'Hello World')
        
        # Test empty string
        result = self.reader._clean_cell_value('   ')
        self.assertIsNone(result)
        
        # Test None/NaN
        result = self.reader._clean_cell_value(None)
        self.assertIsNone(result)
        
        result = self.reader._clean_cell_value(pd.NA)
        self.assertIsNone(result)
    
    def test_clean_cell_value_numbers(self):
        """Test cell value cleaning with numeric input."""
        # Test integer
        result = self.reader._clean_cell_value(123)
        self.assertEqual(result, '123')
        
        # Test float
        result = self.reader._clean_cell_value(123.45)
        self.assertEqual(result, '123.45')
    
    def test_extract_text_content(self):
        """Test text content extraction from response data."""
        data = {
            'ID': '1',
            'Name': 'John Doe',
            'Response': 'This is a great product!',
            'Rating': '5',
            'Comment': 'Highly recommended'
        }
        
        text_content = self.reader._extract_text_content(data)
        
        # Should contain meaningful text but exclude likely ID fields
        self.assertIn('John Doe', text_content)
        self.assertIn('This is a great product!', text_content)
        self.assertIn('Highly recommended', text_content)
        # ID might be excluded as it's likely an ID field
    
    def test_is_likely_id_or_number_column(self):
        """Test identification of ID or number columns."""
        # Test ID-like column names
        self.assertTrue(self.reader._is_likely_id_or_number_column('ID', '123'))
        self.assertTrue(self.reader._is_likely_id_or_number_column('user_id', '456'))
        self.assertTrue(self.reader._is_likely_id_or_number_column('Number', '789'))
        
        # Test non-ID column names with text content
        self.assertFalse(self.reader._is_likely_id_or_number_column('Response', 'This is a long response'))
        self.assertFalse(self.reader._is_likely_id_or_number_column('Comment', 'Great service'))
        
        # Test numeric values in non-ID columns
        self.assertFalse(self.reader._is_likely_id_or_number_column('Response', 'I rate this 5 stars'))
    
    def test_get_file_info_success(self):
        """Test getting file information."""
        sample_data = {
            'Question': ['Q1', 'Q2', 'Q3'],
            'Response': ['Answer 1', 'Answer 2', 'Answer 3']
        }
        file_path = self._create_sample_excel('info_test.xlsx', sample_data)
        
        info = self.reader.get_file_info(file_path)
        
        # Check required fields
        self.assertIn('file_path', info)
        self.assertIn('file_size', info)
        self.assertIn('file_size_mb', info)
        self.assertIn('columns', info)
        self.assertIn('column_count', info)
        
        # Check values
        self.assertEqual(info['column_count'], 2)
        self.assertEqual(info['columns'], ['Question', 'Response'])
        self.assertGreater(info['file_size'], 0)
    
    def test_get_file_info_nonexistent_file(self):
        """Test getting file information for nonexistent file."""
        with self.assertRaises(ExcelReaderError):
            self.reader.get_file_info('nonexistent.xlsx')
    
    @patch('pandas.read_excel')
    def test_load_responses_pandas_error(self, mock_read_excel):
        """Test error handling when pandas fails to read Excel."""
        mock_read_excel.side_effect = Exception('Pandas read error')
        
        # Create a dummy file (won't be read due to mock)
        file_path = self._create_sample_excel('dummy.xlsx', {'col': [1]})
        
        with self.assertRaises(ExcelReaderError) as context:
            self.reader.load_responses(file_path)
        
        self.assertIn('Failed to load Excel file', str(context.exception))
    
    def test_load_responses_permission_error(self):
        """Test error handling for permission errors."""
        # Create a file and then make it unreadable (Unix-like systems)
        sample_data = {'col': [1, 2]}
        file_path = self._create_sample_excel('permission_test.xlsx', sample_data)
        
        # Try to make file unreadable (this might not work on all systems)
        try:
            os.chmod(file_path, 0o000)
            
            with self.assertRaises(ExcelReaderError) as context:
                self.reader.load_responses(file_path)
            
            # On Windows, permission errors might manifest differently
            error_msg = str(context.exception).lower()
            self.assertTrue(
                'permission denied' in error_msg or 
                'access is denied' in error_msg or
                'failed to load' in error_msg
            )
        except (OSError, PermissionError):
            # Skip this test if we can't change permissions
            self.skipTest("Cannot test permission errors on this system")
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(file_path, 0o644)
            except (OSError, PermissionError):
                pass
    
    def test_process_data_filters_empty_responses(self):
        """Test that responses with no meaningful text content are filtered out."""
        # Create DataFrame with some empty/meaningless responses
        df = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'Response': ['Good service', '', None, '   '],
            'Rating': [5, 3, 4, 2]
        })
        
        responses = self.reader._process_data(df, 'test.xlsx')
        
        # Should only have 1 response (the one with "Good service")
        self.assertEqual(len(responses), 1)
        self.assertIn('Good service', responses[0]['text_content'])


class TestExcelReaderIntegration(unittest.TestCase):
    """Integration tests for ExcelReader with real Excel files."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reader = ExcelReader()
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up after tests."""
        logging.disable(logging.NOTSET)
    
    def test_load_real_excel_file_if_exists(self):
        """Test loading real Excel files if they exist in the project."""
        # Test with respostas.xlsx if it exists
        if os.path.exists('respostas.xlsx'):
            try:
                responses = self.reader.load_responses('respostas.xlsx')
                self.assertIsInstance(responses, list)
                self.assertGreater(len(responses), 0)
                
                # Check structure of first response
                if responses:
                    first_response = responses[0]
                    self.assertIn('text_content', first_response)
                    self.assertIn('data', first_response)
                    self.assertIn('headers', first_response)
                    
            except ExcelReaderError as e:
                self.fail(f"Failed to load respostas.xlsx: {e}")
        
        # Test with 100.xlsx if it exists
        if os.path.exists('100.xlsx'):
            try:
                responses = self.reader.load_responses('100.xlsx')
                self.assertIsInstance(responses, list)
                
            except ExcelReaderError as e:
                self.fail(f"Failed to load 100.xlsx: {e}")
    
    def test_get_file_info_real_files(self):
        """Test getting file info for real Excel files."""
        for filename in ['respostas.xlsx', '100.xlsx']:
            if os.path.exists(filename):
                try:
                    info = self.reader.get_file_info(filename)
                    self.assertIsInstance(info, dict)
                    self.assertIn('columns', info)
                    self.assertIn('file_size', info)
                    
                except ExcelReaderError as e:
                    self.fail(f"Failed to get info for {filename}: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)