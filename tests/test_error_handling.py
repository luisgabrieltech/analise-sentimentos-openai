#!/usr/bin/env python3
"""
Comprehensive error handling tests for the sentiment analysis system.

This module tests all error scenarios across the entire system to ensure
robust error handling, user-friendly messages, and proper logging.
"""

import unittest
import tempfile
import os
import json
import logging
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path
import pandas as pd

from src.excel_reader import ExcelReader, ExcelReaderError
from src.openai_client import OpenAIClient, OpenAIError, RateLimitError, AuthenticationError, TimeoutError
from src.config_manager import ConfigurationManager, Configuration, ConfigurationError
from src.sentiment_analyzer import SentimentAnalyzer, SentimentAnalyzerError
from src.report_generator import ReportGenerator
from src.models import SentimentResult, AnalysisResults
import main


class TestExcelReaderErrorHandling(unittest.TestCase):
    """Test comprehensive error handling in ExcelReader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reader = ExcelReader()
        self.temp_dir = tempfile.mkdtemp()
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        import time
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                # On Windows, files might still be locked, wait and try again
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.temp_dir)
                except PermissionError:
                    pass  # Skip cleanup if still locked
        logging.disable(logging.NOTSET)
    
    def test_file_not_found_error_message(self):
        """Test user-friendly error message for missing files."""
        with self.assertRaises(ExcelReaderError) as context:
            self.reader.load_responses("nonexistent_file.xlsx")
        
        error_msg = str(context.exception)
        self.assertIn("does not exist", error_msg)
        self.assertIn("nonexistent_file.xlsx", error_msg)
    
    def test_invalid_file_extension_error(self):
        """Test error handling for invalid file extensions."""
        # Create a text file with wrong extension
        file_path = os.path.join(self.temp_dir, "test.txt")
        with open(file_path, 'w') as f:
            f.write("This is not an Excel file")
        
        with self.assertRaises(ExcelReaderError) as context:
            self.reader.load_responses(file_path)
        
        error_msg = str(context.exception)
        self.assertIn("not an Excel file", error_msg)
    
    def test_empty_file_error_handling(self):
        """Test handling of completely empty Excel files."""
        file_path = os.path.join(self.temp_dir, "empty.xlsx")
        pd.DataFrame().to_excel(file_path, index=False)
        
        with self.assertRaises(ExcelReaderError) as context:
            self.reader.load_responses(file_path)
        
        error_msg = str(context.exception)
        self.assertIn("no data", error_msg.lower())
        self.assertIn("headers and data rows", error_msg)
    
    def test_duplicate_column_names_error(self):
        """Test error handling for duplicate column names by testing validation directly."""
        # Since Excel automatically renames duplicate columns, test the validation method directly
        df = pd.DataFrame({
            'Response': ['This is a good response with meaningful text', 'This is a bad response with meaningful text'],
            'Response_copy': ['This is great feedback with meaningful text', 'This is terrible feedback with meaningful text']
        })
        
        # Force duplicate columns for testing validation
        df.columns = ['Response', 'Response']  # This creates actual duplicates in memory
        
        with self.assertRaises(ExcelReaderError) as context:
            self.reader._validate_column_structure(df)
        
        error_msg = str(context.exception)
        self.assertIn("duplicate column names", error_msg)
        self.assertIn("unique names", error_msg)
    
    def test_mostly_empty_data_error(self):
        """Test error handling for files with mostly empty data."""
        # Create a file that's 90% empty
        data = {}
        for i in range(10):
            data[f'col_{i}'] = [None] * 10
        
        # Add just a tiny bit of data
        data['col_0'][0] = 'some data'
        
        df = pd.DataFrame(data)
        file_path = os.path.join(self.temp_dir, "mostly_empty.xlsx")
        df.to_excel(file_path, index=False)
        
        with self.assertRaises(ExcelReaderError) as context:
            self.reader.load_responses(file_path)
        
        error_msg = str(context.exception)
        self.assertIn("empty", error_msg.lower())
        self.assertIn("complete data", error_msg)
    
    def test_no_text_columns_error(self):
        """Test error handling when no suitable text columns are found."""
        # Create file with only numeric/ID columns
        df = pd.DataFrame({
            'ID': [1, 2, 3],
            'Number': [100, 200, 300],
            'Code': ['A1', 'B2', 'C3']  # Short codes, not text
        })
        
        file_path = os.path.join(self.temp_dir, "no_text.xlsx")
        df.to_excel(file_path, index=False)
        
        with self.assertRaises(ExcelReaderError) as context:
            self.reader.load_responses(file_path)
        
        error_msg = str(context.exception)
        self.assertIn("No text columns", error_msg)
        self.assertIn("textual responses", error_msg)
    
    def test_file_permission_error_handling(self):
        """Test handling of file permission errors."""
        # Skip this test on Windows as permission handling is different
        import platform
        if platform.system() == 'Windows':
            self.skipTest("Permission error testing not reliable on Windows")
        
        # Create a file and make it unreadable (Unix-like systems)
        df = pd.DataFrame({'Response': ['Test response with meaningful text']})
        file_path = os.path.join(self.temp_dir, "permission_test.xlsx")
        df.to_excel(file_path, index=False)
        
        try:
            os.chmod(file_path, 0o000)  # Remove all permissions
            
            with self.assertRaises(ExcelReaderError) as context:
                self.reader.load_responses(file_path)
            
            error_msg = str(context.exception).lower()
            self.assertTrue(
                'permission denied' in error_msg or 
                'access is denied' in error_msg or
                'cannot access' in error_msg
            )
            
        except (OSError, PermissionError):
            self.skipTest("Cannot test permission errors on this system")
        finally:
            try:
                os.chmod(file_path, 0o644)  # Restore permissions for cleanup
            except (OSError, PermissionError):
                pass
    
    @patch('pandas.read_excel')
    def test_pandas_corruption_error(self, mock_read_excel):
        """Test handling of corrupted Excel files."""
        mock_read_excel.side_effect = Exception("Excel file appears to be corrupted")
        
        # Create a dummy file (won't be read due to mock)
        file_path = os.path.join(self.temp_dir, "corrupted.xlsx")
        with open(file_path, 'wb') as f:
            f.write(b"corrupted data")
        
        with self.assertRaises(ExcelReaderError) as context:
            self.reader.load_responses(file_path)
        
        error_msg = str(context.exception)
        self.assertIn("Failed to load Excel file", error_msg)
        self.assertIn("corrupted", error_msg)


class TestOpenAIClientErrorHandling(unittest.TestCase):
    """Test comprehensive error handling in OpenAIClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Configuration(
            openai_api_key="test-key-12345678901234567890",
            openai_model="gpt-3.5-turbo",
            api_timeout=30,
            max_retries=2
        )
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        logging.disable(logging.NOTSET)
    
    @patch('src.openai_client.OpenAI')
    def test_quota_exceeded_error_message(self, mock_openai):
        """Test user-friendly message for quota exceeded errors."""
        # Create a mock error that behaves like RateLimitError
        class MockRateLimitError(Exception):
            def __init__(self, message):
                self.message = message
                super().__init__(message)
            
            def __str__(self):
                return self.message
        
        # Mock quota exceeded error
        mock_client = Mock()
        quota_error = MockRateLimitError("You exceeded your current quota")
        mock_client.chat.completions.create.side_effect = quota_error
        mock_openai.return_value = mock_client
        
        # Patch the openai module to recognize our mock error
        with patch('src.openai_client.openai.RateLimitError', MockRateLimitError):
            client = OpenAIClient(self.config)
            result = client.analyze_sentiment("Test text")
        
        self.assertFalse(result.success)
        error_msg = result.error_message.lower()
        self.assertIn("quota", error_msg)
        self.assertIn("billing", error_msg)
        self.assertIn("https://platform.openai.com/account/billing", result.error_message)
    
    @patch('src.openai_client.OpenAI')
    def test_rate_limit_rpm_error_message(self, mock_openai):
        """Test user-friendly message for requests per minute rate limits."""
        # Create a mock error that behaves like RateLimitError
        class MockRateLimitError(Exception):
            def __init__(self, message):
                self.message = message
                super().__init__(message)
            
            def __str__(self):
                return self.message
        
        mock_client = Mock()
        rpm_error = MockRateLimitError("Rate limit reached for requests per minute")
        mock_client.chat.completions.create.side_effect = rpm_error
        mock_openai.return_value = mock_client
        
        with patch('src.openai_client.openai.RateLimitError', MockRateLimitError):
            client = OpenAIClient(self.config)
            result = client.analyze_sentiment("Test text")
        
        self.assertFalse(result.success)
        error_msg = result.error_message
        self.assertIn("sending requests too quickly", error_msg)
        self.assertIn("automatically retry", error_msg)
    
    @patch('src.openai_client.OpenAI')
    def test_invalid_api_key_error_message(self, mock_openai):
        """Test user-friendly message for invalid API key."""
        # Create a mock error that behaves like AuthenticationError
        class MockAuthenticationError(Exception):
            def __init__(self, message):
                self.message = message
                super().__init__(message)
            
            def __str__(self):
                return self.message
        
        mock_client = Mock()
        auth_error = MockAuthenticationError("Incorrect API key provided")
        mock_client.chat.completions.create.side_effect = auth_error
        mock_openai.return_value = mock_client
        
        with patch('src.openai_client.openai.AuthenticationError', MockAuthenticationError):
            client = OpenAIClient(self.config)
            result = client.analyze_sentiment("Test text")
        
        self.assertFalse(result.success)
        error_msg = result.error_message
        self.assertIn("Invalid OpenAI API key", error_msg)
        self.assertIn("https://platform.openai.com/api-keys", error_msg)
    
    @patch('src.openai_client.OpenAI')
    def test_model_not_found_error_message(self, mock_openai):
        """Test user-friendly message for invalid model."""
        # Create a mock error that behaves like APIError
        class MockAPIError(Exception):
            def __init__(self, message):
                self.message = message
                super().__init__(message)
            
            def __str__(self):
                return self.message
        
        mock_client = Mock()
        model_error = MockAPIError("The model 'invalid-model' does not exist")
        mock_client.chat.completions.create.side_effect = model_error
        mock_openai.return_value = mock_client
        
        with patch('src.openai_client.openai.APIError', MockAPIError):
            client = OpenAIClient(self.config)
            result = client.analyze_sentiment("Test text")
        
        self.assertFalse(result.success)
        error_msg = result.error_message
        self.assertIn("model", error_msg.lower())
        self.assertIn("not available", error_msg)
        self.assertIn("gpt-3.5-turbo", error_msg)
    
    @patch('src.openai_client.OpenAI')
    def test_timeout_error_message(self, mock_openai):
        """Test user-friendly message for timeout errors."""
        import openai
        
        mock_client = Mock()
        timeout_error = openai.APITimeoutError("Request timed out")
        mock_client.chat.completions.create.side_effect = timeout_error
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("Test text")
        
        self.assertFalse(result.success)
        error_msg = result.error_message
        self.assertIn("timed out", error_msg)
        self.assertIn("network issues", error_msg)
        self.assertIn("automatically retry", error_msg)
    
    @patch('src.openai_client.OpenAI')
    def test_connection_error_message(self, mock_openai):
        """Test user-friendly message for connection errors."""
        # Create a mock error that behaves like APIConnectionError
        class MockConnectionError(Exception):
            def __init__(self, message):
                self.message = message
                super().__init__(message)
            
            def __str__(self):
                return self.message
        
        mock_client = Mock()
        conn_error = MockConnectionError("Connection failed")
        mock_client.chat.completions.create.side_effect = conn_error
        mock_openai.return_value = mock_client
        
        with patch('src.openai_client.openai.APIConnectionError', MockConnectionError):
            client = OpenAIClient(self.config)
            result = client.analyze_sentiment("Test text")
        
        self.assertFalse(result.success)
        error_msg = result.error_message
        self.assertIn("Failed to connect", error_msg)
        self.assertIn("network issues", error_msg)
        self.assertIn("internet connection", error_msg)


class TestConfigurationErrorHandling(unittest.TestCase):
    """Test comprehensive error handling in ConfigurationManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Store original environment variables
        self.original_env = {}
        env_vars = ["OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_MODEL", "API_TIMEOUT", "MAX_RETRIES"]
        for var in env_vars:
            if var in os.environ:
                self.original_env[var] = os.environ[var]
                del os.environ[var]
        
        self.config_manager = ConfigurationManager()
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clear environment variables
        env_vars = ["OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_MODEL", "API_TIMEOUT", "MAX_RETRIES"]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
        
        # Restore original environment variables
        for var, value in self.original_env.items():
            os.environ[var] = value
        
        logging.disable(logging.NOTSET)
    
    def test_missing_api_key_helpful_message(self):
        """Test helpful error message when API key is missing."""
        # Create a config manager that doesn't load the real .env file
        config_manager = ConfigurationManager(env_file="nonexistent.env")
        
        with self.assertRaises(ConfigurationError) as context:
            config_manager.load_configuration()
        
        error_msg = str(context.exception)
        self.assertIn("OpenAI API key is required", error_msg)
        self.assertIn("OPENAI_API_KEY", error_msg)
        self.assertIn(".env file", error_msg)
        self.assertIn(".env.example", error_msg)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-short"})
    def test_invalid_api_key_format_message(self):
        """Test helpful error message for invalid API key format."""
        # Create a config manager that doesn't load the real .env file
        config_manager = ConfigurationManager(env_file="nonexistent.env")
        
        with self.assertRaises(ConfigurationError) as context:
            config_manager.load_configuration()
        
        error_msg = str(context.exception)
        self.assertIn("API key appears to be invalid", error_msg)
        self.assertIn("too short", error_msg)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "your_openai_api_key_here"})
    def test_placeholder_api_key_message(self):
        """Test helpful error message for placeholder API key."""
        # Create a config manager that doesn't load the real .env file
        config_manager = ConfigurationManager(env_file="nonexistent.env")
        
        with self.assertRaises(ConfigurationError) as context:
            config_manager.load_configuration()
        
        error_msg = str(context.exception)
        self.assertIn("replace the placeholder", error_msg)
        self.assertIn("https://platform.openai.com/api-keys", error_msg)
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
        "API_TIMEOUT": "not_a_number"
    })
    def test_invalid_timeout_format_message(self):
        """Test helpful error message for invalid timeout format."""
        # Create a config manager that doesn't load the real .env file
        config_manager = ConfigurationManager(env_file="nonexistent.env")
        
        with self.assertRaises(ConfigurationError) as context:
            config_manager.load_configuration()
        
        error_msg = str(context.exception)
        self.assertIn("Invalid value for API_TIMEOUT", error_msg)
        self.assertIn("positive integer", error_msg)
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
        "API_TIMEOUT": "1"
    })
    def test_timeout_out_of_range_message(self):
        """Test helpful error message for timeout out of valid range."""
        # Create a config manager that doesn't load the real .env file
        config_manager = ConfigurationManager(env_file="nonexistent.env")
        
        with self.assertRaises(ConfigurationError) as context:
            config_manager.load_configuration()
        
        error_msg = str(context.exception)
        self.assertIn("API timeout must be between 5 and 300", error_msg)


class TestMainApplicationErrorHandling(unittest.TestCase):
    """Test comprehensive error handling in main application."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        import time
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                # On Windows, files might still be locked, wait and try again
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.temp_dir)
                except PermissionError:
                    pass  # Skip cleanup if still locked
        logging.disable(logging.NOTSET)
    
    def test_validate_input_file_missing_suggestions(self):
        """Test that file validation provides helpful suggestions."""
        # Store original directory
        original_dir = os.getcwd()
        
        try:
            # Create some Excel files in temp directory
            os.chdir(self.temp_dir)
            
            # Create sample Excel files
            sample1_path = os.path.join(self.temp_dir, 'sample1.xlsx')
            sample2_path = os.path.join(self.temp_dir, 'sample2.xlsx')
            
            pd.DataFrame({'col': [1]}).to_excel(sample1_path, index=False)
            pd.DataFrame({'col': [2]}).to_excel(sample2_path, index=False)
            
            # Capture output
            with patch('builtins.print') as mock_print:
                result = main.validate_input_file('nonexistent.xlsx')
            
            self.assertFalse(result)
            
            # Check that helpful information was printed
            printed_text = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
            self.assertIn("does not exist", printed_text)
            self.assertIn("Current directory", printed_text)
            self.assertIn("Available Excel files", printed_text)
            self.assertIn("sample1.xlsx", printed_text)
            self.assertIn("sample2.xlsx", printed_text)
            
        finally:
            # Always restore original directory
            os.chdir(original_dir)
    
    def test_validate_input_file_large_file_warning(self):
        """Test warning for very large files."""
        # Create a large dummy file
        large_file = os.path.join(self.temp_dir, 'large.xlsx')
        with open(large_file, 'wb') as f:
            f.write(b'0' * (101 * 1024 * 1024))  # 101MB
        
        with patch('builtins.input', return_value='n'):  # User says no
            with patch('builtins.print') as mock_print:
                result = main.validate_input_file(large_file)
        
        self.assertFalse(result)
        
        printed_text = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("very large", printed_text)
        self.assertIn("101.0MB", printed_text)
        # The "Continue anyway" text is in the input prompt, not print output
        self.assertIn("Processing may take a long time", printed_text)
    
    @patch('main.validate_input_file')
    @patch('main.ConfigurationManager')
    def test_configuration_error_suggestions(self, mock_config_manager, mock_validate):
        """Test that configuration errors provide helpful suggestions."""
        # Mock file validation to pass
        mock_validate.return_value = True
        
        # Mock configuration error
        mock_manager = Mock()
        mock_manager.load_configuration.side_effect = ConfigurationError("OpenAI API key is required")
        mock_manager.display_setup_instructions = Mock()
        mock_config_manager.return_value = mock_manager
        
        # Mock args
        args = Mock()
        args.file = 'test.xlsx'
        args.verbose = False
        args.log_file = None
        args.validate_only = False
        args.max_responses = None
        args.output_dir = 'output'
        args.no_save = False
        args.no_samples = False
        
        with patch('builtins.print') as mock_print:
            result = main.main(args)
        
        self.assertEqual(result, 1)
        
        # Check that configuration error was handled
        printed_text = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("Configuration Error", printed_text)


class TestSentimentAnalyzerErrorHandling(unittest.TestCase):
    """Test comprehensive error handling in SentimentAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Configuration(
            openai_api_key="test-key-12345678901234567890",
            openai_model="gpt-3.5-turbo"
        )
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        logging.disable(logging.NOTSET)
    
    @patch('src.sentiment_analyzer.ExcelReader')
    def test_excel_reader_error_propagation(self, mock_excel_reader):
        """Test that ExcelReader errors are properly handled and propagated."""
        mock_reader = Mock()
        mock_reader.load_responses.side_effect = ExcelReaderError("Test Excel error")
        mock_excel_reader.return_value = mock_reader
        
        analyzer = SentimentAnalyzer(self.config)
        
        with self.assertRaises(SentimentAnalyzerError) as context:
            analyzer.process_responses("test.xlsx")
        
        error_msg = str(context.exception)
        self.assertIn("Failed to load Excel file", error_msg)
        self.assertIn("Test Excel error", error_msg)
    
    @patch('src.sentiment_analyzer.ExcelReader')
    @patch('src.sentiment_analyzer.OpenAIClient')
    def test_individual_response_error_handling(self, mock_openai_client, mock_excel_reader):
        """Test that individual response errors don't crash the entire process."""
        # Mock Excel reader to return sample responses
        mock_reader = Mock()
        mock_reader.load_responses.return_value = [
            {'text_content': 'Good response', 'headers': ['Response'], 'row_index': 0},
            {'text_content': 'Bad response', 'headers': ['Response'], 'row_index': 1}
        ]
        mock_excel_reader.return_value = mock_reader
        
        # Mock OpenAI client to fail on second response
        mock_client = Mock()
        mock_client.analyze_sentiment.side_effect = [
            SentimentResult("Good response", "positive", 0.8, "Good", 1.0, True),
            Exception("API error")
        ]
        mock_openai_client.return_value = mock_client
        
        analyzer = SentimentAnalyzer(self.config)
        results = analyzer.process_responses("test.xlsx")
        
        # Should have 2 results, one successful, one failed
        self.assertEqual(len(results.individual_results), 2)
        self.assertTrue(results.individual_results[0].success)
        self.assertFalse(results.individual_results[1].success)
        self.assertIn("Unexpected error", results.individual_results[1].error_message)
    
    def test_generate_insights_no_results_error(self):
        """Test error handling when trying to generate insights without results."""
        analyzer = SentimentAnalyzer(self.config)
        
        with self.assertRaises(SentimentAnalyzerError) as context:
            analyzer.generate_insights()
        
        error_msg = str(context.exception)
        self.assertIn("No analysis results available", error_msg)


class TestReportGeneratorErrorHandling(unittest.TestCase):
    """Test comprehensive error handling in ReportGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        import time
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                # On Windows, files might still be locked, wait and try again
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.temp_dir)
                except PermissionError:
                    pass  # Skip cleanup if still locked
        logging.disable(logging.NOTSET)
    
    def test_invalid_output_directory_handling(self):
        """Test handling of invalid output directories."""
        # Try to create report generator with invalid directory
        invalid_path = os.path.join(self.temp_dir, "nonexistent", "deeply", "nested")
        
        # Should create the directory automatically
        generator = ReportGenerator(invalid_path)
        self.assertTrue(os.path.exists(invalid_path))
    
    def test_file_write_permission_error(self):
        """Test handling of file write permission errors."""
        import platform
        if platform.system() == 'Windows':
            self.skipTest("Permission error testing not reliable on Windows")
        
        generator = ReportGenerator(self.temp_dir)
        
        # Create mock results
        results = Mock()
        results.timestamp = Mock()
        results.timestamp.strftime.return_value = "20240101_120000"
        results.to_dict.return_value = {"test": "data"}
        
        # Make directory read-only (Unix-like systems)
        try:
            os.chmod(self.temp_dir, 0o444)  # Read-only
            
            with self.assertRaises(IOError) as context:
                generator.save_detailed_results(results)
            
            error_msg = str(context.exception)
            self.assertIn("Failed to save results", error_msg)
            
        except (OSError, PermissionError):
            self.skipTest("Cannot test permission errors on this system")
        finally:
            try:
                os.chmod(self.temp_dir, 0o755)  # Restore permissions
            except (OSError, PermissionError):
                pass
    
    def test_empty_results_report_generation(self):
        """Test report generation with empty results."""
        generator = ReportGenerator(self.temp_dir)
        
        # Create results with no individual results
        results = Mock()
        results.individual_results = []
        results.timestamp = Mock()
        results.timestamp.strftime.return_value = "2024-01-01 12:00:00"
        results.total_processed = 0
        results.success_rate = 0.0
        
        report = generator.create_summary_report(results)
        
        self.assertIn("No results available", report)
        self.assertIn("check your input data", report)


class TestLoggingAndMonitoring(unittest.TestCase):
    """Test logging and monitoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Close all logging handlers to release file locks
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        
        # Reset logging
        logging.basicConfig()
        
        # Clean up temp directory
        import shutil
        import time
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                # On Windows, files might still be locked, wait and try again
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.temp_dir)
                except PermissionError:
                    pass  # Skip cleanup if still locked
    
    def test_file_logging_setup(self):
        """Test that file logging is set up correctly."""
        log_file = os.path.join(self.temp_dir, "test.log")
        
        main.setup_logging(verbose=True, log_file=log_file)
        
        # Log a test message
        logger = logging.getLogger("test")
        logger.info("Test message")
        
        # Check that log file was created and contains the message
        self.assertTrue(os.path.exists(log_file))
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        self.assertIn("Test message", log_content)
        self.assertIn("INFO", log_content)
    
    def test_log_file_directory_creation(self):
        """Test that log file directories are created automatically."""
        log_file = os.path.join(self.temp_dir, "logs", "subdir", "test.log")
        
        main.setup_logging(verbose=False, log_file=log_file)
        
        # Log a message to trigger file creation
        logger = logging.getLogger("test")
        logger.info("Test message")
        
        # Check that directory structure was created
        self.assertTrue(os.path.exists(os.path.dirname(log_file)))
        self.assertTrue(os.path.exists(log_file))
    
    def test_logging_error_handling(self):
        """Test that logging errors are handled gracefully."""
        # Try to log to an invalid location
        invalid_log_file = "/invalid/path/test.log"
        
        # Should not raise an exception
        with patch('builtins.print') as mock_print:
            main.setup_logging(verbose=False, log_file=invalid_log_file)
        
        # Should have logged a warning about file logging failure
        logger = logging.getLogger("test")
        logger.info("Test message")  # Should still work with console logging


if __name__ == '__main__':
    # Run with high verbosity to see all test details
    unittest.main(verbosity=2)