#!/usr/bin/env python3
"""
Integration test for the main application flow.

This test verifies that the complete application works end-to-end
with mocked OpenAI API calls and tests all command-line interface features.
"""

import unittest
import tempfile
import os
import pandas as pd
import json
import argparse
from unittest.mock import patch, Mock, MagicMock
from io import StringIO
import sys
import shutil

from src.models import SentimentResult, AnalysisResults, Summary
from src.config_manager import Configuration


class TestMainIntegration(unittest.TestCase):
    """Integration test for the main application."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_excel_path = os.path.join(self.temp_dir, "respostas.xlsx")
        self.output_dir = os.path.join(self.temp_dir, "output")
        
        # Create test Excel file
        self._create_test_excel_file()
        
        # Change to temp directory so main.py can find the file
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original directory
        os.chdir(self.original_cwd)
        
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_excel_file(self):
        """Create a test Excel file with sample responses."""
        data = {
            'ID': [1, 2, 3, 4, 5],
            'Feedback': [
                'I absolutely love this product! It works perfectly and exceeded my expectations.',
                'This service is terrible. I am very disappointed and would not recommend it.',
                'The product is okay, nothing special but it does what it needs to do.',
                'Amazing experience! Highly recommend to everyone.',
                'Could be better, but it serves its purpose.'
            ],
            'Rating': [5, 1, 3, 5, 3]
        }
        df = pd.DataFrame(data)
        df.to_excel(self.test_excel_path, index=False)
    
    def _create_mock_sentiment_result(self, text: str, sentiment: str, confidence: float):
        """Create a mock SentimentResult for testing."""
        return SentimentResult(
            original_text=text,
            sentiment=sentiment,
            confidence=confidence,
            reasoning=f"Mock reasoning for {sentiment} sentiment",
            processing_time=0.5,
            success=True
        )
    
    def _create_mock_args(self, **kwargs):
        """Create mock command-line arguments."""
        defaults = {
            'file': 'respostas.xlsx',
            'output_dir': 'output',
            'no_save': False,
            'no_samples': False,
            'verbose': False,
            'log_file': None,
            'validate_only': False,
            'max_responses': None,
            'async_processing': False,
            'batch_size': 10,
            'memory_optimized': False,
            'chunk_size': 100
        }
        defaults.update(kwargs)
        
        args = argparse.Namespace()
        for key, value in defaults.items():
            setattr(args, key, value)
        return args
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key_12345'})
    @patch('main.ExcelReader')
    @patch('main.ConfigurationManager')
    @patch('main.ReportGenerator')
    @patch('main.SentimentAnalyzer')
    def test_complete_main_flow_with_file_saving(self, mock_analyzer_class, mock_report_generator_class, mock_config_manager_class, mock_excel_reader_class):
        """Test the complete main application flow with file saving."""
        # Mock Excel reader
        mock_excel_reader = Mock()
        mock_excel_reader_class.return_value = mock_excel_reader
        mock_excel_reader.get_file_info.return_value = {
            'file_size_mb': 0.1,
            'column_count': 2,
            'columns': ['ID', 'Feedback']
        }
        mock_excel_reader.load_responses.return_value = [
            {
                'text_content': 'I absolutely love this product! It works perfectly and exceeded my expectations.',
                'headers': ['ID', 'Feedback'],
                'row_index': 0,
                'source_file': 'respostas.xlsx'
            },
            {
                'text_content': 'This service is terrible. I am very disappointed and would not recommend it.',
                'headers': ['ID', 'Feedback'],
                'row_index': 1,
                'source_file': 'respostas.xlsx'
            },
            {
                'text_content': 'The product is okay, nothing special but it does what it needs to do.',
                'headers': ['ID', 'Feedback'],
                'row_index': 2,
                'source_file': 'respostas.xlsx'
            },
            {
                'text_content': 'Amazing experience! Highly recommend to everyone.',
                'headers': ['ID', 'Feedback'],
                'row_index': 3,
                'source_file': 'respostas.xlsx'
            },
            {
                'text_content': 'Could be better, but it serves its purpose.',
                'headers': ['ID', 'Feedback'],
                'row_index': 4,
                'source_file': 'respostas.xlsx'
            }
        ]
        
        # Mock analyzer and its results
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Mock sentiment analysis results
        mock_results = [
            self._create_mock_sentiment_result(
                "I absolutely love this product! It works perfectly and exceeded my expectations.",
                "positive", 0.95
            ),
            self._create_mock_sentiment_result(
                "This service is terrible. I am very disappointed and would not recommend it.",
                "negative", 0.90
            ),
            self._create_mock_sentiment_result(
                "The product is okay, nothing special but it does what it needs to do.",
                "neutral", 0.75
            ),
            self._create_mock_sentiment_result(
                "Amazing experience! Highly recommend to everyone.",
                "positive", 0.92
            ),
            self._create_mock_sentiment_result(
                "Could be better, but it serves its purpose.",
                "neutral", 0.68
            )
        ]
        
        # Create mock analysis results
        from src.models import AnalysisResults, Summary
        from datetime import datetime
        
        mock_analysis_results = AnalysisResults(
            individual_results=mock_results,
            summary_stats={},
            processing_metadata={},
            timestamp=datetime.now(),
            total_processed=5,
            success_rate=1.0
        )
        
        mock_summary = Summary(
            sentiment_distribution={'positive': 2, 'negative': 1, 'neutral': 2},
            sentiment_percentages={'positive': 40.0, 'negative': 20.0, 'neutral': 40.0},
            common_themes=[],
            key_insights=['Test insight'],
            recommendations=['Test recommendation'],
            confidence_stats={'mean': 0.84}
        )
        
        mock_analyzer.process_responses_from_data.return_value = mock_analysis_results
        mock_analyzer.generate_insights.return_value = mock_summary
        
        # Mock report generator
        mock_report_generator = Mock()
        mock_report_generator_class.return_value = mock_report_generator
        mock_report_generator.display_console_summary.return_value = None
        mock_report_generator.save_detailed_results.return_value = os.path.join(self.output_dir, "results.json")
        mock_report_generator.save_summary_report.return_value = os.path.join(self.output_dir, "summary.txt")
        
        # Mock configuration manager
        mock_config_manager = Mock()
        mock_config_manager_class.return_value = mock_config_manager
        mock_config = Configuration(openai_api_key='test_key_12345')
        mock_config_manager.load_configuration.return_value = mock_config
        mock_config_manager.get_safe_config_summary.return_value = {
            'api_key_preview': 'test_key_*****',
            'model': 'gpt-3.5-turbo',
            'timeout': 30.0,
            'max_retries': 3
        }
        
        # Capture stdout to verify output
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            # Import and run main with default arguments
            from main import main
            
            args = self._create_mock_args()
            result = main(args)
            
            # Verify successful execution
            self.assertEqual(result, 0)
            
            # Get the output
            output = captured_output.getvalue()
            
            # Verify key components of output
            self.assertIn("SENTIMENT ANALYSIS SYSTEM", output)
            self.assertIn("Configuration loaded successfully", output)
            self.assertIn("Successfully loaded 5 responses", output)
            self.assertIn("Initializing sentiment analyzer", output)
            self.assertIn("Analysis completed!", output)
            
            # Verify analyzer was called correctly
            mock_analyzer.process_responses_from_data.assert_called_once()
            mock_analyzer.generate_insights.assert_called_once()
            
            # Verify report generator was called
            mock_report_generator.display_console_summary.assert_called_once()
            mock_report_generator.save_detailed_results.assert_called_once()
            mock_report_generator.save_summary_report.assert_called_once()
            
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key_12345'})
    @patch('main.ExcelReader')
    @patch('main.ConfigurationManager')
    @patch('main.SentimentAnalyzer')
    def test_main_with_no_save_option(self, mock_analyzer_class, mock_config_manager_class, mock_excel_reader_class):
        """Test main application with --no-save option."""
        # Mock Excel reader
        mock_excel_reader = Mock()
        mock_excel_reader_class.return_value = mock_excel_reader
        mock_excel_reader.get_file_info.return_value = {
            'file_size_mb': 0.1,
            'column_count': 2,
            'columns': ['ID', 'Feedback']
        }
        mock_excel_reader.load_responses.return_value = [
            {
                'text_content': 'Great product!',
                'headers': ['ID', 'Feedback'],
                'row_index': 0,
                'source_file': 'respostas.xlsx'
            }
        ]
        
        # Mock analyzer and its results
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_results = [
            self._create_mock_sentiment_result("Great product!", "positive", 0.9)
        ]
        
        # Create mock analysis results
        from src.models import AnalysisResults, Summary
        from datetime import datetime
        
        mock_analysis_results = AnalysisResults(
            individual_results=mock_results,
            summary_stats={},
            processing_metadata={},
            timestamp=datetime.now(),
            total_processed=1,
            success_rate=1.0
        )
        
        mock_summary = Summary(
            sentiment_distribution={'positive': 1, 'negative': 0, 'neutral': 0},
            sentiment_percentages={'positive': 100.0, 'negative': 0.0, 'neutral': 0.0},
            common_themes=[],
            key_insights=['Test insight'],
            recommendations=['Test recommendation'],
            confidence_stats={'mean': 0.9}
        )
        
        mock_analyzer.process_responses_from_data.return_value = mock_analysis_results
        mock_analyzer.generate_insights.return_value = mock_summary
        
        # Mock configuration manager
        mock_config_manager = Mock()
        mock_config_manager_class.return_value = mock_config_manager
        mock_config = Configuration(openai_api_key='test_key_12345')
        mock_config_manager.load_configuration.return_value = mock_config
        mock_config_manager.get_safe_config_summary.return_value = {
            'api_key_preview': 'test_key_*****',
            'model': 'gpt-3.5-turbo',
            'timeout': 30.0,
            'max_retries': 3
        }
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            from main import main
            args = self._create_mock_args(no_save=True)
            result = main(args)
            
            # Verify successful execution
            self.assertEqual(result, 0)
            
            # Get the output
            output = captured_output.getvalue()
            
            # Verify no file saving occurred
            self.assertNotIn("Saving results to", output)
            
            # Verify no output directory was created
            self.assertFalse(os.path.exists(self.output_dir))
            
        finally:
            sys.stdout = sys.__stdout__
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key_12345'})
    @patch('main.ExcelReader')
    @patch('main.ConfigurationManager')
    @patch('main.ReportGenerator')
    @patch('main.SentimentAnalyzer')
    def test_main_with_custom_file_and_output_dir(self, mock_analyzer_class, mock_report_generator_class, mock_config_manager_class, mock_excel_reader_class):
        """Test main application with custom file and output directory."""
        # Create custom Excel file
        custom_file = os.path.join(self.temp_dir, "custom_survey.xlsx")
        data = {
            'Response': ['This is excellent!', 'Very poor quality']
        }
        df = pd.DataFrame(data)
        df.to_excel(custom_file, index=False)
        
        # Mock Excel reader
        mock_excel_reader = Mock()
        mock_excel_reader_class.return_value = mock_excel_reader
        mock_excel_reader.get_file_info.return_value = {
            'file_size_mb': 0.1,
            'column_count': 1,
            'columns': ['Response']
        }
        mock_excel_reader.load_responses.return_value = [
            {
                'text_content': 'This is excellent!',
                'headers': ['Response'],
                'row_index': 0,
                'source_file': 'custom_survey.xlsx'
            },
            {
                'text_content': 'Very poor quality',
                'headers': ['Response'],
                'row_index': 1,
                'source_file': 'custom_survey.xlsx'
            }
        ]
        
        # Mock analyzer and its results
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_results = [
            self._create_mock_sentiment_result("This is excellent!", "positive", 0.95),
            self._create_mock_sentiment_result("Very poor quality", "negative", 0.88)
        ]
        
        # Create mock analysis results
        from src.models import AnalysisResults, Summary
        from datetime import datetime
        
        mock_analysis_results = AnalysisResults(
            individual_results=mock_results,
            summary_stats={},
            processing_metadata={},
            timestamp=datetime.now(),
            total_processed=2,
            success_rate=1.0
        )
        
        mock_summary = Summary(
            sentiment_distribution={'positive': 1, 'negative': 1, 'neutral': 0},
            sentiment_percentages={'positive': 50.0, 'negative': 50.0, 'neutral': 0.0},
            common_themes=[],
            key_insights=['Test insight'],
            recommendations=['Test recommendation'],
            confidence_stats={'mean': 0.915}
        )
        
        mock_analyzer.process_responses_from_data.return_value = mock_analysis_results
        mock_analyzer.generate_insights.return_value = mock_summary
        
        # Mock report generator
        mock_report_generator = Mock()
        mock_report_generator_class.return_value = mock_report_generator
        mock_report_generator.display_console_summary.return_value = None
        mock_report_generator.save_detailed_results.return_value = "/mock/path/results.json"
        mock_report_generator.save_summary_report.return_value = "/mock/path/summary.txt"
        
        # Mock configuration manager
        mock_config_manager = Mock()
        mock_config_manager_class.return_value = mock_config_manager
        mock_config = Configuration(openai_api_key='test_key_12345')
        mock_config_manager.load_configuration.return_value = mock_config
        mock_config_manager.get_safe_config_summary.return_value = {
            'api_key_preview': 'test_key_*****',
            'model': 'gpt-3.5-turbo',
            'timeout': 30.0,
            'max_retries': 3
        }
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            from main import main
            args = self._create_mock_args(
                file='custom_survey.xlsx',
                output_dir='custom_output'
            )
            result = main(args)
            
            # Verify successful execution
            self.assertEqual(result, 0)
            
            # Get the output
            output = captured_output.getvalue()
            
            # Verify custom file was processed
            self.assertIn("custom_survey.xlsx", output)
            self.assertIn("Successfully loaded 2 responses", output)
            
            # Verify custom output directory was used
            self.assertIn("custom_output", output)
            
        finally:
            sys.stdout = sys.__stdout__
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key_12345'})
    @patch('main.ExcelReader')
    @patch('main.ConfigurationManager')
    @patch('main.SentimentAnalyzer')
    def test_main_with_verbose_logging(self, mock_analyzer_class, mock_config_manager_class, mock_excel_reader_class):
        """Test main application with verbose logging enabled."""
        # Mock Excel reader
        mock_excel_reader = Mock()
        mock_excel_reader_class.return_value = mock_excel_reader
        mock_excel_reader.get_file_info.return_value = {
            'file_size_mb': 0.1,
            'column_count': 2,
            'columns': ['ID', 'Feedback']
        }
        mock_excel_reader.load_responses.return_value = [
            {
                'text_content': 'Good product',
                'headers': ['ID', 'Feedback'],
                'row_index': 0,
                'source_file': 'respostas.xlsx'
            }
        ]
        
        # Mock analyzer and its results
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_results = [
            self._create_mock_sentiment_result("Good product", "positive", 0.8)
        ]
        
        # Create mock analysis results
        from src.models import AnalysisResults, Summary
        from datetime import datetime
        
        mock_analysis_results = AnalysisResults(
            individual_results=mock_results,
            summary_stats={},
            processing_metadata={},
            timestamp=datetime.now(),
            total_processed=1,
            success_rate=1.0
        )
        
        mock_summary = Summary(
            sentiment_distribution={'positive': 1, 'negative': 0, 'neutral': 0},
            sentiment_percentages={'positive': 100.0, 'negative': 0.0, 'neutral': 0.0},
            common_themes=[],
            key_insights=['Test insight'],
            recommendations=['Test recommendation'],
            confidence_stats={'mean': 0.8}
        )
        
        mock_analyzer.process_responses_from_data.return_value = mock_analysis_results
        mock_analyzer.generate_insights.return_value = mock_summary
        
        # Mock configuration manager
        mock_config_manager = Mock()
        mock_config_manager_class.return_value = mock_config_manager
        mock_config = Configuration(openai_api_key='test_key_12345')
        mock_config_manager.load_configuration.return_value = mock_config
        mock_config_manager.get_safe_config_summary.return_value = {
            'api_key_preview': 'test_key_*****',
            'model': 'gpt-3.5-turbo',
            'timeout': 30.0,
            'max_retries': 3
        }
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            from main import main
            args = self._create_mock_args(verbose=True, no_save=True)
            result = main(args)
            
            # Verify successful execution
            self.assertEqual(result, 0)
            
        finally:
            sys.stdout = sys.__stdout__
    
    def test_command_line_argument_parsing(self):
        """Test command-line argument parsing."""
        from main import parse_arguments
        
        # Test default arguments
        with patch('sys.argv', ['main.py']):
            args = parse_arguments()
            self.assertEqual(args.file, 'respostas.xlsx')
            self.assertEqual(args.output_dir, 'output')
            self.assertFalse(args.no_save)
            self.assertFalse(args.no_samples)
            self.assertFalse(args.verbose)
        
        # Test custom arguments
        with patch('sys.argv', ['main.py', '-f', 'custom.xlsx', '-o', 'results', '--no-save', '--no-samples', '-v']):
            args = parse_arguments()
            self.assertEqual(args.file, 'custom.xlsx')
            self.assertEqual(args.output_dir, 'results')
            self.assertTrue(args.no_save)
            self.assertTrue(args.no_samples)
            self.assertTrue(args.verbose)
    
    def test_input_file_validation(self):
        """Test input file validation function."""
        from main import validate_input_file
        
        # Test existing file
        self.assertTrue(validate_input_file(self.test_excel_path))
        
        # Test non-existent file
        self.assertFalse(validate_input_file('nonexistent.xlsx'))


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    unittest.main()