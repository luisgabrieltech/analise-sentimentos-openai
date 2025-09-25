"""
Unit tests for the ReportGenerator class.

Tests report formatting, file operations, console output, and error handling
for the sentiment analysis report generation system.
"""

import unittest
import tempfile
import shutil
import os
import json
from datetime import datetime
from unittest.mock import patch, mock_open
from io import StringIO

from src.report_generator import ReportGenerator
from src.models import SentimentResult, AnalysisResults, Summary


class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.report_generator = ReportGenerator(output_directory=self.temp_dir)
        
        # Create sample data for testing
        self.sample_results = [
            SentimentResult(
                original_text="This is a great product! I love it.",
                sentiment="positive",
                confidence=0.95,
                reasoning="Positive language with enthusiasm",
                processing_time=1.2,
                success=True
            ),
            SentimentResult(
                original_text="The service was okay, nothing special.",
                sentiment="neutral",
                confidence=0.75,
                reasoning="Neutral tone with mild satisfaction",
                processing_time=1.1,
                success=True
            ),
            SentimentResult(
                original_text="Terrible experience, would not recommend.",
                sentiment="negative",
                confidence=0.88,
                reasoning="Strong negative language",
                processing_time=1.3,
                success=True
            ),
            SentimentResult(
                original_text="Failed to analyze this text",
                sentiment="neutral",
                confidence=0.0,
                reasoning="",
                processing_time=0.5,
                success=False,
                error_message="API Error: Rate limit exceeded"
            )
        ]
        
        self.sample_analysis_results = AnalysisResults(
            individual_results=self.sample_results,
            summary_stats={
                "sentiment_distribution": {"positive": 1, "neutral": 1, "negative": 1},
                "confidence_stats": {"mean": 0.86, "median": 0.88, "min": 0.75, "max": 0.95}
            },
            processing_metadata={
                "total_api_calls": 4,
                "failed_api_calls": 1,
                "processing_duration": 4.1,
                "client_statistics": {
                    "total_requests": 4,
                    "success_rate_percent": 75.0,
                    "average_request_time": 1.025,
                    "rate_limit_hits": 0,
                    "timeout_errors": 0
                }
            },
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            total_processed=4,
            success_rate=0.75
        )
        
        self.sample_summary = Summary(
            sentiment_distribution={"positive": 1, "neutral": 1, "negative": 1},
            sentiment_percentages={"positive": 33.3, "neutral": 33.3, "negative": 33.4},
            common_themes=["product quality", "service experience", "recommendations"],
            key_insights=["Mixed sentiment across responses", "High confidence in classifications"],
            recommendations=["Focus on improving negative experiences", "Maintain positive aspects"],
            confidence_stats={"mean": 0.86, "median": 0.88, "min": 0.75, "max": 0.95, "std_dev": 0.08}
        )
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory and all its contents
        shutil.rmtree(self.temp_dir)
    
    def test_init_creates_output_directory(self):
        """Test that ReportGenerator creates output directory if it doesn't exist."""
        new_temp_dir = os.path.join(self.temp_dir, "new_output")
        self.assertFalse(os.path.exists(new_temp_dir))
        
        generator = ReportGenerator(output_directory=new_temp_dir)
        self.assertTrue(os.path.exists(new_temp_dir))
    
    def test_create_summary_report_with_results(self):
        """Test creating a summary report with valid results."""
        report = self.report_generator.create_summary_report(self.sample_analysis_results)
        
        # Check that report contains expected sections
        self.assertIn("SENTIMENT ANALYSIS REPORT", report)
        self.assertIn("EXECUTIVE SUMMARY", report)
        self.assertIn("DETAILED STATISTICS", report)
        self.assertIn("SAMPLE RESULTS", report)
        self.assertIn("PROCESSING INFORMATION", report)
        
        # Check specific data points
        self.assertIn("Total Responses Analyzed: 4", report)
        self.assertIn("Success Rate: 75.0%", report)
        self.assertIn("Positive: 1 responses", report)
        self.assertIn("Neutral:  1 responses", report)
        self.assertIn("Negative: 1 responses", report)
        
        # Check timestamp formatting
        self.assertIn("2024-01-15 10:30:00", report)
    
    def test_create_summary_report_empty_results(self):
        """Test creating a summary report with no results."""
        empty_results = AnalysisResults(
            individual_results=[],
            summary_stats={},
            processing_metadata={},
            timestamp=datetime.now(),
            total_processed=0,
            success_rate=0.0
        )
        
        report = self.report_generator.create_summary_report(empty_results)
        
        self.assertIn("No results available for analysis", report)
        self.assertIn("No responses found in the input file", report)
    
    def test_save_detailed_results_default_filename(self):
        """Test saving detailed results with default filename."""
        filepath = self.report_generator.save_detailed_results(self.sample_analysis_results)
        
        # Check that file was created
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.json'))
        
        # Check file contents
        with open(filepath, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data['total_processed'], 4)
        self.assertEqual(saved_data['success_rate'], 0.75)
        self.assertEqual(len(saved_data['individual_results']), 4)
        self.assertIn('timestamp', saved_data)
    
    def test_save_detailed_results_custom_filename(self):
        """Test saving detailed results with custom filename."""
        custom_filename = "custom_results.json"
        filepath = self.report_generator.save_detailed_results(
            self.sample_analysis_results, 
            filename=custom_filename
        )
        
        expected_path = os.path.join(self.temp_dir, custom_filename)
        self.assertEqual(filepath, expected_path)
        self.assertTrue(os.path.exists(filepath))
    
    def test_save_detailed_results_file_error(self):
        """Test handling of file save errors."""
        # Create a ReportGenerator and then mock the file operation to fail
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.side_effect = PermissionError("Permission denied")
            
            with self.assertRaises(IOError) as context:
                self.report_generator.save_detailed_results(self.sample_analysis_results)
            
            error_message = str(context.exception)
            self.assertTrue(
                "Permission denied" in error_message or "Failed to save results" in error_message,
                f"Expected error message not found in: {error_message}"
            )
    
    def test_save_summary_report(self):
        """Test saving summary report to text file."""
        filepath = self.report_generator.save_summary_report(self.sample_analysis_results)
        
        # Check that file was created
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.txt'))
        
        # Check file contents
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("SENTIMENT ANALYSIS REPORT", content)
        self.assertIn("Total Responses Analyzed: 4", content)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_display_console_summary_with_results(self, mock_stdout):
        """Test displaying console summary with results."""
        self.report_generator.display_console_summary(self.sample_analysis_results)
        
        output = mock_stdout.getvalue()
        
        self.assertIn("SENTIMENT ANALYSIS COMPLETE", output)
        self.assertIn("Total Processed: 4", output)
        self.assertIn("Success Rate: 75.0%", output)
        self.assertIn("SENTIMENT BREAKDOWN:", output)
        self.assertIn("SAMPLE RESULTS:", output)
        
        # Check for progress bars
        self.assertIn("█", output)  # Should contain progress bar characters
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_display_console_summary_no_results(self, mock_stdout):
        """Test displaying console summary with no results."""
        empty_results = AnalysisResults(
            individual_results=[],
            summary_stats={},
            processing_metadata={},
            timestamp=datetime.now(),
            total_processed=0,
            success_rate=0.0
        )
        
        self.report_generator.display_console_summary(empty_results)
        
        output = mock_stdout.getvalue()
        self.assertIn("No results to display", output)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_display_console_summary_no_samples(self, mock_stdout):
        """Test displaying console summary without sample results."""
        self.report_generator.display_console_summary(
            self.sample_analysis_results, 
            show_samples=False
        )
        
        output = mock_stdout.getvalue()
        self.assertIn("SENTIMENT BREAKDOWN:", output)
        self.assertNotIn("SAMPLE RESULTS:", output)
    
    def test_calculate_summary_statistics_with_results(self):
        """Test calculation of summary statistics with valid results."""
        stats = self.report_generator._calculate_summary_statistics(self.sample_results)
        
        # Check sentiment distribution
        self.assertEqual(stats['sentiment_distribution']['positive'], 1)
        self.assertEqual(stats['sentiment_distribution']['neutral'], 1)
        self.assertEqual(stats['sentiment_distribution']['negative'], 1)
        
        # Check confidence stats
        self.assertAlmostEqual(stats['confidence_stats']['mean'], 0.86, places=2)
        self.assertEqual(stats['confidence_stats']['min'], 0.75)
        self.assertEqual(stats['confidence_stats']['max'], 0.95)
        
        # Check processing time stats
        self.assertGreater(stats['processing_time_stats']['total'], 0)
        self.assertGreater(stats['processing_time_stats']['mean'], 0)
    
    def test_calculate_summary_statistics_no_successful_results(self):
        """Test calculation of summary statistics with no successful results."""
        failed_results = [
            SentimentResult(
                original_text="Failed text",
                sentiment="neutral",
                confidence=0.0,
                reasoning="",
                processing_time=0.0,
                success=False,
                error_message="API Error"
            )
        ]
        
        stats = self.report_generator._calculate_summary_statistics(failed_results)
        
        # All stats should be zero or empty
        self.assertEqual(stats['sentiment_distribution']['positive'], 0)
        self.assertEqual(stats['confidence_stats']['mean'], 0)
        self.assertEqual(stats['processing_time_stats']['total'], 0)
    
    def test_calculate_std_dev(self):
        """Test standard deviation calculation."""
        # Test with known values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        std_dev = self.report_generator._calculate_std_dev(values)
        self.assertAlmostEqual(std_dev, 1.58, places=2)
        
        # Test with single value
        single_value = [5.0]
        std_dev_single = self.report_generator._calculate_std_dev(single_value)
        self.assertEqual(std_dev_single, 0.0)
        
        # Test with empty list
        empty_list = []
        std_dev_empty = self.report_generator._calculate_std_dev(empty_list)
        self.assertEqual(std_dev_empty, 0.0)
    
    def test_create_progress_display(self):
        """Test creation of progress display string."""
        progress_str = self.report_generator.create_progress_display(
            current=25,
            total=100,
            current_item="Processing response about product quality",
            success_rate=0.95,
            elapsed_time=120.5
        )
        
        self.assertIn("Progress:", progress_str)
        self.assertIn("25.0%", progress_str)
        self.assertIn("25/100", progress_str)
        self.assertIn("95.0%", progress_str)
        self.assertIn("2m 0s", progress_str)  # 120.5 seconds formatted
        self.assertIn("Processing response about product", progress_str)
        
        # Check progress bar
        self.assertIn("█", progress_str)
        self.assertIn("░", progress_str)
    
    def test_create_progress_display_invalid_total(self):
        """Test progress display with invalid total."""
        progress_str = self.report_generator.create_progress_display(
            current=10,
            total=0
        )
        
        self.assertIn("Invalid progress parameters", progress_str)
    
    def test_create_progress_display_time_formatting(self):
        """Test different time formatting scenarios."""
        # Test seconds only
        progress_str = self.report_generator.create_progress_display(
            current=1, total=10, elapsed_time=45
        )
        self.assertIn("45s", progress_str)
        
        # Test hours and minutes
        progress_str = self.report_generator.create_progress_display(
            current=1, total=10, elapsed_time=3665  # 1 hour, 1 minute, 5 seconds
        )
        self.assertIn("1h 1m", progress_str)
    
    def test_report_sections_with_insights(self):
        """Test that report includes insights section when summary is provided."""
        # Add summary to analysis results
        self.sample_analysis_results.summary = self.sample_summary
        
        report = self.report_generator.create_summary_report(self.sample_analysis_results)
        
        self.assertIn("INSIGHTS & RECOMMENDATIONS", report)
        self.assertIn("Mixed sentiment across responses", report)
        self.assertIn("Focus on improving negative experiences", report)
        self.assertIn("product quality", report)
    
    def test_sample_results_section(self):
        """Test the sample results section formatting."""
        report = self.report_generator.create_summary_report(self.sample_analysis_results)
        
        self.assertIn("SAMPLE RESULTS", report)
        self.assertIn("Positive Examples:", report)
        self.assertIn("Negative Examples:", report)
        self.assertIn("Neutral Examples:", report)
        
        # Check that confidence scores are included
        self.assertIn("Confidence: 0.950", report)
        self.assertIn("Confidence: 0.880", report)
    
    def test_processing_info_section(self):
        """Test the processing information section."""
        report = self.report_generator.create_summary_report(self.sample_analysis_results)
        
        self.assertIn("PROCESSING INFORMATION", report)
        self.assertIn("Processing Details:", report)
        self.assertIn("Total API Requests: 4", report)
        self.assertIn("Failed Analyses (1 total):", report)
        self.assertIn("API Error: 1 occurrences", report)
    
    def test_unicode_handling(self):
        """Test handling of unicode characters in text."""
        unicode_result = SentimentResult(
            original_text="Great product! 😊 Very satisfied with the quality 👍",
            sentiment="positive",
            confidence=0.92,
            reasoning="Positive sentiment with emojis",
            processing_time=1.1,
            success=True
        )
        
        unicode_results = AnalysisResults(
            individual_results=[unicode_result],
            summary_stats={},
            processing_metadata={},
            timestamp=datetime.now(),
            total_processed=1,
            success_rate=1.0
        )
        
        # Should not raise any encoding errors
        report = self.report_generator.create_summary_report(unicode_results)
        self.assertIn("😊", report)
        self.assertIn("👍", report)
        
        # Test saving to file with unicode
        filepath = self.report_generator.save_detailed_results(unicode_results)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("😊", content)


if __name__ == '__main__':
    unittest.main()