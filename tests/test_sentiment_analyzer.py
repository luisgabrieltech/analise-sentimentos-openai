#!/usr/bin/env python3
"""
Integration tests for SentimentAnalyzer

This module contains comprehensive integration tests for the SentimentAnalyzer class,
testing the complete processing flow with real and mocked components.
"""

import unittest
import tempfile
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.sentiment_analyzer import SentimentAnalyzer, SentimentAnalyzerError
from src.models import SentimentResult, AnalysisResults, Summary
from src.config_manager import Configuration
from src.excel_reader import ExcelReaderError
from src.openai_client import OpenAIError


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Configuration(
            openai_api_key="test_key_12345",
            openai_model="gpt-3.5-turbo",
            api_timeout=30,
            max_retries=3
        )
        
        # Create temporary Excel file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_excel_path = os.path.join(self.temp_dir, "test_responses.xlsx")
        self._create_test_excel_file()
        
        # Create analyzer instance
        self.analyzer = SentimentAnalyzer(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        if os.path.exists(self.test_excel_path):
            os.remove(self.test_excel_path)
        os.rmdir(self.temp_dir)
    
    def _create_test_excel_file(self):
        """Create a test Excel file with sample responses."""
        data = {
            'ID': [1, 2, 3, 4, 5],
            'Response': [
                'I love this product! It works perfectly.',
                'This is terrible. I hate it completely.',
                'It is okay, nothing special.',
                'Amazing quality and great service!',
                'Could be better, but not bad.'
            ],
            'Category': ['Product', 'Product', 'Product', 'Service', 'General']
        }
        df = pd.DataFrame(data)
        df.to_excel(self.test_excel_path, index=False)
    
    def _create_mock_sentiment_result(self, text: str, sentiment: str, confidence: float, success: bool = True):
        """Create a mock SentimentResult for testing."""
        return SentimentResult(
            original_text=text,
            sentiment=sentiment,
            confidence=confidence,
            reasoning=f"Test reasoning for {sentiment} sentiment",
            processing_time=0.5,
            success=success,
            error_message=None if success else "Test error"
        )
    
    @patch('src.sentiment_analyzer.OpenAIClient')
    def test_process_responses_success(self, mock_openai_client_class):
        """Test successful processing of responses."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_client_class.return_value = mock_client
        
        # Ensure the analyzer uses the mocked client
        self.analyzer.openai_client = mock_client
        
        # Mock sentiment analysis results
        mock_results = [
            self._create_mock_sentiment_result("I love this product! It works perfectly.", "positive", 0.9),
            self._create_mock_sentiment_result("This is terrible. I hate it completely.", "negative", 0.8),
            self._create_mock_sentiment_result("It is okay, nothing special.", "neutral", 0.6),
            self._create_mock_sentiment_result("Amazing quality and great service!", "positive", 0.95),
            self._create_mock_sentiment_result("Could be better, but not bad.", "neutral", 0.7)
        ]
        
        mock_client.analyze_sentiment.side_effect = mock_results
        mock_client.get_client_stats.return_value = {
            "total_requests": 5,
            "successful_requests": 5,
            "failed_requests": 0,
            "success_rate_percent": 100.0,
            "rate_limit_hits": 0,
            "timeout_errors": 0,
            "consecutive_failures": 0,
            "current_rate_limit_delay": 0.1,
            "average_request_time": 0.5,
            "model": "gpt-3.5-turbo",
            "timeout": 30,
            "max_retries": 3,
            "last_request_time": 0.0,
            "last_rate_limit_time": 0.0
        }
        
        # Process responses
        results = self.analyzer.process_responses(self.test_excel_path)
        
        # Verify results
        self.assertIsInstance(results, AnalysisResults)
        self.assertEqual(len(results.individual_results), 5)
        self.assertEqual(results.total_processed, 5)
        self.assertEqual(results.success_rate, 1.0)
        
        # Verify sentiment distribution
        sentiment_dist = results.summary_stats["sentiment_distribution"]
        self.assertEqual(sentiment_dist["positive"], 2)
        self.assertEqual(sentiment_dist["negative"], 1)
        self.assertEqual(sentiment_dist["neutral"], 2)
        
        # Verify all results are successful
        for result in results.individual_results:
            self.assertTrue(result.success)
        
        # Verify OpenAI client was called correctly
        self.assertEqual(mock_client.analyze_sentiment.call_count, 5)
    
    @patch('src.sentiment_analyzer.OpenAIClient')
    def test_process_responses_with_failures(self, mock_openai_client_class):
        """Test processing with some failed analyses."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_client_class.return_value = mock_client
        
        # Ensure the analyzer uses the mocked client
        self.analyzer.openai_client = mock_client
        
        # Mock sentiment analysis results with some failures
        mock_results = [
            self._create_mock_sentiment_result("I love this product! It works perfectly.", "positive", 0.9),
            self._create_mock_sentiment_result("This is terrible. I hate it completely.", "negative", 0.0, success=False),
            self._create_mock_sentiment_result("It is okay, nothing special.", "neutral", 0.6),
            self._create_mock_sentiment_result("Amazing quality and great service!", "positive", 0.0, success=False),
            self._create_mock_sentiment_result("Could be better, but not bad.", "neutral", 0.7)
        ]
        
        mock_client.analyze_sentiment.side_effect = mock_results
        mock_client.get_client_stats.return_value = {
            "total_requests": 5,
            "successful_requests": 3,
            "failed_requests": 2,
            "success_rate_percent": 60.0,
            "rate_limit_hits": 0,
            "timeout_errors": 0,
            "consecutive_failures": 0,
            "current_rate_limit_delay": 0.1,
            "average_request_time": 0.5,
            "model": "gpt-3.5-turbo",
            "timeout": 30,
            "max_retries": 3,
            "last_request_time": 0.0,
            "last_rate_limit_time": 0.0
        }
        
        # Process responses
        results = self.analyzer.process_responses(self.test_excel_path)
        
        # Verify results
        self.assertEqual(len(results.individual_results), 5)
        self.assertEqual(results.total_processed, 5)
        self.assertEqual(results.success_rate, 0.6)  # 3 out of 5 successful
        
        # Verify success/failure counts
        successful_count = sum(1 for r in results.individual_results if r.success)
        failed_count = sum(1 for r in results.individual_results if not r.success)
        self.assertEqual(successful_count, 3)
        self.assertEqual(failed_count, 2)
    
    def test_process_responses_excel_error(self):
        """Test handling of Excel reading errors."""
        # Try to process non-existent file
        with self.assertRaises(SentimentAnalyzerError) as context:
            self.analyzer.process_responses("nonexistent_file.xlsx")
        
        self.assertIn("Failed to load Excel file", str(context.exception))
    
    @patch('src.sentiment_analyzer.OpenAIClient')
    def test_process_responses_openai_error(self, mock_openai_client_class):
        """Test handling of OpenAI API errors."""
        # Mock OpenAI client to raise errors
        mock_client = Mock()
        mock_openai_client_class.return_value = mock_client
        mock_client.analyze_sentiment.side_effect = OpenAIError("API Error")
        
        # Ensure the analyzer uses the mocked client
        self.analyzer.openai_client = mock_client
        mock_client.get_client_stats.return_value = {
            "total_requests": 5,
            "successful_requests": 0,
            "failed_requests": 5,
            "success_rate_percent": 0.0,
            "rate_limit_hits": 0,
            "timeout_errors": 0,
            "consecutive_failures": 5,
            "current_rate_limit_delay": 0.1,
            "average_request_time": 0.5,
            "model": "gpt-3.5-turbo",
            "timeout": 30,
            "max_retries": 3,
            "last_request_time": 0.0,
            "last_rate_limit_time": 0.0
        }
        
        # Process responses - should handle errors gracefully
        results = self.analyzer.process_responses(self.test_excel_path)
        
        # Verify all results failed but processing completed
        self.assertEqual(len(results.individual_results), 5)
        self.assertEqual(results.success_rate, 0.0)
        
        for result in results.individual_results:
            self.assertFalse(result.success)
            self.assertIn("API Error", result.error_message)
    
    @patch('src.sentiment_analyzer.OpenAIClient')
    def test_generate_insights_success(self, mock_openai_client_class):
        """Test successful insight generation."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_client_class.return_value = mock_client
        
        # Ensure the analyzer uses the mocked client
        self.analyzer.openai_client = mock_client
        
        # Create mock results
        mock_results = [
            self._create_mock_sentiment_result("Positive text", "positive", 0.9),
            self._create_mock_sentiment_result("Negative text", "negative", 0.8),
            self._create_mock_sentiment_result("Neutral text", "neutral", 0.6)
        ]
        
        mock_client.analyze_sentiment.side_effect = mock_results
        mock_client.get_client_stats.return_value = {
            "total_requests": 3,
            "successful_requests": 3,
            "failed_requests": 0,
            "success_rate_percent": 100.0,
            "rate_limit_hits": 0,
            "timeout_errors": 0,
            "consecutive_failures": 0,
            "current_rate_limit_delay": 0.1,
            "average_request_time": 0.5,
            "model": "gpt-3.5-turbo",
            "timeout": 30,
            "max_retries": 3,
            "last_request_time": 0.0,
            "last_rate_limit_time": 0.0
        }
        
        # Process responses first
        analysis_results = self.analyzer.process_responses(self.test_excel_path)
        
        # Generate insights
        summary = self.analyzer.generate_insights(analysis_results)
        
        # Verify summary structure
        self.assertIsInstance(summary, Summary)
        self.assertIn("positive", summary.sentiment_distribution)
        self.assertIn("negative", summary.sentiment_distribution)
        self.assertIn("neutral", summary.sentiment_distribution)
        
        # Verify percentages sum to 100
        total_percentage = sum(summary.sentiment_percentages.values())
        self.assertAlmostEqual(total_percentage, 100.0, places=1)
        
        # Verify insights and recommendations are generated
        self.assertIsInstance(summary.key_insights, list)
        self.assertIsInstance(summary.recommendations, list)
        self.assertIsInstance(summary.common_themes, list)
        
        # Verify insights contain meaningful content
        self.assertGreater(len(summary.key_insights), 0)
        self.assertGreater(len(summary.recommendations), 0)
    
    def test_generate_insights_no_analysis(self):
        """Test insight generation without prior analysis."""
        with self.assertRaises(SentimentAnalyzerError) as context:
            self.analyzer.generate_insights()
        
        self.assertIn("No analysis results available", str(context.exception))
    
    @patch('src.sentiment_analyzer.OpenAIClient')
    def test_empty_excel_file(self, mock_openai_client_class):
        """Test handling of empty Excel file."""
        # Create empty Excel file
        empty_excel_path = os.path.join(self.temp_dir, "empty.xlsx")
        df = pd.DataFrame()
        df.to_excel(empty_excel_path, index=False)
        
        # Try to process empty file
        with self.assertRaises(SentimentAnalyzerError) as context:
            self.analyzer.process_responses(empty_excel_path)
        
        self.assertIn("Failed to load Excel file", str(context.exception))
        
        # Clean up
        os.remove(empty_excel_path)
    
    @patch('src.sentiment_analyzer.OpenAIClient')
    def test_progress_tracking(self, mock_openai_client_class):
        """Test progress tracking functionality."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_client_class.return_value = mock_client
        
        # Ensure the analyzer uses the mocked client
        self.analyzer.openai_client = mock_client
        
        # Mock sentiment analysis results
        mock_results = [
            self._create_mock_sentiment_result("Text 1", "positive", 0.9),
            self._create_mock_sentiment_result("Text 2", "negative", 0.8)
        ]
        
        mock_client.analyze_sentiment.side_effect = mock_results
        mock_client.get_client_stats.return_value = {
            "total_requests": 2,
            "successful_requests": 2,
            "failed_requests": 0,
            "success_rate_percent": 100.0,
            "rate_limit_hits": 0,
            "timeout_errors": 0,
            "consecutive_failures": 0,
            "current_rate_limit_delay": 0.1,
            "average_request_time": 0.5,
            "model": "gpt-3.5-turbo",
            "timeout": 30,
            "max_retries": 3,
            "last_request_time": 0.0,
            "last_rate_limit_time": 0.0
        }
        
        # Process responses
        results = self.analyzer.process_responses(self.test_excel_path)
        
        # Verify progress tracker was created and used
        self.assertIsNotNone(self.analyzer.progress_tracker)
        self.assertTrue(self.analyzer.progress_tracker.is_complete())
        
        # Verify progress tracker statistics
        progress_stats = self.analyzer.progress_tracker.get_status_summary()
        self.assertEqual(progress_stats["total_items"], 5)
        self.assertEqual(progress_stats["processed_items"], 5)
        self.assertEqual(progress_stats["progress_percentage"], 100.0)
    
    @patch('src.sentiment_analyzer.OpenAIClient')
    def test_summary_statistics_calculation(self, mock_openai_client_class):
        """Test detailed summary statistics calculation."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_client_class.return_value = mock_client
        
        # Ensure the analyzer uses the mocked client
        self.analyzer.openai_client = mock_client
        
        # Create specific mock results for testing statistics
        mock_results = [
            self._create_mock_sentiment_result("Positive 1", "positive", 0.9),
            self._create_mock_sentiment_result("Positive 2", "positive", 0.8),
            self._create_mock_sentiment_result("Negative 1", "negative", 0.7),
            self._create_mock_sentiment_result("Neutral 1", "neutral", 0.6),
            self._create_mock_sentiment_result("Failed", "neutral", 0.0, success=False)
        ]
        
        mock_client.analyze_sentiment.side_effect = mock_results
        mock_client.get_client_stats.return_value = {
            "total_requests": 5,
            "successful_requests": 4,
            "failed_requests": 1,
            "success_rate_percent": 80.0,
            "rate_limit_hits": 0,
            "timeout_errors": 0,
            "consecutive_failures": 0,
            "current_rate_limit_delay": 0.1,
            "average_request_time": 0.5,
            "model": "gpt-3.5-turbo",
            "timeout": 30,
            "max_retries": 3,
            "last_request_time": 0.0,
            "last_rate_limit_time": 0.0
        }
        
        # Process responses
        results = self.analyzer.process_responses(self.test_excel_path)
        
        # Verify detailed statistics
        stats = results.summary_stats
        
        # Check basic counts
        self.assertEqual(stats["total_responses"], 5)
        self.assertEqual(stats["successful_analyses"], 4)
        self.assertEqual(stats["failed_analyses"], 1)
        self.assertEqual(stats["success_rate"], 80.0)
        
        # Check sentiment distribution (only successful analyses)
        sentiment_dist = stats["sentiment_distribution"]
        self.assertEqual(sentiment_dist["positive"], 2)
        self.assertEqual(sentiment_dist["negative"], 1)
        self.assertEqual(sentiment_dist["neutral"], 1)
        
        # Check sentiment percentages (should sum to 100%)
        sentiment_pct = stats["sentiment_percentages"]
        total_pct = sum(sentiment_pct.values())
        self.assertAlmostEqual(total_pct, 100.0, places=1)
        
        # Check confidence statistics
        confidence_stats = stats["confidence_stats"]
        self.assertGreater(confidence_stats["mean"], 0)
        self.assertGreater(confidence_stats["max"], confidence_stats["min"])
    
    @patch('src.sentiment_analyzer.OpenAIClient')
    def test_metadata_creation(self, mock_openai_client_class):
        """Test processing metadata creation."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_client_class.return_value = mock_client
        mock_client.analyze_sentiment.return_value = self._create_mock_sentiment_result("Test", "positive", 0.9)
        
        # Ensure the analyzer uses the mocked client
        self.analyzer.openai_client = mock_client
        mock_client.get_client_stats.return_value = {
            "total_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
            "success_rate_percent": 100.0,
            "rate_limit_hits": 0,
            "timeout_errors": 0,
            "consecutive_failures": 0,
            "current_rate_limit_delay": 0.1,
            "average_request_time": 0.5,
            "model": "gpt-3.5-turbo",
            "timeout": 30,
            "max_retries": 3,
            "last_request_time": 0.0,
            "last_rate_limit_time": 0.0
        }
        
        # Process responses
        results = self.analyzer.process_responses(self.test_excel_path)
        
        # Verify metadata
        metadata = results.processing_metadata
        self.assertEqual(metadata["source_file"], self.test_excel_path)
        self.assertEqual(metadata["openai_model"], "gpt-3.5-turbo")
        self.assertEqual(metadata["api_timeout"], 30)
        self.assertEqual(metadata["max_retries"], 3)
        self.assertIn("client_statistics", metadata)
        self.assertIn("progress_summary", metadata)
        self.assertIn("total_processing_time", metadata)
        self.assertGreater(metadata["total_processing_time"], 0)


class TestSentimentAnalyzerIntegration(unittest.TestCase):
    """Integration tests that test the complete flow with minimal mocking."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = Configuration(
            openai_api_key="test_key_12345",
            openai_model="gpt-3.5-turbo",
            api_timeout=30,
            max_retries=3
        )
        
        # Create temporary Excel file
        self.temp_dir = tempfile.mkdtemp()
        self.test_excel_path = os.path.join(self.temp_dir, "integration_test.xlsx")
        self._create_integration_test_file()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        if os.path.exists(self.test_excel_path):
            os.remove(self.test_excel_path)
        os.rmdir(self.temp_dir)
    
    def _create_integration_test_file(self):
        """Create a comprehensive test Excel file."""
        data = {
            'ID': [1, 2, 3],
            'Feedback': [
                'This product exceeded my expectations! Excellent quality.',
                'Disappointing experience. Would not recommend.',
                'Average product, nothing special but works fine.'
            ],
            'Rating': [5, 2, 3],
            'Category': ['Product', 'Service', 'Product']
        }
        df = pd.DataFrame(data)
        df.to_excel(self.test_excel_path, index=False)
    
    @patch('src.sentiment_analyzer.OpenAIClient')
    def test_complete_analysis_flow(self, mock_openai_client_class):
        """Test the complete analysis flow from Excel to insights."""
        # Mock OpenAI client with realistic responses
        mock_client = Mock()
        mock_openai_client_class.return_value = mock_client
        
        mock_results = [
            SentimentResult(
                original_text="This product exceeded my expectations! Excellent quality.",
                sentiment="positive",
                confidence=0.95,
                reasoning="Strong positive language with words like 'exceeded expectations' and 'excellent'",
                processing_time=1.2,
                success=True
            ),
            SentimentResult(
                original_text="Disappointing experience. Would not recommend.",
                sentiment="negative",
                confidence=0.88,
                reasoning="Clear negative sentiment with 'disappointing' and 'would not recommend'",
                processing_time=1.1,
                success=True
            ),
            SentimentResult(
                original_text="Average product, nothing special but works fine.",
                sentiment="neutral",
                confidence=0.75,
                reasoning="Balanced language indicating neutral sentiment",
                processing_time=1.0,
                success=True
            )
        ]
        
        mock_client.analyze_sentiment.side_effect = mock_results
        mock_client.get_client_stats.return_value = {
            "total_requests": 3,
            "successful_requests": 3,
            "failed_requests": 0,
            "success_rate_percent": 100.0,
            "rate_limit_hits": 0,
            "timeout_errors": 0,
            "consecutive_failures": 0,
            "current_rate_limit_delay": 0.1,
            "average_request_time": 1.1,
            "model": "gpt-3.5-turbo",
            "timeout": 30,
            "max_retries": 3,
            "last_request_time": 0.0,
            "last_rate_limit_time": 0.0
        }
        
        # Create analyzer and run complete flow
        analyzer = SentimentAnalyzer(self.config)
        
        # Ensure the analyzer uses the mocked client
        analyzer.openai_client = mock_client
        
        # Step 1: Process responses
        analysis_results = analyzer.process_responses(self.test_excel_path)
        
        # Verify analysis results
        self.assertIsInstance(analysis_results, AnalysisResults)
        self.assertEqual(len(analysis_results.individual_results), 3)
        self.assertEqual(analysis_results.success_rate, 1.0)
        
        # Step 2: Generate insights
        summary = analyzer.generate_insights(analysis_results)
        
        # Verify summary
        self.assertIsInstance(summary, Summary)
        self.assertEqual(summary.sentiment_distribution["positive"], 1)
        self.assertEqual(summary.sentiment_distribution["negative"], 1)
        self.assertEqual(summary.sentiment_distribution["neutral"], 1)
        
        # Verify percentages
        for sentiment, percentage in summary.sentiment_percentages.items():
            self.assertAlmostEqual(percentage, 33.33, places=1)
        
        # Verify insights and recommendations exist
        self.assertGreater(len(summary.key_insights), 0)
        self.assertGreater(len(summary.recommendations), 0)
        self.assertGreater(len(summary.common_themes), 0)
        
        # Step 3: Verify analyzer state
        current_analysis = analyzer.get_current_analysis()
        self.assertEqual(current_analysis, analysis_results)
        
        progress_tracker = analyzer.get_progress_tracker()
        self.assertIsNotNone(progress_tracker)
        self.assertTrue(progress_tracker.is_complete())


class TestSummaryCalculations(unittest.TestCase):
    """Specific tests for summary calculations and insights generation."""
    
    def setUp(self):
        """Set up test fixtures for summary calculations."""
        self.config = Configuration(
            openai_api_key="test_key_12345",
            openai_model="gpt-3.5-turbo",
            api_timeout=30,
            max_retries=3
        )
        self.analyzer = SentimentAnalyzer(self.config)
    
    def test_sentiment_distribution_calculation(self):
        """Test accurate sentiment distribution calculation."""
        # Create mock results with known distribution
        mock_results = [
            SentimentResult("Text 1", "positive", 0.9, "Positive reasoning", 1.0, True),
            SentimentResult("Text 2", "positive", 0.8, "Positive reasoning", 1.0, True),
            SentimentResult("Text 3", "negative", 0.7, "Negative reasoning", 1.0, True),
            SentimentResult("Text 4", "neutral", 0.6, "Neutral reasoning", 1.0, True),
            SentimentResult("Text 5", "neutral", 0.0, "Failed", 1.0, False)  # Failed analysis
        ]
        
        # Calculate summary stats
        stats = self.analyzer._calculate_summary_stats(mock_results)
        
        # Verify sentiment distribution (only successful analyses)
        self.assertEqual(stats["sentiment_distribution"]["positive"], 2)
        self.assertEqual(stats["sentiment_distribution"]["negative"], 1)
        self.assertEqual(stats["sentiment_distribution"]["neutral"], 1)
        
        # Verify percentages (should be based on 4 successful analyses)
        self.assertAlmostEqual(stats["sentiment_percentages"]["positive"], 50.0, places=1)
        self.assertAlmostEqual(stats["sentiment_percentages"]["negative"], 25.0, places=1)
        self.assertAlmostEqual(stats["sentiment_percentages"]["neutral"], 25.0, places=1)
        
        # Verify total percentage sums to 100
        total_pct = sum(stats["sentiment_percentages"].values())
        self.assertAlmostEqual(total_pct, 100.0, places=1)
    
    def test_confidence_statistics_calculation(self):
        """Test confidence statistics calculation."""
        mock_results = [
            SentimentResult("Text 1", "positive", 0.9, "Reasoning", 1.0, True),
            SentimentResult("Text 2", "positive", 0.8, "Reasoning", 1.0, True),
            SentimentResult("Text 3", "negative", 0.6, "Reasoning", 1.0, True),
            SentimentResult("Text 4", "neutral", 0.4, "Reasoning", 1.0, True),
            SentimentResult("Text 5", "neutral", 0.0, "Failed", 1.0, False)  # Failed - should be excluded
        ]
        
        stats = self.analyzer._calculate_summary_stats(mock_results)
        confidence_stats = stats["confidence_stats"]
        
        # Expected values from successful analyses: [0.9, 0.8, 0.6, 0.4]
        self.assertAlmostEqual(confidence_stats["mean"], 0.675, places=3)  # (0.9+0.8+0.6+0.4)/4
        self.assertAlmostEqual(confidence_stats["median"], 0.7, places=3)  # median of [0.4,0.6,0.8,0.9]
        self.assertEqual(confidence_stats["min"], 0.4)
        self.assertEqual(confidence_stats["max"], 0.9)
    
    def test_common_themes_extraction(self):
        """Test common themes extraction from sentiment results."""
        mock_results = [
            SentimentResult("Great product", "positive", 0.9, "Positive language", 1.0, True),
            SentimentResult("Love it", "positive", 0.8, "Positive sentiment", 1.0, True),
            SentimentResult("Terrible service", "negative", 0.7, "Negative feedback", 1.0, True),
            SentimentResult("It's okay", "neutral", 0.6, "Neutral response", 1.0, True),
            SentimentResult("Failed text", "neutral", 0.0, "Error", 1.0, False)
        ]
        
        themes = self.analyzer._extract_common_themes(mock_results)
        
        # Verify themes are generated
        self.assertIsInstance(themes, list)
        self.assertGreater(len(themes), 0)
        
        # Verify sentiment-based themes are included
        theme_text = " ".join(themes)
        self.assertIn("positive", theme_text.lower())
        self.assertIn("negative", theme_text.lower())
        self.assertIn("neutral", theme_text.lower())
        
        # Verify confidence-based themes
        self.assertTrue(any("confidence" in theme.lower() for theme in themes))
    
    def test_key_insights_generation(self):
        """Test key insights generation from analysis statistics."""
        # Create stats with predominantly positive sentiment
        stats = {
            "successful_analyses": 10,
            "sentiment_percentages": {"positive": 70.0, "negative": 20.0, "neutral": 10.0},
            "confidence_stats": {"mean": 0.85, "median": 0.8, "min": 0.6, "max": 0.95},
            "success_rate": 90.0,
            "processing_time_stats": {"average": 2.0}
        }
        
        mock_results = AnalysisResults(
            individual_results=[],
            summary_stats=stats,
            processing_metadata={},
            timestamp=datetime.now(),
            total_processed=10,
            success_rate=0.9
        )
        
        insights = self.analyzer._generate_key_insights(stats, mock_results)
        
        # Verify insights are generated
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)
        
        # Verify positive sentiment insight
        insights_text = " ".join(insights)
        self.assertIn("positive", insights_text.lower())
        
        # Verify confidence insight
        self.assertTrue(any("confidence" in insight.lower() for insight in insights))
        
        # Verify success rate insight
        self.assertTrue(any("success" in insight.lower() for insight in insights))
    
    def test_recommendations_generation(self):
        """Test actionable recommendations generation."""
        # Create stats with high negative sentiment
        stats = {
            "successful_analyses": 10,
            "sentiment_percentages": {"positive": 20.0, "negative": 60.0, "neutral": 20.0},
            "confidence_stats": {"mean": 0.45, "median": 0.4, "min": 0.2, "max": 0.8},
            "success_rate": 75.0,
            "failed_analyses": 3
        }
        
        mock_results = AnalysisResults(
            individual_results=[],
            summary_stats=stats,
            processing_metadata={},
            timestamp=datetime.now(),
            total_processed=10,
            success_rate=0.75
        )
        
        recommendations = self.analyzer._generate_recommendations(stats, mock_results)
        
        # Verify recommendations are generated
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Verify negative sentiment recommendation
        recommendations_text = " ".join(recommendations)
        self.assertIn("negative", recommendations_text.lower())
        
        # Verify confidence-based recommendation
        self.assertTrue(any("confidence" in rec.lower() for rec in recommendations))
        
        # Verify failed analysis recommendation
        self.assertTrue(any("failed" in rec.lower() for rec in recommendations))
    
    def test_empty_results_handling(self):
        """Test handling of empty results in summary calculations."""
        empty_results = []
        stats = self.analyzer._calculate_summary_stats(empty_results)
        
        # Verify empty results are handled gracefully
        self.assertEqual(stats["total_responses"], 0)
        self.assertEqual(stats["successful_analyses"], 0)
        self.assertEqual(stats["failed_analyses"], 0)
        self.assertEqual(stats["sentiment_distribution"]["positive"], 0)
        self.assertEqual(stats["sentiment_distribution"]["negative"], 0)
        self.assertEqual(stats["sentiment_distribution"]["neutral"], 0)
        
        # Verify percentages are set to equal values
        self.assertAlmostEqual(stats["sentiment_percentages"]["positive"], 33.33, places=1)
        self.assertAlmostEqual(stats["sentiment_percentages"]["negative"], 33.33, places=1)
        self.assertAlmostEqual(stats["sentiment_percentages"]["neutral"], 33.34, places=1)
    
    def test_all_failed_results_handling(self):
        """Test handling when all analyses fail."""
        failed_results = [
            SentimentResult("Text 1", "neutral", 0.0, "Failed", 1.0, False),
            SentimentResult("Text 2", "neutral", 0.0, "Failed", 1.0, False),
            SentimentResult("Text 3", "neutral", 0.0, "Failed", 1.0, False)
        ]
        
        stats = self.analyzer._calculate_summary_stats(failed_results)
        
        # Verify failed results are handled
        self.assertEqual(stats["total_responses"], 3)
        self.assertEqual(stats["successful_analyses"], 0)
        self.assertEqual(stats["failed_analyses"], 3)
        self.assertEqual(stats["success_rate"], 0.0)
        
        # Verify sentiment distribution is empty for successful analyses
        self.assertEqual(stats["sentiment_distribution"]["positive"], 0)
        self.assertEqual(stats["sentiment_distribution"]["negative"], 0)
        self.assertEqual(stats["sentiment_distribution"]["neutral"], 0)
        
        # Verify confidence stats are zero
        self.assertEqual(stats["confidence_stats"]["mean"], 0.0)
        self.assertEqual(stats["confidence_stats"]["max"], 0.0)
    
    def test_generate_insights_comprehensive(self):
        """Test comprehensive insights generation with various scenarios."""
        # Create a realistic mix of results
        mock_results = [
            SentimentResult("Excellent service!", "positive", 0.95, "Strong positive", 1.2, True),
            SentimentResult("Great experience", "positive", 0.88, "Positive feedback", 1.1, True),
            SentimentResult("Good quality", "positive", 0.82, "Positive sentiment", 1.0, True),
            SentimentResult("It's okay", "neutral", 0.65, "Neutral response", 0.9, True),
            SentimentResult("Not great", "negative", 0.75, "Negative sentiment", 1.3, True),
            SentimentResult("Terrible", "negative", 0.90, "Strong negative", 1.1, True),
            SentimentResult("Failed analysis", "neutral", 0.0, "Error occurred", 0.5, False)
        ]
        
        # Create analysis results
        analysis_results = AnalysisResults(
            individual_results=mock_results,
            summary_stats=self.analyzer._calculate_summary_stats(mock_results),
            processing_metadata={"source_file": "test.xlsx"},
            timestamp=datetime.now(),
            total_processed=7,
            success_rate=6/7  # 6 successful out of 7
        )
        
        # Generate insights
        summary = self.analyzer.generate_insights(analysis_results)
        
        # Verify Summary object structure
        self.assertIsInstance(summary, Summary)
        
        # Verify sentiment distribution (6 successful: 3 positive, 1 neutral, 2 negative)
        self.assertEqual(summary.sentiment_distribution["positive"], 3)
        self.assertEqual(summary.sentiment_distribution["negative"], 2)
        self.assertEqual(summary.sentiment_distribution["neutral"], 1)
        
        # Verify percentages sum to 100
        total_pct = sum(summary.sentiment_percentages.values())
        self.assertAlmostEqual(total_pct, 100.0, places=1)
        
        # Verify insights and recommendations exist
        self.assertGreater(len(summary.key_insights), 0)
        self.assertGreater(len(summary.recommendations), 0)
        self.assertGreater(len(summary.common_themes), 0)
        
        # Verify confidence stats are calculated correctly
        expected_confidences = [0.95, 0.88, 0.82, 0.65, 0.75, 0.90]  # Only successful ones
        expected_mean = sum(expected_confidences) / len(expected_confidences)
        self.assertAlmostEqual(summary.confidence_stats["mean"], expected_mean, places=2)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    unittest.main()