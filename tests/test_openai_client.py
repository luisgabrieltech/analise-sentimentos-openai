#!/usr/bin/env python3
"""
Unit tests for OpenAI API Client

This module contains comprehensive tests for the OpenAIClient class,
including mocked API responses and error scenarios.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import time
from dataclasses import dataclass
import openai

from src.openai_client import OpenAIClient, OpenAIError, RateLimitError, AuthenticationError, TimeoutError, APIResponse
from src.models import SentimentResult
from src.config_manager import Configuration


# Custom exception classes for testing
class MockRateLimitError(openai.RateLimitError):
    def __init__(self, message):
        self.message = message
        super(Exception, self).__init__(message)


class MockAPIError(openai.APIError):
    def __init__(self, message):
        self.message = message
        super(Exception, self).__init__(message)


class MockAuthenticationError(openai.AuthenticationError):
    def __init__(self, message):
        self.message = message
        super(Exception, self).__init__(message)


class MockTimeoutError(openai.APITimeoutError):
    def __init__(self, message):
        self.message = message
        super(Exception, self).__init__(message)


class MockConnectionError(openai.APIConnectionError):
    def __init__(self, message):
        self.message = message
        super(Exception, self).__init__(message)


class TestOpenAIClient(unittest.TestCase):
    """Test cases for OpenAIClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Configuration(
            openai_api_key="test-api-key",
            openai_org_id="test-org",
            openai_model="gpt-3.5-turbo",
            api_timeout=30,
            max_retries=3
        )
        
    @patch('src.openai_client.OpenAI')
    def test_client_initialization(self, mock_openai):
        """Test that the client initializes correctly."""
        client = OpenAIClient(self.config)
        
        # Verify OpenAI client was created with correct parameters
        mock_openai.assert_called_once_with(
            api_key="test-api-key",
            organization="test-org",
            timeout=30
        )
        
        # Verify configuration is stored
        self.assertEqual(client.config, self.config)
        
    @patch('src.openai_client.OpenAI')
    def test_analyze_sentiment_success(self, mock_openai):
        """Test successful sentiment analysis."""
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "sentiment": "positive",
            "confidence": 0.85,
            "reasoning": "The text expresses satisfaction and happiness"
        })
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("I love this product!")
        
        # Verify the result
        self.assertIsInstance(result, SentimentResult)
        self.assertEqual(result.original_text, "I love this product!")
        self.assertEqual(result.sentiment, "positive")
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.reasoning, "The text expresses satisfaction and happiness")
        self.assertTrue(result.success)
        self.assertIsNone(result.error_message)
        self.assertGreater(result.processing_time, 0)
        
    @patch('src.openai_client.OpenAI')
    def test_analyze_sentiment_empty_text(self, mock_openai):
        """Test analysis of empty text."""
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("")
        
        # Verify the result for empty text
        self.assertEqual(result.original_text, "")
        self.assertEqual(result.sentiment, "neutral")
        self.assertEqual(result.confidence, 0.0)
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Cannot analyze empty text")
        
    @patch('src.openai_client.OpenAI')
    def test_analyze_sentiment_whitespace_only(self, mock_openai):
        """Test analysis of whitespace-only text."""
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("   \n\t  ")
        
        # Verify the result for whitespace-only text
        self.assertEqual(result.original_text, "   \n\t  ")
        self.assertEqual(result.sentiment, "neutral")
        self.assertEqual(result.confidence, 0.0)
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Cannot analyze empty text")
        
    @patch('src.openai_client.OpenAI')
    def test_analyze_sentiment_negative(self, mock_openai):
        """Test negative sentiment analysis."""
        # Mock the API response for negative sentiment
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "sentiment": "negative",
            "confidence": 0.92,
            "reasoning": "The text expresses strong dissatisfaction and frustration"
        })
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("This is terrible and I hate it!")
        
        # Verify the result
        self.assertEqual(result.sentiment, "negative")
        self.assertEqual(result.confidence, 0.92)
        self.assertTrue(result.success)
        
    @patch('src.openai_client.OpenAI')
    def test_analyze_sentiment_neutral(self, mock_openai):
        """Test neutral sentiment analysis."""
        # Mock the API response for neutral sentiment
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "sentiment": "neutral",
            "confidence": 0.75,
            "reasoning": "The text is factual and doesn't express clear emotional sentiment"
        })
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("The meeting is scheduled for 3 PM.")
        
        # Verify the result
        self.assertEqual(result.sentiment, "neutral")
        self.assertEqual(result.confidence, 0.75)
        self.assertTrue(result.success)
        
    @patch('src.openai_client.OpenAI')
    def test_invalid_json_response(self, mock_openai):
        """Test handling of invalid JSON response."""
        # Mock the API response with invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is not valid JSON"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("Test text")
        
        # Verify the result handles invalid JSON
        self.assertFalse(result.success)
        self.assertIn("Failed to parse JSON response", result.error_message)
        
    @patch('src.openai_client.OpenAI')
    def test_invalid_sentiment_value(self, mock_openai):
        """Test handling of invalid sentiment value in response."""
        # Mock the API response with invalid sentiment
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "sentiment": "invalid_sentiment",
            "confidence": 0.85,
            "reasoning": "Test reasoning"
        })
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("Test text")
        
        # Verify the result handles invalid sentiment
        self.assertFalse(result.success)
        self.assertIn("Invalid sentiment value", result.error_message)
        
    @patch('src.openai_client.OpenAI')
    def test_invalid_confidence_value(self, mock_openai):
        """Test handling of invalid confidence value in response."""
        # Mock the API response with invalid confidence
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "sentiment": "positive",
            "confidence": 1.5,  # Invalid: > 1.0
            "reasoning": "Test reasoning"
        })
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("Test text")
        
        # Verify the result handles invalid confidence
        self.assertFalse(result.success)
        self.assertIn("Invalid confidence value", result.error_message)
        
    @patch('src.openai_client.OpenAI')
    @patch('src.openai_client.time.sleep')  # Mock sleep to speed up tests
    def test_rate_limit_error_with_retry(self, mock_sleep, mock_openai):
        """Test rate limit error handling with retry logic."""
        mock_client = Mock()
        # First call raises rate limit error, second succeeds
        mock_client.chat.completions.create.side_effect = [
            MockRateLimitError("Rate limit exceeded"),
            self._create_successful_response()
        ]
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("Test text")
        
        # Verify retry was attempted and eventually succeeded
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        self.assertTrue(result.success)
        # Verify backoff delay was used (may be called multiple times due to rate limiting logic)
        self.assertGreater(mock_sleep.call_count, 0)
        
    @patch('src.openai_client.OpenAI')
    @patch('src.openai_client.time.sleep')
    def test_rate_limit_error_max_retries_exceeded(self, mock_sleep, mock_openai):
        """Test rate limit error when max retries are exceeded."""
        mock_client = Mock()
        # All calls raise rate limit error
        mock_client.chat.completions.create.side_effect = MockRateLimitError("Rate limit exceeded")
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("Test text")
        
        # Verify all retries were attempted and failed
        self.assertEqual(mock_client.chat.completions.create.call_count, self.config.max_retries + 1)
        self.assertFalse(result.success)
        self.assertIn("Rate limit exceeded after all retries", result.error_message)
        
    @patch('src.openai_client.OpenAI')
    def test_authentication_error(self, mock_openai):
        """Test authentication error handling."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = MockAuthenticationError("Invalid API key")
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("Test text")
        
        # Verify authentication error is handled without retries
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)
        self.assertFalse(result.success)
        self.assertIn("Authentication failed", result.error_message)
        
    @patch('src.openai_client.OpenAI')
    @patch('src.openai_client.time.sleep')
    def test_api_error_with_retry(self, mock_sleep, mock_openai):
        """Test API error handling with retry logic."""
        mock_client = Mock()
        # First call raises API error, second succeeds
        mock_client.chat.completions.create.side_effect = [
            MockAPIError("Server error"),
            self._create_successful_response()
        ]
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("Test text")
        
        # Verify retry was attempted and eventually succeeded
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        self.assertTrue(result.success)
        
    def test_create_sentiment_prompt(self):
        """Test sentiment prompt creation."""
        client = OpenAIClient(self.config)
        text = "I love this product!"
        prompt = client._create_sentiment_prompt(text)
        
        # Verify prompt contains the text and instructions
        self.assertIn(text, prompt)
        self.assertIn("sentiment", prompt.lower())
        self.assertIn("confidence", prompt.lower())
        self.assertIn("reasoning", prompt.lower())
        self.assertIn("positive", prompt.lower())
        self.assertIn("negative", prompt.lower())
        self.assertIn("neutral", prompt.lower())
        
    def test_create_sentiment_prompt_long_text(self):
        """Test sentiment prompt creation with long text."""
        client = OpenAIClient(self.config)
        long_text = "A" * 3000  # Text longer than max_text_length
        prompt = client._create_sentiment_prompt(long_text)
        
        # Verify text was truncated
        self.assertIn("...", prompt)
        self.assertLess(len(prompt), len(long_text) + 1000)  # Should be much shorter
        
    def test_calculate_backoff_delay(self):
        """Test exponential backoff delay calculation."""
        client = OpenAIClient(self.config)
        
        # Test exponential increase (with jitter, so check ranges)
        delay0 = client._calculate_backoff_delay(0)
        delay1 = client._calculate_backoff_delay(1)
        delay2 = client._calculate_backoff_delay(2)
        
        # Base delays are 1, 2, 4 seconds with jitter (10-30% added)
        self.assertGreaterEqual(delay0, 1.0)
        self.assertLessEqual(delay0, 1.5)  # 1.0 + 30% jitter
        
        self.assertGreaterEqual(delay1, 2.0)
        self.assertLessEqual(delay1, 3.0)  # 2.0 + 30% jitter
        
        self.assertGreaterEqual(delay2, 4.0)
        self.assertLessEqual(delay2, 6.0)  # 4.0 + 30% jitter
        
        # Test maximum delay cap
        delay_large = client._calculate_backoff_delay(10)
        self.assertLessEqual(delay_large, 60.0)  # Should be capped at max_delay
        
    @patch('src.openai_client.OpenAI')
    def test_get_client_stats(self, mock_openai):
        """Test client statistics retrieval."""
        client = OpenAIClient(self.config)
        stats = client.get_client_stats()
        
        # Verify stats structure
        self.assertIn("total_requests", stats)
        self.assertIn("current_rate_limit_delay", stats)
        self.assertIn("model", stats)
        self.assertIn("timeout", stats)
        self.assertIn("max_retries", stats)
        
        # Verify initial values
        self.assertEqual(stats["total_requests"], 0)
        self.assertEqual(stats["model"], "gpt-3.5-turbo")
        self.assertEqual(stats["timeout"], 30)
        self.assertEqual(stats["max_retries"], 3)
        
    def _create_successful_response(self):
        """Helper method to create a successful mock response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "sentiment": "positive",
            "confidence": 0.85,
            "reasoning": "Test reasoning"
        })
        return mock_response


class TestAPIResponse(unittest.TestCase):
    """Test cases for APIResponse dataclass."""
    
    def test_api_response_creation(self):
        """Test APIResponse creation with all fields."""
        response = APIResponse(
            success=True,
            sentiment="positive",
            confidence=0.85,
            reasoning="Test reasoning",
            processing_time=1.5
        )
        
        self.assertTrue(response.success)
        self.assertEqual(response.sentiment, "positive")
        self.assertEqual(response.confidence, 0.85)
        self.assertEqual(response.reasoning, "Test reasoning")
        self.assertEqual(response.processing_time, 1.5)
        self.assertIsNone(response.error_message)
        
    def test_api_response_failure(self):
        """Test APIResponse creation for failure case."""
        response = APIResponse(
            success=False,
            error_message="Test error"
        )
        
        self.assertFalse(response.success)
        self.assertEqual(response.error_message, "Test error")
        self.assertIsNone(response.sentiment)
        self.assertIsNone(response.confidence)
        self.assertIsNone(response.reasoning)


class TestRetryLogicIntegration(unittest.TestCase):
    """Integration tests for retry logic and rate limiting scenarios."""
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.config = Configuration(
            openai_api_key="test-api-key",
            openai_org_id="test-org",
            openai_model="gpt-3.5-turbo",
            api_timeout=5,  # Shorter timeout for faster tests
            max_retries=3
        )
        
    @patch('src.openai_client.OpenAI')
    @patch('src.openai_client.time.sleep')
    def test_exponential_backoff_timing(self, mock_sleep, mock_openai):
        """Test that exponential backoff delays increase correctly."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = MockAPIError("Server error")
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("Test text")
        
        # Verify exponential backoff was used
        self.assertFalse(result.success)
        self.assertEqual(mock_client.chat.completions.create.call_count, 4)  # Initial + 3 retries
        
        # Check that sleep was called with increasing delays (with jitter)
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list if call[0]]
        self.assertGreater(len(sleep_calls), 0)
        
        # Verify delays are generally increasing (accounting for jitter)
        # With jitter, we can't guarantee strict ordering, but the trend should be upward
        if len(sleep_calls) >= 3:
            # Check that later delays are generally larger than earlier ones
            # Allow for jitter by checking if the average of later delays > average of earlier delays
            early_delays = sleep_calls[:len(sleep_calls)//2]
            late_delays = sleep_calls[len(sleep_calls)//2:]
            avg_early = sum(early_delays) / len(early_delays)
            avg_late = sum(late_delays) / len(late_delays)
            self.assertGreater(avg_late, avg_early, "Later delays should generally be larger than earlier ones")
    
    @patch('src.openai_client.OpenAI')
    @patch('src.openai_client.time.sleep')
    def test_rate_limit_specific_backoff(self, mock_sleep, mock_openai):
        """Test that rate limit errors use longer backoff delays."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = MockRateLimitError("Rate limit exceeded")
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("Test text")
        
        # Verify rate limit handling
        self.assertFalse(result.success)
        self.assertIn("Rate limit exceeded", result.error_message)
        
        # Rate limit delays should be longer than regular backoff
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list if call[0]]
        if sleep_calls:
            # Rate limit delays should start at 5+ seconds
            self.assertGreaterEqual(sleep_calls[0], 5.0)
    
    @patch('src.openai_client.OpenAI')
    @patch('src.openai_client.time.sleep')
    def test_timeout_error_handling(self, mock_sleep, mock_openai):
        """Test timeout error handling and retry logic."""
        mock_client = Mock()
        # First two calls timeout, third succeeds
        mock_client.chat.completions.create.side_effect = [
            MockTimeoutError("Request timed out"),
            MockTimeoutError("Request timed out"),
            self._create_successful_response()
        ]
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("Test text")
        
        # Should eventually succeed after timeouts
        self.assertTrue(result.success)
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)
        
        # Verify timeout statistics
        stats = client.get_client_stats()
        self.assertEqual(stats['timeout_errors'], 2)
        self.assertEqual(stats['successful_requests'], 1)
    
    @patch('src.openai_client.OpenAI')
    @patch('src.openai_client.time.sleep')
    def test_connection_error_retry(self, mock_sleep, mock_openai):
        """Test connection error handling with retry."""
        mock_client = Mock()
        # Connection error then success
        mock_client.chat.completions.create.side_effect = [
            MockConnectionError("Connection failed"),
            self._create_successful_response()
        ]
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("Test text")
        
        # Should succeed after connection error
        self.assertTrue(result.success)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
    
    @patch('src.openai_client.OpenAI')
    @patch('src.openai_client.time.sleep')
    def test_mixed_error_scenarios(self, mock_sleep, mock_openai):
        """Test handling of mixed error types in sequence."""
        mock_client = Mock()
        # Mix of different errors then success
        mock_client.chat.completions.create.side_effect = [
            MockRateLimitError("Rate limit"),
            MockTimeoutError("Timeout"),
            MockAPIError("Server error"),
            self._create_successful_response()
        ]
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        result = client.analyze_sentiment("Test text")
        
        # Should eventually succeed
        self.assertTrue(result.success)
        self.assertEqual(mock_client.chat.completions.create.call_count, 4)
        
        # Verify statistics track different error types
        stats = client.get_client_stats()
        self.assertEqual(stats['rate_limit_hits'], 1)
        self.assertEqual(stats['timeout_errors'], 1)
        self.assertEqual(stats['successful_requests'], 1)
    
    @patch('src.openai_client.OpenAI')
    def test_adaptive_rate_limiting(self, mock_openai):
        """Test adaptive rate limiting based on request patterns."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = self._create_successful_response()
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        
        # Simulate conditions that should trigger enhanced rate limiting
        client._consecutive_failures = 3
        client._rate_limit_hits = 2
        client._last_rate_limit_time = time.time() - 30  # Recent rate limit
        client._rate_limit_delay = 2.0  # Increased delay
        
        # Make a request and verify rate limiting is applied
        with patch('src.openai_client.time.sleep') as mock_sleep:
            # Set last request time to very recent to trigger rate limiting
            client._last_request_time = time.time() - 0.01  # Very recent request
            result = client.analyze_sentiment("Test text")
            
            # Should have applied enhanced rate limiting
            self.assertTrue(result.success)
            # Should have slept due to adaptive rate limiting (at least once)
            self.assertGreaterEqual(mock_sleep.call_count, 1)
    
    @patch('src.openai_client.OpenAI')
    def test_comprehensive_statistics_tracking(self, mock_openai):
        """Test comprehensive statistics tracking across multiple requests."""
        mock_client = Mock()
        # Mix of successes and failures
        mock_client.chat.completions.create.side_effect = [
            self._create_successful_response(),
            MockRateLimitError("Rate limit"),
            self._create_successful_response(),
            MockTimeoutError("Timeout"),
            self._create_successful_response()
        ]
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        
        # Make multiple requests
        results = []
        with patch('src.openai_client.time.sleep'):
            for i in range(5):
                result = client.analyze_sentiment(f"Test text {i}")
                results.append(result)
        
        # Verify statistics
        stats = client.get_client_stats()
        
        # Should have made more requests due to retries
        self.assertGreater(stats['total_requests'], 5)
        self.assertEqual(stats['successful_requests'], 3)  # 3 successful analyses
        self.assertEqual(stats['rate_limit_hits'], 1)
        self.assertEqual(stats['timeout_errors'], 1)
        self.assertGreater(stats['success_rate_percent'], 0)
        
        # Test stats reset
        client.reset_stats()
        reset_stats = client.get_client_stats()
        self.assertEqual(reset_stats['total_requests'], 0)
        self.assertEqual(reset_stats['successful_requests'], 0)
    
    @patch('src.openai_client.OpenAI')
    def test_performance_logging(self, mock_openai):
        """Test performance logging functionality."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = self._create_successful_response()
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        
        # Capture log output
        with patch.object(client.logger, 'info') as mock_log:
            client.log_performance_summary()
            
            # Verify logging was called
            self.assertGreater(mock_log.call_count, 0)
            
            # Check that performance metrics were logged
            log_calls = [str(call) for call in mock_log.call_args_list]
            log_output = ' '.join(log_calls)
            self.assertIn('Performance Summary', log_output)
            self.assertIn('Total requests', log_output)
            self.assertIn('Success rate', log_output)
    
    @patch('src.openai_client.OpenAI')
    @patch('src.openai_client.time.sleep')
    def test_jitter_in_backoff(self, mock_sleep, mock_openai):
        """Test that jitter is applied to backoff delays to prevent thundering herd."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = MockAPIError("Server error")
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(self.config)
        
        # Run multiple times to test jitter variance
        delays = []
        for _ in range(5):
            client.reset_stats()  # Reset for each test
            with patch('src.openai_client.time.sleep') as mock_sleep_inner:
                client.analyze_sentiment("Test text")
                if mock_sleep_inner.call_args_list:
                    # Get first delay (attempt 0)
                    first_delay = mock_sleep_inner.call_args_list[0][0][0]
                    delays.append(first_delay)
        
        # Verify we have some variance in delays (jitter working)
        if len(delays) > 1:
            unique_delays = set(delays)
            # Should have some variation due to jitter
            self.assertGreater(len(unique_delays), 1, "Jitter should create variation in delays")
    
    def _create_successful_response(self):
        """Helper method to create a successful mock response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "sentiment": "positive",
            "confidence": 0.85,
            "reasoning": "Test reasoning"
        })
        return mock_response


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main()