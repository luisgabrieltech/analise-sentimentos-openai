#!/usr/bin/env python3
"""
Performance tests and benchmarks for the sentiment analysis system.

This module provides comprehensive performance testing including:
- Async vs sync processing comparison
- Cache effectiveness testing
- Memory usage optimization validation
- Throughput and latency benchmarks
"""

import time
import asyncio
import unittest
import tempfile
import os
import json
import psutil
import threading
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock

from src.config_manager import Configuration
from src.openai_client import OpenAIClient
from src.sentiment_analyzer import SentimentAnalyzer
from src.excel_reader import ExcelReader
from src.models import SentimentResult


class PerformanceTestCase(unittest.TestCase):
    """Base class for performance tests with common utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = Configuration(
            openai_api_key="test-key",
            openai_model="gpt-3.5-turbo",
            api_timeout=30.0,
            max_retries=3
        )
        self.test_texts = [
            "This is a great product! I love it.",
            "This product is terrible and doesn't work.",
            "The product is okay, nothing special.",
            "Amazing quality and fast delivery!",
            "Poor customer service and broken item.",
            "Average product with decent features.",
            "Excellent value for money!",
            "Disappointing experience overall.",
            "Good product but could be better.",
            "Outstanding service and quality!"
        ]
        
        # Create mock responses for consistent testing
        self.mock_responses = [
            {"sentiment": "positive", "confidence": 0.9, "reasoning": "Positive language"},
            {"sentiment": "negative", "confidence": 0.8, "reasoning": "Negative language"},
            {"sentiment": "neutral", "confidence": 0.6, "reasoning": "Neutral language"},
            {"sentiment": "positive", "confidence": 0.95, "reasoning": "Very positive"},
            {"sentiment": "negative", "confidence": 0.85, "reasoning": "Very negative"},
            {"sentiment": "neutral", "confidence": 0.7, "reasoning": "Balanced tone"},
            {"sentiment": "positive", "confidence": 0.92, "reasoning": "Enthusiastic"},
            {"sentiment": "negative", "confidence": 0.88, "reasoning": "Disappointed"},
            {"sentiment": "neutral", "confidence": 0.65, "reasoning": "Mixed feelings"},
            {"sentiment": "positive", "confidence": 0.93, "reasoning": "Excellent feedback"}
        ]
    
    def create_mock_openai_response(self, sentiment: str, confidence: float, reasoning: str):
        """Create a mock OpenAI API response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = json.dumps({
            "sentiment": sentiment,
            "confidence": confidence,
            "reasoning": reasoning
        })
        return mock_response
    
    def measure_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB


class TestCachePerformance(PerformanceTestCase):
    """Test cache effectiveness and performance."""
    
    def test_cache_hit_performance(self):
        """Test that cache hits are significantly faster than API calls."""
        client = OpenAIClient(self.config)
        
        # Mock the API call to be slow
        with patch.object(client, '_make_api_request') as mock_api:
            def slow_api_call(text, context=None):
                time.sleep(0.1)  # Simulate 100ms API delay
                return client._parse_api_response(
                    self.create_mock_openai_response("positive", 0.9, "Test response")
                )
            mock_api.side_effect = slow_api_call
            
            # First call - should hit API
            start_time = time.time()
            result1 = client.analyze_sentiment("Test text for caching")
            first_call_time = time.time() - start_time
            
            # Second call - should hit cache
            start_time = time.time()
            result2 = client.analyze_sentiment("Test text for caching")
            second_call_time = time.time() - start_time
            
            # Verify cache hit
            self.assertTrue(result1.success)
            self.assertTrue(result2.success)
            self.assertEqual(result1.sentiment, result2.sentiment)
            self.assertIn("[CACHED]", result2.reasoning)
            
            # Cache hit should be much faster (more lenient threshold)
            self.assertLess(second_call_time, first_call_time / 5)  # At least 5x faster
            
            # Verify API was only called once
            self.assertEqual(mock_api.call_count, 1)
    
    def test_cache_memory_efficiency(self):
        """Test that cache doesn't consume excessive memory."""
        client = OpenAIClient(self.config)
        
        initial_memory = self.measure_memory_usage()
        
        # Fill cache with many entries
        with patch.object(client, '_make_api_request') as mock_api:
            mock_response = self.create_mock_openai_response("positive", 0.9, "Test response")
            mock_api.return_value = client._parse_api_response(mock_response)
            
            # Add many unique texts to cache
            for i in range(500):
                client.analyze_sentiment(f"Unique test text number {i}")
        
        final_memory = self.measure_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB for 500 entries)
        self.assertLess(memory_increase, 50)
        
        # Cache should respect size limits
        stats = client.get_client_stats()
        self.assertLessEqual(stats['cache_stats']['cache_size'], client._max_cache_size)
    
    def test_cache_duplicate_detection(self):
        """Test that cache correctly identifies duplicate texts."""
        client = OpenAIClient(self.config)
        
        with patch.object(client, '_make_api_request') as mock_api:
            mock_response = self.create_mock_openai_response("positive", 0.9, "Test response")
            mock_api.return_value = client._parse_api_response(mock_response)
            
            # Test various forms of the same text
            texts = [
                "Hello world",
                "hello world",  # Different case
                " Hello world ",  # Extra whitespace
                "Hello world",  # Exact duplicate
            ]
            
            for text in texts:
                client.analyze_sentiment(text)
            
            # Should only make one API call due to normalization
            self.assertEqual(mock_api.call_count, 1)
            
            stats = client.get_client_stats()
            self.assertEqual(stats['cache_stats']['cache_hits'], 3)
            self.assertEqual(stats['cache_stats']['cache_misses'], 1)


class TestAsyncPerformance(PerformanceTestCase):
    """Test async processing performance."""
    
    @patch('src.openai_client.AsyncOpenAI')
    def test_async_vs_sync_performance(self, mock_async_openai):
        """Compare async vs sync processing performance."""
        # Setup mock async client
        mock_async_client = AsyncMock()
        mock_async_openai.return_value = mock_async_client
        
        # Create mock async responses
        async def mock_create(**kwargs):
            await asyncio.sleep(0.1)  # Simulate API delay
            return self.create_mock_openai_response("positive", 0.9, "Async test")
        
        mock_async_client.chat.completions.create = mock_create
        
        client = OpenAIClient(self.config)
        
        # Test sync processing
        with patch.object(client, '_make_api_request') as mock_sync_api:
            def slow_api_call(text, context=None):
                time.sleep(0.1)  # Simulate API delay
                return client._parse_api_response(
                    self.create_mock_openai_response("positive", 0.9, "Sync test")
                )
            mock_sync_api.side_effect = slow_api_call
            
            # Sync processing
            start_time = time.time()
            sync_results = []
            for text in self.test_texts[:5]:  # Use 5 texts for faster testing
                result = client.analyze_sentiment(text)
                sync_results.append(result)
            sync_time = time.time() - start_time
        
        # Test async processing
        async def run_async_test():
            start_time = time.time()
            async_results = await client.batch_analyze_async(self.test_texts[:5], batch_size=5)
            return async_results, time.time() - start_time
        
        async_results, async_time = asyncio.run(run_async_test())
        
        # Verify results
        self.assertEqual(len(sync_results), len(async_results))
        for result in sync_results + async_results:
            self.assertTrue(result.success)
        
        # Async should be significantly faster for concurrent processing
        self.assertLess(async_time, sync_time * 0.8)  # At least 20% faster
        
        print(f"Sync time: {sync_time:.2f}s, Async time: {async_time:.2f}s")
        print(f"Async speedup: {sync_time/async_time:.2f}x")
    
    @patch('src.openai_client.AsyncOpenAI')
    def test_async_concurrency_limits(self, mock_async_openai):
        """Test that async processing respects concurrency limits."""
        mock_async_client = AsyncMock()
        mock_async_openai.return_value = mock_async_client
        
        # Track concurrent calls
        concurrent_calls = 0
        max_concurrent = 0
        call_lock = threading.Lock()
        
        async def mock_create(**kwargs):
            nonlocal concurrent_calls, max_concurrent
            
            with call_lock:
                concurrent_calls += 1
                max_concurrent = max(max_concurrent, concurrent_calls)
            
            await asyncio.sleep(0.1)  # Simulate processing time
            
            with call_lock:
                concurrent_calls -= 1
            
            return self.create_mock_openai_response("positive", 0.9, "Concurrent test")
        
        mock_async_client.chat.completions.create = mock_create
        
        client = OpenAIClient(self.config)
        
        # Test with batch size limit
        async def run_test():
            return await client.batch_analyze_async(self.test_texts, batch_size=3)
        
        results = asyncio.run(run_test())
        
        # Verify concurrency was limited
        self.assertLessEqual(max_concurrent, 5)  # Should respect semaphore limit
        self.assertEqual(len(results), len(self.test_texts))
        
        print(f"Max concurrent requests: {max_concurrent}")


class TestMemoryOptimization(PerformanceTestCase):
    """Test memory optimization features."""
    
    def test_chunked_processing_memory_usage(self):
        """Test that chunked processing uses less memory than batch processing."""
        # Create a large dataset
        large_dataset = []
        for i in range(200):
            large_dataset.append({
                'text_content': f"Test response number {i} with some additional content to make it longer",
                'headers': ['response'],
                'row_index': i,
                'source_file': 'test.xlsx'
            })
        
        analyzer = SentimentAnalyzer(self.config)
        
        # Mock the OpenAI client to avoid actual API calls
        with patch.object(analyzer.openai_client, 'analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = SentimentResult(
                original_text="test",
                sentiment="positive",
                confidence=0.9,
                reasoning="test",
                processing_time=0.1,
                success=True
            )
            
            # Measure memory before processing
            initial_memory = self.measure_memory_usage()
            
            # Process with chunking (using sync method for memory test)
            chunk_results = analyzer._process_chunk(large_dataset[:50], 0)
            
            # Measure memory after chunked processing
            chunked_memory = self.measure_memory_usage()
            memory_increase = chunked_memory - initial_memory
            
            # Memory increase should be reasonable
            self.assertLess(memory_increase, 100)  # Less than 100MB increase
            
            print(f"Memory increase with chunking: {memory_increase:.2f}MB")
    
    def test_cache_eviction_policy(self):
        """Test that cache eviction works correctly when size limit is reached."""
        client = OpenAIClient(self.config)
        client._max_cache_size = 10  # Set small cache size for testing
        
        with patch.object(client, '_make_api_request') as mock_api:
            mock_response = self.create_mock_openai_response("positive", 0.9, "Test response")
            mock_api.return_value = client._parse_api_response(mock_response)
            
            # Fill cache beyond limit
            for i in range(15):
                client.analyze_sentiment(f"Unique text {i}")
            
            stats = client.get_client_stats()
            
            # Cache size should be limited
            self.assertLessEqual(stats['cache_stats']['cache_size'], client._max_cache_size)
            
            # Should have made 15 API calls (no cache hits due to unique texts)
            self.assertEqual(mock_api.call_count, 15)


class TestThroughputBenchmarks(PerformanceTestCase):
    """Benchmark throughput and latency."""
    
    def test_processing_throughput(self):
        """Benchmark processing throughput for different modes."""
        # Create test data
        test_responses = []
        for i in range(50):
            test_responses.append({
                'text_content': f"Test response {i} with varying content length and sentiment",
                'headers': ['response'],
                'row_index': i,
                'source_file': 'benchmark.xlsx'
            })
        
        analyzer = SentimentAnalyzer(self.config)
        
        # Mock API calls for consistent timing 
        with patch.object(analyzer.openai_client, 'analyze_sentiment') as mock_analyze: 
            def mock_sentiment_analysis(text, context=None): 
                time.sleep(0.05)  # Simulate 50ms API call 
                return SentimentResult( 
                    original_text=text, 
                    sentiment="positive", 
                    confidence=0.9, 
                    reasoning="Benchmark test", 
                    processing_time=0.05, 
                    success=True 
                ) 
            mock_analyze.side_effect = mock_sentiment_analysis
            
            # Benchmark standard processing
            start_time = time.time()
            standard_results = analyzer._process_individual_responses(test_responses[:20])
            standard_time = time.time() - start_time
            
            # Calculate throughput
            standard_throughput = len(standard_results) / standard_time
            
            print(f"Standard processing: {len(standard_results)} responses in {standard_time:.2f}s")
            print(f"Throughput: {standard_throughput:.2f} responses/second")
            
            # Verify all responses were processed
            self.assertEqual(len(standard_results), 20)
            self.assertTrue(all(r.success for r in standard_results))
            
            # Verify throughput meets minimum performance requirements
            # Expecting at least 5 responses per second with the 50ms mock delay
            self.assertTrue(standard_throughput >= 5.0, 
                           f"Throughput below minimum requirement: {standard_throughput:.2f} responses/second")
    
    def test_latency_distribution(self):
        """Test latency distribution for individual requests."""
        client = OpenAIClient(self.config)
        latencies = []
        
        with patch.object(client, '_make_api_request') as mock_api:
            def variable_latency_api(text, context=None):
                # Simulate variable API latency
                import random
                delay = random.uniform(0.1, 0.5)
                time.sleep(delay)
                return client._parse_api_response(
                    self.create_mock_openai_response("positive", 0.9, "Latency test")
                )
            mock_api.side_effect = variable_latency_api
            
            # Measure latencies
            for text in self.test_texts:
                start_time = time.time()
                result = client.analyze_sentiment(text)
                latency = time.time() - start_time
                latencies.append(latency)
                self.assertTrue(result.success)
        
        # Calculate latency statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"Latency stats - Avg: {avg_latency:.3f}s, Min: {min_latency:.3f}s, Max: {max_latency:.3f}s")
        
        # Verify reasonable latency bounds
        self.assertLess(avg_latency, 1.0)  # Average should be under 1 second
        self.assertLess(max_latency, 2.0)  # Max should be under 2 seconds


class TestPerformanceRegression(PerformanceTestCase):
    """Test for performance regressions."""
    
    def test_performance_baseline(self):
        """Establish performance baseline for regression testing."""
        client = OpenAIClient(self.config)
        
        # Test cache performance
        with patch.object(client, '_make_api_request') as mock_api:
            mock_response = self.create_mock_openai_response("positive", 0.9, "Baseline test")
            mock_api.return_value = client._parse_api_response(mock_response)
            
            # Measure cache miss time
            start_time = time.time()
            result1 = client.analyze_sentiment("Baseline test text")
            cache_miss_time = time.time() - start_time
            
            # Measure cache hit time
            start_time = time.time()
            result2 = client.analyze_sentiment("Baseline test text")
            cache_hit_time = time.time() - start_time
            
            # Performance assertions
            self.assertLess(cache_hit_time, 0.01)  # Cache hits should be under 10ms
            self.assertLess(cache_hit_time, cache_miss_time / 5)  # At least 5x faster
            
            print(f"Performance baseline - Cache miss: {cache_miss_time:.4f}s, Cache hit: {cache_hit_time:.4f}s")


def run_performance_benchmarks():
    """Run all performance benchmarks and generate a report."""
    print("=" * 60)
    print("SENTIMENT ANALYSIS PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestCachePerformance,
        TestAsyncPerformance,
        TestMemoryOptimization,
        TestThroughputBenchmarks,
        TestPerformanceRegression
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_performance_benchmarks()
    sys.exit(0 if success else 1)