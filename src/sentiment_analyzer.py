#!/usr/bin/env python3
"""
Sentiment Analyzer - Main orchestration class for sentiment analysis

This module provides the main SentimentAnalyzer class that coordinates
the entire sentiment analysis pipeline, integrating Excel reading,
OpenAI API calls, progress tracking, and result generation.
"""

import time
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import SentimentResult, AnalysisResults, Summary, ProgressTracker
from .excel_reader import ExcelReader, ExcelReaderError
from .openai_client import OpenAIClient, OpenAIError
from .config_manager import Configuration


class SentimentAnalyzerError(Exception):
    """Custom exception for sentiment analyzer related errors."""
    pass


class SentimentAnalyzer:
    """
    Main orchestrator for sentiment analysis processing.
    
    This class coordinates the entire analysis pipeline:
    1. Loading responses from Excel files
    2. Processing each response through OpenAI API
    3. Tracking progress and handling errors
    4. Generating comprehensive insights and summaries
    """
    
    def __init__(self, config: Configuration):
        """
        Initialize the sentiment analyzer with configuration.
        
        Args:
            config: Configuration object containing API settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.excel_reader = ExcelReader()
        self.openai_client = OpenAIClient(config)
        self.progress_tracker: Optional[ProgressTracker] = None
        
        # Processing state
        self._current_analysis: Optional[AnalysisResults] = None
        self._processing_start_time: Optional[float] = None
    
    def process_responses_from_data(self, responses: List[Dict[str, Any]], file_path: str) -> AnalysisResults:
        """
        Process sentiment analysis from pre-loaded response data.
        
        Args:
            responses: List of response dictionaries already loaded
            file_path: Original file path for metadata
            
        Returns:
            AnalysisResults object with complete analysis data
        """
        self.logger.info(f"Starting sentiment analysis for {len(responses)} pre-loaded responses")
        self._processing_start_time = time.time()
        
        try:
            # Step 1: Initialize progress tracking
            self.progress_tracker = ProgressTracker(len(responses))
            self.logger.info("Initialized progress tracking")
            
            # Step 2: Process each response
            self.logger.info("Starting sentiment analysis processing...")
            individual_results = self._process_individual_responses(responses)
            
            # Step 3: Generate summary statistics
            self.logger.info("Generating summary statistics...")
            summary_stats = self._calculate_summary_stats(individual_results)
            
            # Step 4: Create processing metadata
            processing_metadata = self._create_processing_metadata(file_path, responses)
            
            # Step 5: Calculate success rate
            successful_count = sum(1 for result in individual_results if result.success)
            success_rate = successful_count / len(individual_results) if individual_results else 0.0
            
            # Step 6: Create final results object
            analysis_results = AnalysisResults(
                individual_results=individual_results,
                summary_stats=summary_stats,
                processing_metadata=processing_metadata,
                timestamp=datetime.now(),
                total_processed=len(individual_results),
                success_rate=success_rate
            )
            
            self._current_analysis = analysis_results
            
            # Log completion summary
            processing_time = time.time() - self._processing_start_time
            self.logger.info(f"Analysis completed in {processing_time:.2f}s")
            self.logger.info(f"Success rate: {success_rate:.2%} ({successful_count}/{len(individual_results)})")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Unexpected error during analysis: {e}")
            raise SentimentAnalyzerError(f"Analysis failed: {e}")

    def process_responses(self, file_path: str) -> AnalysisResults:
        """
        Main processing pipeline for sentiment analysis.
        
        This method orchestrates the complete analysis process:
        1. Load responses from Excel file
        2. Initialize progress tracking
        3. Process each response through OpenAI API
        4. Handle individual response failures gracefully
        5. Generate comprehensive results with metadata
        
        Args:
            file_path: Path to the Excel file containing responses
            
        Returns:
            AnalysisResults object with complete analysis data
            
        Raises:
            SentimentAnalyzerError: If the analysis process fails
        """
        self.logger.info(f"Starting sentiment analysis for file: {file_path}")
        self._processing_start_time = time.time()
        
        try:
            # Step 1: Load responses from Excel file
            self.logger.info("Loading responses from Excel file...")
            responses = self._load_responses(file_path)
            self.logger.info(f"Loaded {len(responses)} responses for analysis")
            
            # Step 2: Initialize progress tracking
            self.progress_tracker = ProgressTracker(len(responses))
            self.logger.info("Initialized progress tracking")
            
            # Step 3: Process each response
            self.logger.info("Starting sentiment analysis processing...")
            individual_results = self._process_individual_responses(responses)
            
            # Step 4: Generate summary statistics
            self.logger.info("Generating summary statistics...")
            summary_stats = self._calculate_summary_stats(individual_results)
            
            # Step 5: Create processing metadata
            processing_metadata = self._create_processing_metadata(file_path, responses)
            
            # Step 6: Calculate success rate
            successful_count = sum(1 for result in individual_results if result.success)
            success_rate = successful_count / len(individual_results) if individual_results else 0.0
            
            # Step 7: Create final results object
            analysis_results = AnalysisResults(
                individual_results=individual_results,
                summary_stats=summary_stats,
                processing_metadata=processing_metadata,
                timestamp=datetime.now(),
                total_processed=len(individual_results),
                success_rate=success_rate
            )
            
            self._current_analysis = analysis_results
            
            # Log completion summary
            processing_time = time.time() - self._processing_start_time
            self.logger.info(f"Analysis completed in {processing_time:.2f}s")
            self.logger.info(f"Success rate: {success_rate:.2%} ({successful_count}/{len(individual_results)})")
            
            return analysis_results
            
        except ExcelReaderError as e:
            self.logger.error(f"Excel reading failed: {e}")
            raise SentimentAnalyzerError(f"Failed to load Excel file: {e}")
        
        except Exception as e:
            self.logger.error(f"Unexpected error during analysis: {e}")
            raise SentimentAnalyzerError(f"Analysis failed: {e}")
    
    async def process_responses_async(self, file_path: str, batch_size: int = 10, enable_async: bool = True) -> AnalysisResults:
        """
        Async version of the main processing pipeline with concurrent processing.
        
        Args:
            file_path: Path to the Excel file containing responses
            batch_size: Number of concurrent requests to process at once
            enable_async: Whether to use async processing (for testing/comparison)
            
        Returns:
            AnalysisResults object with complete analysis data
        """
        self.logger.info(f"Starting async sentiment analysis for file: {file_path} (batch_size: {batch_size})")
        self._processing_start_time = time.time()
        
        try:
            # Step 1: Load responses from Excel file
            self.logger.info("Loading responses from Excel file...")
            responses = self._load_responses(file_path)
            self.logger.info(f"Loaded {len(responses)} responses for analysis")
            
            # Step 2: Initialize progress tracking
            self.progress_tracker = ProgressTracker(len(responses))
            self.logger.info("Initialized progress tracking")
            
            # Step 3: Process responses (async or sync based on flag)
            if enable_async and len(responses) > 1:
                self.logger.info("Starting async sentiment analysis processing...")
                individual_results = await self._process_responses_async_batch(responses, batch_size)
            else:
                self.logger.info("Starting synchronous processing...")
                individual_results = self._process_individual_responses(responses)
            
            # Step 4: Generate summary statistics
            self.logger.info("Generating summary statistics...")
            summary_stats = self._calculate_summary_stats(individual_results)
            
            # Step 5: Create processing metadata
            processing_metadata = self._create_processing_metadata(file_path, responses)
            
            # Step 6: Calculate success rate
            successful_count = sum(1 for result in individual_results if result.success)
            success_rate = successful_count / len(individual_results) if individual_results else 0.0
            
            # Step 7: Create final results object
            analysis_results = AnalysisResults(
                individual_results=individual_results,
                summary_stats=summary_stats,
                processing_metadata=processing_metadata,
                timestamp=datetime.now(),
                total_processed=len(individual_results),
                success_rate=success_rate
            )
            
            self._current_analysis = analysis_results
            
            # Log completion summary
            processing_time = time.time() - self._processing_start_time
            self.logger.info(f"Async analysis completed in {processing_time:.2f}s")
            self.logger.info(f"Success rate: {success_rate:.2%} ({successful_count}/{len(individual_results)})")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Unexpected error during async analysis: {e}")
            raise SentimentAnalyzerError(f"Async analysis failed: {e}")
    
    async def _process_responses_async_batch(self, responses: List[Dict[str, Any]], batch_size: int) -> List[SentimentResult]:
        """
        Process responses using async batch processing for improved performance.
        
        Args:
            responses: List of response dictionaries from Excel
            batch_size: Number of concurrent requests to process
            
        Returns:
            List of SentimentResult objects
        """
        self.logger.info(f"Processing {len(responses)} responses with async batching (batch_size: {batch_size})")
        
        # Extract texts for batch processing
        texts = [response['text_content'] for response in responses]
        
        # Use the OpenAI client's batch processing
        start_time = time.time()
        results = await self.openai_client.batch_analyze_async(texts, batch_size)
        processing_time = time.time() - start_time
        
        # Update progress tracker
        successful_count = sum(1 for r in results if r.success)
        for i, result in enumerate(results):
            self.progress_tracker.update_progress(
                success=result.success,
                processing_time=result.processing_time,
                current_item=result.original_text,
                status=f"Processed {i+1}/{len(results)}" if i < len(results) - 1 else "Complete"
            )
            
            # Show progress periodically
            if i % max(1, len(results) // 10) == 0 or i == len(results) - 1:
                progress_display = self.progress_tracker.display_progress(show_current_item=False)
                print(f"\n{progress_display}")
        
        # Log final statistics
        self.logger.info(
            f"Async batch processing completed in {processing_time:.2f}s: "
            f"{successful_count}/{len(results)} successful ({successful_count/len(results)*100:.1f}% success rate)"
        )
        
        return results
    
    def process_responses_memory_optimized(self, file_path: str, chunk_size: int = 100) -> AnalysisResults:
        """
        Memory-optimized processing for large datasets using chunked processing.
        
        Args:
            file_path: Path to the Excel file containing responses
            chunk_size: Number of responses to process in each chunk
            
        Returns:
            AnalysisResults object with complete analysis data
        """
        self.logger.info(f"Starting memory-optimized sentiment analysis for file: {file_path} (chunk_size: {chunk_size})")
        self._processing_start_time = time.time()
        
        try:
            # Step 1: Load responses from Excel file
            self.logger.info("Loading responses from Excel file...")
            responses = self._load_responses(file_path)
            total_responses = len(responses)
            self.logger.info(f"Loaded {total_responses} responses for chunked analysis")
            
            # Step 2: Initialize progress tracking
            self.progress_tracker = ProgressTracker(total_responses)
            self.logger.info("Initialized progress tracking for chunked processing")
            
            # Step 3: Process responses in chunks to optimize memory usage
            self.logger.info(f"Starting chunked processing with chunk size: {chunk_size}")
            all_results = []
            
            for chunk_start in range(0, total_responses, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_responses)
                chunk = responses[chunk_start:chunk_end]
                
                self.logger.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_responses + chunk_size - 1)//chunk_size} "
                               f"(responses {chunk_start + 1}-{chunk_end})")
                
                # Process this chunk
                chunk_results = self._process_chunk(chunk, chunk_start)
                all_results.extend(chunk_results)
                
                # Update progress
                for i, result in enumerate(chunk_results):
                    global_index = chunk_start + i
                    self.progress_tracker.update_progress(
                        success=result.success,
                        processing_time=result.processing_time,
                        current_item=result.original_text,
                        status=f"Processed {global_index + 1}/{total_responses}"
                    )
                
                # Show progress
                progress_display = self.progress_tracker.display_progress(show_current_item=False)
                print(f"\n{progress_display}")
                
                # Clear chunk from memory to optimize memory usage
                del chunk
                del chunk_results
                
                # Force garbage collection after each chunk
                import gc
                gc.collect()
            
            # Step 4: Generate summary statistics
            self.logger.info("Generating summary statistics...")
            summary_stats = self._calculate_summary_stats(all_results)
            
            # Step 5: Create processing metadata
            processing_metadata = self._create_processing_metadata(file_path, responses)
            processing_metadata['processing_mode'] = 'memory_optimized'
            processing_metadata['chunk_size'] = chunk_size
            processing_metadata['total_chunks'] = (total_responses + chunk_size - 1) // chunk_size
            
            # Step 6: Calculate success rate
            successful_count = sum(1 for result in all_results if result.success)
            success_rate = successful_count / len(all_results) if all_results else 0.0
            
            # Step 7: Create final results object
            analysis_results = AnalysisResults(
                individual_results=all_results,
                summary_stats=summary_stats,
                processing_metadata=processing_metadata,
                timestamp=datetime.now(),
                total_processed=len(all_results),
                success_rate=success_rate
            )
            
            self._current_analysis = analysis_results
            
            # Log completion summary
            processing_time = time.time() - self._processing_start_time
            self.logger.info(f"Memory-optimized analysis completed in {processing_time:.2f}s")
            self.logger.info(f"Success rate: {success_rate:.2%} ({successful_count}/{len(all_results)})")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Unexpected error during memory-optimized analysis: {e}")
            raise SentimentAnalyzerError(f"Memory-optimized analysis failed: {e}")
    
    def _process_chunk(self, chunk: List[Dict[str, Any]], chunk_start_index: int) -> List[SentimentResult]:
        """
        Process a single chunk of responses.
        
        Args:
            chunk: List of response dictionaries for this chunk
            chunk_start_index: Starting index of this chunk in the overall dataset
            
        Returns:
            List of SentimentResult objects for this chunk
        """
        chunk_results = []
        
        for i, response in enumerate(chunk):
            global_index = chunk_start_index + i
            try:
                result = self._analyze_single_response(response, global_index)
                chunk_results.append(result)
                
                # Log progress for this chunk
                if result.success:
                    self.logger.debug(f"Chunk response {i+1}: {result.sentiment} (confidence: {result.confidence:.2f})")
                else:
                    self.logger.warning(f"Chunk response {i+1} failed: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"Error processing chunk response {i+1}: {e}")
                error_result = SentimentResult(
                    original_text=response.get('text_content', ''),
                    sentiment="neutral",
                    confidence=0.0,
                    reasoning="Chunk processing error",
                    processing_time=0.0,
                    success=False,
                    error_message=f"Chunk error: {str(e)}"
                )
                chunk_results.append(error_result)
        
        return chunk_results
    
    def _load_responses(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load responses from Excel file with error handling.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            List of response dictionaries
            
        Raises:
            SentimentAnalyzerError: If loading fails
        """
        try:
            responses = self.excel_reader.load_responses(file_path)
            
            if not responses:
                raise SentimentAnalyzerError("No valid responses found in Excel file")
            
            # Log sample of loaded data for verification
            sample_response = responses[0]
            self.logger.info(f"Sample response preview: {sample_response['text_content'][:100]}...")
            self.logger.info(f"Data columns: {', '.join(sample_response['headers'])}")
            
            return responses
            
        except ExcelReaderError:
            raise  # Re-raise as-is
        except Exception as e:
            raise SentimentAnalyzerError(f"Unexpected error loading responses: {e}")
    
    def _process_individual_responses(self, responses: List[Dict[str, Any]]) -> List[SentimentResult]:
        """
        Process each response through the OpenAI API with progress tracking.
        
        Args:
            responses: List of response dictionaries from Excel
            
        Returns:
            List of SentimentResult objects
        """
        individual_results = []
        consecutive_failures = 0
        max_consecutive_failures = 10  # Stop if too many consecutive failures
        
        self.logger.info(f"Processing {len(responses)} responses...")
        
        for i, response in enumerate(responses):
            try:
                # Check for too many consecutive failures
                if consecutive_failures >= max_consecutive_failures:
                    self.logger.error(
                        f"Stopping processing after {consecutive_failures} consecutive failures. "
                        f"This may indicate a systemic issue."
                    )
                    
                    # Add failed results for remaining responses
                    for j in range(i, len(responses)):
                        failed_result = SentimentResult(
                            original_text=responses[j].get('text_content', ''),
                            sentiment="neutral",
                            confidence=0.0,
                            reasoning="Processing stopped due to consecutive failures",
                            processing_time=0.0,
                            success=False,
                            error_message="Processing stopped due to systemic failures"
                        )
                        individual_results.append(failed_result)
                    break
                
                # Process the response
                current_text = response['text_content']
                start_time = time.time()
                result = self._analyze_single_response(response, i)
                processing_time = time.time() - start_time
                
                # Reset consecutive failures on success
                if result.success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                
                # Update progress tracker with actual result
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        success=result.success,
                        processing_time=processing_time,
                        current_item=current_text,
                        status=f"Analyzing response {i+1}/{len(responses)}" if i < len(responses) - 1 else "Complete"
                    )
                    
                    # Display progress
                    if i % 5 == 0 or i == len(responses) - 1:  # Show progress every 5 items and at the end
                        progress_display = self.progress_tracker.display_progress(show_current_item=False)
                        print(f"\n{progress_display}")
                
                individual_results.append(result)
                
                # Log individual result
                if result.success:
                    self.logger.debug(
                        f"Response {i+1}: {result.sentiment} "
                        f"(confidence: {result.confidence:.2f}) - {processing_time:.2f}s"
                    )
                else:
                    self.logger.warning(
                        f"Response {i+1} failed: {result.error_message} - {processing_time:.2f}s"
                    )
                    
                    # Log pattern of failures for debugging
                    if consecutive_failures > 3:
                        self.logger.warning(
                            f"Consecutive failures: {consecutive_failures}. "
                            f"This may indicate an API or configuration issue."
                        )
                
            except KeyboardInterrupt:
                self.logger.info("Processing interrupted by user")
                print("\n⚠️  Processing interrupted by user. Saving partial results...")
                break
                
            except Exception as e:
                # Handle individual response processing errors gracefully
                consecutive_failures += 1
                self.logger.error(f"Error processing response {i+1}: {e}", exc_info=True)
                
                # Create failed result with detailed error information
                error_details = f"Unexpected error: {str(e)}"
                if hasattr(e, '__class__'):
                    error_details = f"{e.__class__.__name__}: {str(e)}"
                
                failed_result = SentimentResult(
                    original_text=response.get('text_content', ''),
                    sentiment="neutral",
                    confidence=0.0,
                    reasoning="Processing failed due to unexpected error",
                    processing_time=0.0,
                    success=False,
                    error_message=error_details
                )
                
                individual_results.append(failed_result)
                
                # Update progress tracker
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        success=False,
                        current_item=response.get('text_content', ''),
                        status="Error occurred"
                    )
        
        # Final progress display
        if self.progress_tracker:
            final_progress = self.progress_tracker.display_progress(show_current_item=False)
            print(f"\n{final_progress}")
        
        # Log final statistics
        successful_count = sum(1 for r in individual_results if r.success)
        self.logger.info(
            f"Processing completed: {successful_count}/{len(individual_results)} successful "
            f"({successful_count/len(individual_results)*100:.1f}% success rate)"
        )
        
        return individual_results
    
    def _analyze_single_response(self, response: Dict[str, Any], index: int) -> SentimentResult:
        """
        Analyze a single response using the OpenAI client.
        
        Args:
            response: Response dictionary from Excel
            index: Index of the response for logging
            
        Returns:
            SentimentResult object with analysis results
        """
        text_content = response.get('text_content', '')
        
        if not text_content or not text_content.strip():
            return SentimentResult(
                original_text=text_content,
                sentiment="neutral",
                confidence=0.0,
                reasoning="Empty or whitespace-only text",
                processing_time=0.0,
                success=False,
                error_message="Cannot analyze empty text"
            )
        
        try:
            # Analyze sentiment using OpenAI client
            result = self.openai_client.analyze_sentiment(text_content)
            
            # Add additional metadata from Excel response
            if hasattr(result, 'excel_metadata'):
                result.excel_metadata = {
                    'row_index': response.get('row_index'),
                    'source_file': response.get('source_file'),
                    'headers': response.get('headers')
                }
            
            return result
            
        except OpenAIError as e:
            self.logger.warning(f"OpenAI API error for response {index+1}: {e}")
            return SentimentResult(
                original_text=text_content,
                sentiment="neutral",
                confidence=0.0,
                reasoning="API error occurred",
                processing_time=0.0,
                success=False,
                error_message=f"OpenAI API error: {str(e)}"
            )
        
        except Exception as e:
            self.logger.error(f"Unexpected error analyzing response {index+1}: {e}")
            return SentimentResult(
                original_text=text_content,
                sentiment="neutral",
                confidence=0.0,
                reasoning="Unexpected error occurred",
                processing_time=0.0,
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def _calculate_summary_stats(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """
        Calculate comprehensive summary statistics from analysis results.
        
        Args:
            results: List of SentimentResult objects
            
        Returns:
            Dictionary containing summary statistics
        """
        if not results:
            return {
                "total_responses": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                "sentiment_percentages": {"positive": 33.33, "negative": 33.33, "neutral": 33.34},
                "confidence_stats": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
                "processing_time_stats": {"total": 0.0, "average": 0.0, "min": 0.0, "max": 0.0}
            }
        
        # Basic counts
        total_responses = len(results)
        successful_results = [r for r in results if r.success]
        successful_count = len(successful_results)
        failed_count = total_responses - successful_count
        
        # Sentiment distribution (only from successful analyses)
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for result in successful_results:
            sentiment_counts[result.sentiment] += 1
        
        # Calculate percentages
        sentiment_percentages = {}
        if successful_count > 0:
            for sentiment, count in sentiment_counts.items():
                sentiment_percentages[sentiment] = (count / successful_count) * 100.0
        else:
            # When no successful results, set equal percentages that sum to 100
            sentiment_percentages = {"positive": 33.33, "negative": 33.33, "neutral": 33.34}
        
        # Confidence statistics (only from successful analyses)
        confidence_values = [r.confidence for r in successful_results]
        if confidence_values:
            sorted_values = sorted(confidence_values)
            n = len(sorted_values)
            if n % 2 == 0:
                # Even number of values - take average of middle two
                median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
            else:
                # Odd number of values - take middle value
                median = sorted_values[n // 2]
            
            confidence_stats = {
                "mean": sum(confidence_values) / len(confidence_values),
                "median": median,
                "min": min(confidence_values),
                "max": max(confidence_values)
            }
        else:
            confidence_stats = {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        
        # Processing time statistics
        processing_times = [r.processing_time for r in results]
        processing_time_stats = {
            "total": sum(processing_times),
            "average": sum(processing_times) / len(processing_times) if processing_times else 0.0,
            "min": min(processing_times) if processing_times else 0.0,
            "max": max(processing_times) if processing_times else 0.0
        }
        
        return {
            "total_responses": total_responses,
            "successful_analyses": successful_count,
            "failed_analyses": failed_count,
            "success_rate": (successful_count / total_responses) * 100.0 if total_responses > 0 else 0.0,
            "sentiment_distribution": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "confidence_stats": confidence_stats,
            "processing_time_stats": processing_time_stats
        }
    
    def _create_processing_metadata(self, file_path: str, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create metadata about the processing run.
        
        Args:
            file_path: Path to the source Excel file
            responses: Original response data
            
        Returns:
            Dictionary containing processing metadata
        """
        processing_time = time.time() - self._processing_start_time if self._processing_start_time else 0.0
        
        # Get OpenAI client statistics
        client_stats = self.openai_client.get_client_stats()
        
        # Get progress tracker summary
        progress_summary = self.progress_tracker.get_status_summary() if self.progress_tracker else {}
        
        return {
            "source_file": file_path,
            "processing_start_time": self._processing_start_time,
            "total_processing_time": processing_time,
            "openai_model": self.config.openai_model,
            "api_timeout": self.config.api_timeout,
            "max_retries": self.config.max_retries,
            "client_statistics": client_stats,
            "progress_summary": progress_summary,
            "original_response_count": len(responses),
            "excel_headers": responses[0]['headers'] if responses else []
        }
    
    def generate_insights(self, results: Optional[AnalysisResults] = None) -> Summary:
        """
        Generate comprehensive insights and recommendations from analysis results.
        
        Args:
            results: AnalysisResults object, or None to use current analysis
            
        Returns:
            Summary object with insights and recommendations
            
        Raises:
            SentimentAnalyzerError: If no analysis results are available
        """
        if results is None:
            results = self._current_analysis
        
        if results is None:
            raise SentimentAnalyzerError("No analysis results available for insight generation")
        
        self.logger.info("Generating insights from analysis results...")
        
        # Extract basic statistics
        stats = results.summary_stats
        sentiment_distribution = stats["sentiment_distribution"]
        sentiment_percentages = stats["sentiment_percentages"]
        confidence_stats = stats["confidence_stats"]
        
        # Generate common themes (simplified approach)
        common_themes = self._extract_common_themes(results.individual_results)
        
        # Generate key insights
        key_insights = self._generate_key_insights(stats, results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(stats, results)
        
        summary = Summary(
            sentiment_distribution=sentiment_distribution,
            sentiment_percentages=sentiment_percentages,
            common_themes=common_themes,
            key_insights=key_insights,
            recommendations=recommendations,
            confidence_stats=confidence_stats
        )
        
        self.logger.info("Insights generation completed")
        return summary
    
    def _extract_common_themes(self, results: List[SentimentResult]) -> List[str]:
        """
        Extract common themes from successful analysis results.
        
        This method analyzes the reasoning text and original responses to identify
        common themes and keywords that appear frequently across the dataset.
        
        Args:
            results: List of SentimentResult objects
            
        Returns:
            List of common themes identified
        """
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return ["No successful analyses to extract themes from"]
        
        themes = []
        total = len(successful_results)
        
        # 1. Sentiment distribution themes
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for result in successful_results:
            sentiment_counts[result.sentiment] += 1
        
        for sentiment, count in sentiment_counts.items():
            if count > 0:
                percentage = (count / total) * 100
                themes.append(f"{sentiment.capitalize()} sentiment: {count} responses ({percentage:.1f}%)")
        
        # 2. Confidence-based themes
        high_confidence = sum(1 for r in successful_results if r.confidence > 0.8)
        if high_confidence > 0:
            themes.append(f"High confidence analyses: {high_confidence} responses")
        
        low_confidence = sum(1 for r in successful_results if r.confidence < 0.5)
        if low_confidence > 0:
            themes.append(f"Low confidence analyses: {low_confidence} responses")
        
        # 3. Extract keywords from reasoning text
        reasoning_keywords = self._extract_keywords_from_reasoning(successful_results)
        if reasoning_keywords:
            themes.extend([f"Common reasoning keywords: {', '.join(reasoning_keywords[:5])}"])
        
        # 4. Text length patterns
        short_texts = sum(1 for r in successful_results if len(r.original_text) < 50)
        long_texts = sum(1 for r in successful_results if len(r.original_text) > 200)
        
        if short_texts > total * 0.3:  # More than 30% are short
            themes.append(f"Many short responses: {short_texts} responses under 50 characters")
        
        if long_texts > total * 0.2:  # More than 20% are long
            themes.append(f"Detailed responses: {long_texts} responses over 200 characters")
        
        # 5. Sentiment strength patterns
        strong_positive = sum(1 for r in successful_results 
                            if r.sentiment == "positive" and r.confidence > 0.8)
        strong_negative = sum(1 for r in successful_results 
                            if r.sentiment == "negative" and r.confidence > 0.8)
        
        if strong_positive > 0:
            themes.append(f"Strong positive sentiment: {strong_positive} highly confident positive responses")
        
        if strong_negative > 0:
            themes.append(f"Strong negative sentiment: {strong_negative} highly confident negative responses")
        
        # 6. Mixed or ambiguous responses
        ambiguous = sum(1 for r in successful_results if r.confidence < 0.6)
        if ambiguous > total * 0.25:  # More than 25% are ambiguous
            themes.append(f"Ambiguous responses: {ambiguous} responses with low confidence scores")
        
        return themes[:12]  # Limit to top 12 themes
    
    def _extract_keywords_from_reasoning(self, results: List[SentimentResult]) -> List[str]:
        """
        Extract common keywords from the reasoning text of successful analyses.
        
        Args:
            results: List of successful SentimentResult objects
            
        Returns:
            List of common keywords found in reasoning text
        """
        # Combine all reasoning text
        all_reasoning = " ".join([r.reasoning.lower() for r in results if r.reasoning])
        
        # Common sentiment-related keywords to look for
        positive_keywords = ["positive", "good", "great", "excellent", "love", "amazing", "wonderful", "fantastic"]
        negative_keywords = ["negative", "bad", "terrible", "hate", "awful", "disappointing", "poor", "worst"]
        neutral_keywords = ["neutral", "okay", "average", "mixed", "balanced", "moderate"]
        
        found_keywords = []
        
        # Count occurrences of each keyword category
        pos_count = sum(1 for keyword in positive_keywords if keyword in all_reasoning)
        neg_count = sum(1 for keyword in negative_keywords if keyword in all_reasoning)
        neu_count = sum(1 for keyword in neutral_keywords if keyword in all_reasoning)
        
        if pos_count > 0:
            found_keywords.append("positive language")
        if neg_count > 0:
            found_keywords.append("negative language")
        if neu_count > 0:
            found_keywords.append("neutral language")
        
        # Look for specific common words that might indicate themes
        common_words = ["quality", "service", "experience", "product", "customer", "support", "price", "value"]
        for word in common_words:
            if word in all_reasoning and all_reasoning.count(word) >= len(results) * 0.2:  # Appears in 20%+ of reasoning
                found_keywords.append(word)
        
        return found_keywords[:8]  # Limit to top 8 keywords
    
    def _generate_key_insights(self, stats: Dict[str, Any], results: AnalysisResults) -> List[str]:
        """
        Generate key insights from the analysis statistics.
        
        Args:
            stats: Summary statistics dictionary
            results: Complete analysis results
            
        Returns:
            List of key insights
        """
        insights = []
        
        # Overall sentiment insights
        total = stats["successful_analyses"]
        if total > 0:
            pos_pct = stats["sentiment_percentages"]["positive"]
            neg_pct = stats["sentiment_percentages"]["negative"]
            neu_pct = stats["sentiment_percentages"]["neutral"]
            
            # Dominant sentiment
            if pos_pct > neg_pct and pos_pct > neu_pct:
                insights.append(f"Overall sentiment is predominantly positive ({pos_pct:.1f}% of responses)")
            elif neg_pct > pos_pct and neg_pct > neu_pct:
                insights.append(f"Overall sentiment is predominantly negative ({neg_pct:.1f}% of responses)")
            else:
                insights.append(f"Sentiment is fairly balanced with neutral being most common ({neu_pct:.1f}%)")
            
            # Confidence insights
            avg_confidence = stats["confidence_stats"]["mean"]
            if avg_confidence > 0.8:
                insights.append(f"High average confidence in sentiment classification ({avg_confidence:.2f})")
            elif avg_confidence < 0.5:
                insights.append(f"Low average confidence suggests ambiguous or mixed sentiment ({avg_confidence:.2f})")
            else:
                insights.append(f"Moderate confidence in sentiment classification ({avg_confidence:.2f})")
        
        # Success rate insights
        success_rate = stats["success_rate"]
        if success_rate < 90:
            insights.append(f"Analysis success rate is {success_rate:.1f}% - some responses may need manual review")
        else:
            insights.append(f"High analysis success rate ({success_rate:.1f}%) indicates good data quality")
        
        # Processing performance insights
        avg_time = stats["processing_time_stats"]["average"]
        if avg_time > 5.0:
            insights.append(f"Average processing time is {avg_time:.1f}s per response - consider optimization")
        
        return insights
    
    def _generate_recommendations(self, stats: Dict[str, Any], results: AnalysisResults) -> List[str]:
        """
        Generate actionable recommendations based on the analysis.
        
        Args:
            stats: Summary statistics dictionary
            results: Complete analysis results
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Sentiment-based recommendations
        pos_pct = stats["sentiment_percentages"]["positive"]
        neg_pct = stats["sentiment_percentages"]["negative"]
        
        if neg_pct > 40:
            recommendations.append("High negative sentiment detected - investigate root causes and develop action plan")
        elif pos_pct > 70:
            recommendations.append("Strong positive sentiment - identify and replicate successful practices")
        
        if neg_pct > pos_pct:
            recommendations.append("Focus on addressing negative feedback to improve overall satisfaction")
        
        # Confidence-based recommendations
        avg_confidence = stats["confidence_stats"]["mean"]
        if avg_confidence < 0.6:
            recommendations.append("Low confidence scores suggest responses may need human review for accuracy")
        
        # Success rate recommendations
        success_rate = stats["success_rate"]
        if success_rate < 85:
            recommendations.append("Consider reviewing failed analyses manually and improving data quality")
        
        # Processing recommendations
        failed_count = stats["failed_analyses"]
        if failed_count > 0:
            recommendations.append(f"Review {failed_count} failed analyses to identify patterns or issues")
        
        # Data quality recommendations
        if len(recommendations) == 0:
            recommendations.append("Analysis completed successfully - results can be used for decision making")
        
        return recommendations
    
    def get_current_analysis(self) -> Optional[AnalysisResults]:
        """
        Get the current analysis results.
        
        Returns:
            Current AnalysisResults object or None if no analysis has been run
        """
        return self._current_analysis
    
    def get_progress_tracker(self) -> Optional[ProgressTracker]:
        """
        Get the current progress tracker.
        
        Returns:
            Current ProgressTracker object or None if no analysis is running
        """
        return self.progress_tracker