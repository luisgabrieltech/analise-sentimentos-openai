"""
Data models for sentiment analysis system.

This module contains dataclasses that define the structure for sentiment analysis
results, aggregated analysis data, and summary reports.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import time


@dataclass
class SentimentResult:
    """
    Represents the result of sentiment analysis for a single text response.
    
    Attributes:
        original_text: The original text that was analyzed
        sentiment: Classification as 'positive', 'negative', or 'neutral'
        confidence: Confidence score from 0.0 to 1.0
        reasoning: Explanation of the sentiment classification
        processing_time: Time taken to process this response in seconds
        success: Whether the analysis was successful
        error_message: Error details if analysis failed
    """
    original_text: str
    sentiment: str
    confidence: float
    reasoning: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate the data after initialization."""
        if self.sentiment not in ['positive', 'negative', 'neutral']:
            raise ValueError(f"Invalid sentiment: {self.sentiment}. Must be 'positive', 'negative', or 'neutral'")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {self.confidence}. Must be between 0.0 and 1.0")
        
        if self.processing_time < 0:
            raise ValueError(f"Invalid processing_time: {self.processing_time}. Must be non-negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert the dataclass to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class AnalysisResults:
    """
    Represents aggregated results from sentiment analysis of multiple responses.
    
    Attributes:
        individual_results: List of SentimentResult objects for each response
        summary_stats: Dictionary containing statistical summaries
        processing_metadata: Metadata about the processing run
        timestamp: When the analysis was completed
        total_processed: Total number of responses processed
        success_rate: Percentage of successful analyses (0.0 to 1.0)
    """
    individual_results: List[SentimentResult]
    summary_stats: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    timestamp: datetime
    total_processed: int
    success_rate: float
    
    def __post_init__(self):
        """Validate the data after initialization."""
        if self.total_processed < 0:
            raise ValueError(f"Invalid total_processed: {self.total_processed}. Must be non-negative")
        
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError(f"Invalid success_rate: {self.success_rate}. Must be between 0.0 and 1.0")
        
        if len(self.individual_results) > self.total_processed:
            raise ValueError("Number of individual results cannot exceed total processed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        data = asdict(self)
        # Convert datetime to ISO format string for JSON serialization
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert the dataclass to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class Summary:
    """
    Represents summary insights and statistics from sentiment analysis.
    
    Attributes:
        sentiment_distribution: Count of each sentiment type
        sentiment_percentages: Percentage distribution of sentiments
        common_themes: List of common themes found in responses
        key_insights: List of key insights from the analysis
        recommendations: List of actionable recommendations
        confidence_stats: Statistics about confidence scores
    """
    sentiment_distribution: Dict[str, int]
    sentiment_percentages: Dict[str, float]
    common_themes: List[str]
    key_insights: List[str]
    recommendations: List[str]
    confidence_stats: Dict[str, float]
    
    def __post_init__(self):
        """Validate the data after initialization."""
        # Validate sentiment distribution keys
        valid_sentiments = {'positive', 'negative', 'neutral'}
        for sentiment in self.sentiment_distribution.keys():
            if sentiment not in valid_sentiments:
                raise ValueError(f"Invalid sentiment in distribution: {sentiment}")
        
        # Validate sentiment percentages
        for sentiment, percentage in self.sentiment_percentages.items():
            if sentiment not in valid_sentiments:
                raise ValueError(f"Invalid sentiment in percentages: {sentiment}")
            if not 0.0 <= percentage <= 100.0:
                raise ValueError(f"Invalid percentage for {sentiment}: {percentage}. Must be between 0.0 and 100.0")
        
        # Validate that percentages sum to approximately 100%
        total_percentage = sum(self.sentiment_percentages.values())
        if abs(total_percentage - 100.0) > 0.1:  # Allow small floating point errors
            raise ValueError(f"Sentiment percentages must sum to 100%, got {total_percentage}")
        
        # Validate confidence stats
        for stat_name, value in self.confidence_stats.items():
            if stat_name in ['mean', 'median', 'min', 'max'] and not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid confidence stat {stat_name}: {value}. Must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert the dataclass to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ProgressTracker:
    """
    Tracks progress of sentiment analysis processing with real-time updates.
    
    Provides progress indicators, estimated time remaining calculations,
    and processing status for each response during analysis.
    """
    
    def __init__(self, total_items: int):
        """
        Initialize the progress tracker.
        
        Args:
            total_items: Total number of items to be processed
        """
        if total_items <= 0:
            raise ValueError("Total items must be greater than 0")
        
        self.total_items = total_items
        self.processed_items = 0
        self.successful_items = 0
        self.failed_items = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.processing_times = []
        self.current_item_text = ""
        self.current_status = "Initializing..."
    
    def update_progress(self, success: bool = True, processing_time: float = 0.0, 
                       current_item: str = "", status: str = ""):
        """
        Update progress with the result of processing one item.
        
        Args:
            success: Whether the current item was processed successfully
            processing_time: Time taken to process the current item
            current_item: Text or identifier of the current item being processed
            status: Current processing status message
        """
        self.processed_items += 1
        
        if success:
            self.successful_items += 1
        else:
            self.failed_items += 1
        
        if processing_time > 0:
            self.processing_times.append(processing_time)
        
        self.current_item_text = current_item[:50] + "..." if len(current_item) > 50 else current_item
        self.current_status = status if status else ("Processing..." if self.processed_items < self.total_items else "Complete")
        self.last_update_time = time.time()
    
    def get_progress_percentage(self) -> float:
        """
        Get the current progress as a percentage.
        
        Returns:
            Progress percentage (0.0 to 100.0)
        """
        if self.total_items == 0:
            return 100.0
        return (self.processed_items / self.total_items) * 100.0
    
    def get_estimated_time_remaining(self) -> float:
        """
        Calculate estimated time remaining based on current progress.
        
        Returns:
            Estimated time remaining in seconds, or 0.0 if calculation not possible
        """
        if self.processed_items == 0 or self.processed_items >= self.total_items:
            return 0.0
        
        # Use average processing time if available, otherwise use elapsed time
        if self.processing_times:
            average_time_per_item = self.get_average_processing_time()
        else:
            elapsed_time = time.time() - self.start_time
            average_time_per_item = elapsed_time / self.processed_items
        
        items_remaining = self.total_items - self.processed_items
        return items_remaining * average_time_per_item
    
    def get_average_processing_time(self) -> float:
        """
        Get the average processing time per item.
        
        Returns:
            Average processing time in seconds, or 0.0 if no items processed
        """
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    def get_success_rate(self) -> float:
        """
        Get the current success rate as a percentage.
        
        Returns:
            Success rate percentage (0.0 to 100.0)
        """
        if self.processed_items == 0:
            return 0.0
        return (self.successful_items / self.processed_items) * 100.0
    
    def get_elapsed_time(self) -> float:
        """
        Get the total elapsed time since tracking started.
        
        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time
    
    def format_time(self, seconds: float) -> str:
        """
        Format time in seconds to a human-readable string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string (e.g., "2m 30s", "1h 15m", "45s")
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes}m {remaining_seconds}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            return f"{hours}h {remaining_minutes}m"
    
    def get_progress_bar(self, width: int = 50) -> str:
        """
        Generate a text-based progress bar.
        
        Args:
            width: Width of the progress bar in characters
            
        Returns:
            Progress bar string
        """
        if width <= 0:
            raise ValueError("Progress bar width must be greater than 0")
        
        progress = self.get_progress_percentage() / 100.0
        filled_width = int(width * progress)
        bar = "█" * filled_width + "░" * (width - filled_width)
        return f"[{bar}]"
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive status summary.
        
        Returns:
            Dictionary containing all progress information
        """
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "progress_percentage": round(self.get_progress_percentage(), 2),
            "success_rate": round(self.get_success_rate(), 2),
            "elapsed_time": round(self.get_elapsed_time(), 2),
            "estimated_time_remaining": round(self.get_estimated_time_remaining(), 2),
            "average_processing_time": round(self.get_average_processing_time(), 2),
            "current_item": self.current_item_text,
            "current_status": self.current_status,
            "is_complete": self.processed_items >= self.total_items
        }
    
    def display_progress(self, show_current_item: bool = True) -> str:
        """
        Generate a formatted progress display string.
        
        Args:
            show_current_item: Whether to include current item information
            
        Returns:
            Formatted progress string for display
        """
        progress_bar = self.get_progress_bar()
        percentage = self.get_progress_percentage()
        elapsed = self.format_time(self.get_elapsed_time())
        remaining = self.format_time(self.get_estimated_time_remaining())
        success_rate = self.get_success_rate()
        
        lines = [
            f"{progress_bar} {percentage:.1f}%",
            f"Progress: {self.processed_items}/{self.total_items} items",
            f"Success Rate: {success_rate:.1f}% ({self.successful_items} successful, {self.failed_items} failed)",
            f"Time: {elapsed} elapsed, {remaining} remaining",
            f"Status: {self.current_status}"
        ]
        
        if show_current_item and self.current_item_text:
            lines.append(f"Current: {self.current_item_text}")
        
        return "\n".join(lines)
    
    def is_complete(self) -> bool:
        """
        Check if all items have been processed.
        
        Returns:
            True if all items are processed, False otherwise
        """
        return self.processed_items >= self.total_items
    
    def reset(self, new_total: Optional[int] = None):
        """
        Reset the progress tracker for a new processing run.
        
        Args:
            new_total: New total number of items, or None to keep current total
        """
        if new_total is not None:
            if new_total <= 0:
                raise ValueError("Total items must be greater than 0")
            self.total_items = new_total
        
        self.processed_items = 0
        self.successful_items = 0
        self.failed_items = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.processing_times = []
        self.current_item_text = ""
        self.current_status = "Initializing..."