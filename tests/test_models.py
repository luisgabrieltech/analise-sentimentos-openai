"""
Unit tests for data models in the sentiment analysis system.

Tests cover validation, serialization, and edge cases for all dataclasses.
"""

import pytest
import json
import time
from datetime import datetime
from unittest.mock import patch
from src.models import SentimentResult, AnalysisResults, Summary, ProgressTracker


class TestSentimentResult:
    """Test cases for SentimentResult dataclass."""
    
    def test_valid_sentiment_result_creation(self):
        """Test creating a valid SentimentResult."""
        result = SentimentResult(
            original_text="This is a great product!",
            sentiment="positive",
            confidence=0.85,
            reasoning="The text contains positive language and enthusiasm",
            processing_time=1.2,
            success=True
        )
        
        assert result.original_text == "This is a great product!"
        assert result.sentiment == "positive"
        assert result.confidence == 0.85
        assert result.reasoning == "The text contains positive language and enthusiasm"
        assert result.processing_time == 1.2
        assert result.success is True
        assert result.error_message is None
    
    def test_sentiment_result_with_error(self):
        """Test creating a SentimentResult with error."""
        result = SentimentResult(
            original_text="Some text",
            sentiment="neutral",
            confidence=0.0,
            reasoning="",
            processing_time=0.5,
            success=False,
            error_message="API timeout"
        )
        
        assert result.success is False
        assert result.error_message == "API timeout"
    
    def test_invalid_sentiment_raises_error(self):
        """Test that invalid sentiment values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid sentiment"):
            SentimentResult(
                original_text="Test",
                sentiment="invalid",
                confidence=0.5,
                reasoning="Test",
                processing_time=1.0,
                success=True
            )
    
    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid confidence"):
            SentimentResult(
                original_text="Test",
                sentiment="positive",
                confidence=1.5,  # Invalid: > 1.0
                reasoning="Test",
                processing_time=1.0,
                success=True
            )
        
        with pytest.raises(ValueError, match="Invalid confidence"):
            SentimentResult(
                original_text="Test",
                sentiment="positive",
                confidence=-0.1,  # Invalid: < 0.0
                reasoning="Test",
                processing_time=1.0,
                success=True
            )
    
    def test_invalid_processing_time_raises_error(self):
        """Test that negative processing time raises ValueError."""
        with pytest.raises(ValueError, match="Invalid processing_time"):
            SentimentResult(
                original_text="Test",
                sentiment="positive",
                confidence=0.5,
                reasoning="Test",
                processing_time=-1.0,  # Invalid: negative
                success=True
            )
    
    def test_sentiment_result_to_dict(self):
        """Test converting SentimentResult to dictionary."""
        result = SentimentResult(
            original_text="Test text",
            sentiment="positive",
            confidence=0.8,
            reasoning="Test reasoning",
            processing_time=1.5,
            success=True
        )
        
        result_dict = result.to_dict()
        expected_dict = {
            'original_text': 'Test text',
            'sentiment': 'positive',
            'confidence': 0.8,
            'reasoning': 'Test reasoning',
            'processing_time': 1.5,
            'success': True,
            'error_message': None
        }
        
        assert result_dict == expected_dict
    
    def test_sentiment_result_to_json(self):
        """Test converting SentimentResult to JSON."""
        result = SentimentResult(
            original_text="Test text",
            sentiment="negative",
            confidence=0.7,
            reasoning="Test reasoning",
            processing_time=2.0,
            success=True
        )
        
        json_str = result.to_json()
        parsed = json.loads(json_str)
        
        assert parsed['original_text'] == 'Test text'
        assert parsed['sentiment'] == 'negative'
        assert parsed['confidence'] == 0.7


class TestAnalysisResults:
    """Test cases for AnalysisResults dataclass."""
    
    def test_valid_analysis_results_creation(self):
        """Test creating valid AnalysisResults."""
        individual_results = [
            SentimentResult("Text 1", "positive", 0.8, "Reason 1", 1.0, True),
            SentimentResult("Text 2", "negative", 0.7, "Reason 2", 1.2, True)
        ]
        
        results = AnalysisResults(
            individual_results=individual_results,
            summary_stats={'positive': 1, 'negative': 1, 'neutral': 0},
            processing_metadata={'version': '1.0', 'model': 'gpt-3.5-turbo'},
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            total_processed=2,
            success_rate=1.0
        )
        
        assert len(results.individual_results) == 2
        assert results.total_processed == 2
        assert results.success_rate == 1.0
        assert results.summary_stats['positive'] == 1
    
    def test_invalid_total_processed_raises_error(self):
        """Test that negative total_processed raises ValueError."""
        with pytest.raises(ValueError, match="Invalid total_processed"):
            AnalysisResults(
                individual_results=[],
                summary_stats={},
                processing_metadata={},
                timestamp=datetime.now(),
                total_processed=-1,  # Invalid: negative
                success_rate=1.0
            )
    
    def test_invalid_success_rate_raises_error(self):
        """Test that invalid success_rate raises ValueError."""
        with pytest.raises(ValueError, match="Invalid success_rate"):
            AnalysisResults(
                individual_results=[],
                summary_stats={},
                processing_metadata={},
                timestamp=datetime.now(),
                total_processed=0,
                success_rate=1.5  # Invalid: > 1.0
            )
    
    def test_too_many_individual_results_raises_error(self):
        """Test that more individual results than total processed raises ValueError."""
        individual_results = [
            SentimentResult("Text 1", "positive", 0.8, "Reason", 1.0, True),
            SentimentResult("Text 2", "negative", 0.7, "Reason", 1.0, True)
        ]
        
        with pytest.raises(ValueError, match="Number of individual results cannot exceed total processed"):
            AnalysisResults(
                individual_results=individual_results,
                summary_stats={},
                processing_metadata={},
                timestamp=datetime.now(),
                total_processed=1,  # Less than len(individual_results)
                success_rate=1.0
            )
    
    def test_analysis_results_to_dict(self):
        """Test converting AnalysisResults to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        results = AnalysisResults(
            individual_results=[],
            summary_stats={'test': 'value'},
            processing_metadata={'version': '1.0'},
            timestamp=timestamp,
            total_processed=0,
            success_rate=1.0
        )
        
        result_dict = results.to_dict()
        assert result_dict['timestamp'] == timestamp.isoformat()
        assert result_dict['summary_stats'] == {'test': 'value'}
    
    def test_analysis_results_to_json(self):
        """Test converting AnalysisResults to JSON."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        results = AnalysisResults(
            individual_results=[],
            summary_stats={'positive': 1},
            processing_metadata={'version': '1.0'},
            timestamp=timestamp,
            total_processed=1,
            success_rate=1.0
        )
        
        json_str = results.to_json()
        parsed = json.loads(json_str)
        
        assert parsed['timestamp'] == timestamp.isoformat()
        assert parsed['total_processed'] == 1


class TestSummary:
    """Test cases for Summary dataclass."""
    
    def test_valid_summary_creation(self):
        """Test creating a valid Summary."""
        summary = Summary(
            sentiment_distribution={'positive': 5, 'negative': 3, 'neutral': 2},
            sentiment_percentages={'positive': 50.0, 'negative': 30.0, 'neutral': 20.0},
            common_themes=['quality', 'price', 'service'],
            key_insights=['Overall positive sentiment', 'Price concerns noted'],
            recommendations=['Improve pricing strategy', 'Maintain quality'],
            confidence_stats={'mean': 0.75, 'median': 0.8, 'min': 0.5, 'max': 0.95}
        )
        
        assert summary.sentiment_distribution['positive'] == 5
        assert summary.sentiment_percentages['positive'] == 50.0
        assert len(summary.common_themes) == 3
        assert summary.confidence_stats['mean'] == 0.75
    
    def test_invalid_sentiment_in_distribution_raises_error(self):
        """Test that invalid sentiment keys in distribution raise ValueError."""
        with pytest.raises(ValueError, match="Invalid sentiment in distribution"):
            Summary(
                sentiment_distribution={'invalid': 1, 'positive': 2},
                sentiment_percentages={'positive': 100.0},
                common_themes=[],
                key_insights=[],
                recommendations=[],
                confidence_stats={'mean': 0.5}
            )
    
    def test_invalid_sentiment_in_percentages_raises_error(self):
        """Test that invalid sentiment keys in percentages raise ValueError."""
        with pytest.raises(ValueError, match="Invalid sentiment in percentages"):
            Summary(
                sentiment_distribution={'positive': 1},
                sentiment_percentages={'invalid': 100.0},
                common_themes=[],
                key_insights=[],
                recommendations=[],
                confidence_stats={'mean': 0.5}
            )
    
    def test_invalid_percentage_value_raises_error(self):
        """Test that invalid percentage values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid percentage"):
            Summary(
                sentiment_distribution={'positive': 1},
                sentiment_percentages={'positive': 150.0},  # Invalid: > 100.0
                common_themes=[],
                key_insights=[],
                recommendations=[],
                confidence_stats={'mean': 0.5}
            )
    
    def test_percentages_not_summing_to_100_raises_error(self):
        """Test that percentages not summing to 100% raise ValueError."""
        with pytest.raises(ValueError, match="Sentiment percentages must sum to 100%"):
            Summary(
                sentiment_distribution={'positive': 1, 'negative': 1},
                sentiment_percentages={'positive': 60.0, 'negative': 30.0},  # Sum = 90%
                common_themes=[],
                key_insights=[],
                recommendations=[],
                confidence_stats={'mean': 0.5}
            )
    
    def test_invalid_confidence_stats_raises_error(self):
        """Test that invalid confidence statistics raise ValueError."""
        with pytest.raises(ValueError, match="Invalid confidence stat"):
            Summary(
                sentiment_distribution={'positive': 1},
                sentiment_percentages={'positive': 100.0},
                common_themes=[],
                key_insights=[],
                recommendations=[],
                confidence_stats={'mean': 1.5}  # Invalid: > 1.0
            )
    
    def test_summary_to_dict(self):
        """Test converting Summary to dictionary."""
        summary = Summary(
            sentiment_distribution={'positive': 1},
            sentiment_percentages={'positive': 100.0},
            common_themes=['theme1'],
            key_insights=['insight1'],
            recommendations=['rec1'],
            confidence_stats={'mean': 0.8}
        )
        
        result_dict = summary.to_dict()
        assert result_dict['sentiment_distribution'] == {'positive': 1}
        assert result_dict['common_themes'] == ['theme1']
    
    def test_summary_to_json(self):
        """Test converting Summary to JSON."""
        summary = Summary(
            sentiment_distribution={'positive': 2, 'negative': 1},
            sentiment_percentages={'positive': 66.7, 'negative': 33.3},
            common_themes=['quality'],
            key_insights=['Mostly positive'],
            recommendations=['Keep it up'],
            confidence_stats={'mean': 0.75}
        )
        
        json_str = summary.to_json()
        parsed = json.loads(json_str)
        
        assert parsed['sentiment_distribution']['positive'] == 2
        assert parsed['confidence_stats']['mean'] == 0.75


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_confidence_sentiment_result(self):
        """Test SentimentResult with zero confidence."""
        result = SentimentResult(
            original_text="Ambiguous text",
            sentiment="neutral",
            confidence=0.0,  # Minimum valid value
            reasoning="Very uncertain",
            processing_time=0.0,  # Minimum valid value
            success=True
        )
        
        assert result.confidence == 0.0
        assert result.processing_time == 0.0
    
    def test_maximum_confidence_sentiment_result(self):
        """Test SentimentResult with maximum confidence."""
        result = SentimentResult(
            original_text="Very clear text",
            sentiment="positive",
            confidence=1.0,  # Maximum valid value
            reasoning="Absolutely certain",
            processing_time=10.0,
            success=True
        )
        
        assert result.confidence == 1.0
    
    def test_empty_analysis_results(self):
        """Test AnalysisResults with no individual results."""
        results = AnalysisResults(
            individual_results=[],
            summary_stats={},
            processing_metadata={},
            timestamp=datetime.now(),
            total_processed=0,
            success_rate=0.0  # Valid when no results processed
        )
        
        assert len(results.individual_results) == 0
        assert results.total_processed == 0
        assert results.success_rate == 0.0
    
    def test_summary_with_floating_point_precision(self):
        """Test Summary with floating point precision in percentages."""
        # Test that small floating point errors are tolerated
        summary = Summary(
            sentiment_distribution={'positive': 1, 'negative': 2},
            sentiment_percentages={'positive': 33.33, 'negative': 66.67},  # Sum = 100.00
            common_themes=[],
            key_insights=[],
            recommendations=[],
            confidence_stats={'mean': 0.5}
        )
        
        # Should not raise an error despite small floating point difference
        assert summary.sentiment_percentages['positive'] == 33.33


class TestProgressTracker:
    """Test cases for ProgressTracker class."""
    
    def test_valid_progress_tracker_creation(self):
        """Test creating a valid ProgressTracker."""
        tracker = ProgressTracker(total_items=10)
        
        assert tracker.total_items == 10
        assert tracker.processed_items == 0
        assert tracker.successful_items == 0
        assert tracker.failed_items == 0
        assert tracker.current_status == "Initializing..."
        assert len(tracker.processing_times) == 0
    
    def test_invalid_total_items_raises_error(self):
        """Test that invalid total_items raises ValueError."""
        with pytest.raises(ValueError, match="Total items must be greater than 0"):
            ProgressTracker(total_items=0)
        
        with pytest.raises(ValueError, match="Total items must be greater than 0"):
            ProgressTracker(total_items=-5)
    
    def test_update_progress_success(self):
        """Test updating progress with successful processing."""
        tracker = ProgressTracker(total_items=5)
        
        tracker.update_progress(
            success=True,
            processing_time=1.5,
            current_item="Test item 1",
            status="Processing item 1"
        )
        
        assert tracker.processed_items == 1
        assert tracker.successful_items == 1
        assert tracker.failed_items == 0
        assert tracker.processing_times == [1.5]
        assert tracker.current_item_text == "Test item 1"
        assert tracker.current_status == "Processing item 1"
    
    def test_update_progress_failure(self):
        """Test updating progress with failed processing."""
        tracker = ProgressTracker(total_items=5)
        
        tracker.update_progress(
            success=False,
            processing_time=0.8,
            current_item="Failed item",
            status="Processing failed"
        )
        
        assert tracker.processed_items == 1
        assert tracker.successful_items == 0
        assert tracker.failed_items == 1
        assert tracker.processing_times == [0.8]
        assert tracker.current_status == "Processing failed"
    
    def test_update_progress_long_text_truncation(self):
        """Test that long current item text is truncated."""
        tracker = ProgressTracker(total_items=1)
        long_text = "This is a very long text that should be truncated because it exceeds the maximum length"
        
        tracker.update_progress(current_item=long_text)
        
        assert len(tracker.current_item_text) == 53  # 50 chars + "..."
        assert tracker.current_item_text.endswith("...")
    
    def test_get_progress_percentage(self):
        """Test progress percentage calculation."""
        tracker = ProgressTracker(total_items=10)
        
        # Initial progress
        assert tracker.get_progress_percentage() == 0.0
        
        # 50% progress
        for _ in range(5):
            tracker.update_progress()
        assert tracker.get_progress_percentage() == 50.0
        
        # 100% progress
        for _ in range(5):
            tracker.update_progress()
        assert tracker.get_progress_percentage() == 100.0
    
    def test_get_progress_percentage_zero_items(self):
        """Test progress percentage with zero total items."""
        # This should not happen in normal usage, but test edge case
        tracker = ProgressTracker(total_items=1)
        tracker.total_items = 0  # Manually set to test edge case
        
        assert tracker.get_progress_percentage() == 100.0
    
    @patch('time.time')
    def test_get_estimated_time_remaining(self, mock_time):
        """Test estimated time remaining calculation."""
        # Mock time progression
        mock_time.side_effect = [0, 10, 20]  # start, after 1 item, after 2 items
        
        tracker = ProgressTracker(total_items=10)
        
        # No items processed yet
        assert tracker.get_estimated_time_remaining() == 0.0
        
        # Process one item (took 10 seconds)
        tracker.update_progress(processing_time=10.0)
        estimated = tracker.get_estimated_time_remaining()
        assert estimated == 90.0  # 9 items * 10 seconds each
        
        # Process another item (total 20 seconds for 2 items)
        tracker.update_progress(processing_time=10.0)
        estimated = tracker.get_estimated_time_remaining()
        assert estimated == 80.0  # 8 items * 10 seconds each
    
    def test_get_estimated_time_remaining_complete(self):
        """Test estimated time remaining when processing is complete."""
        tracker = ProgressTracker(total_items=2)
        
        tracker.update_progress()
        tracker.update_progress()
        
        assert tracker.get_estimated_time_remaining() == 0.0
    
    def test_get_average_processing_time(self):
        """Test average processing time calculation."""
        tracker = ProgressTracker(total_items=5)
        
        # No processing times recorded
        assert tracker.get_average_processing_time() == 0.0
        
        # Add some processing times
        tracker.update_progress(processing_time=1.0)
        tracker.update_progress(processing_time=2.0)
        tracker.update_progress(processing_time=3.0)
        
        assert tracker.get_average_processing_time() == 2.0  # (1+2+3)/3
    
    def test_get_success_rate(self):
        """Test success rate calculation."""
        tracker = ProgressTracker(total_items=10)
        
        # No items processed
        assert tracker.get_success_rate() == 0.0
        
        # 3 successful, 1 failed
        tracker.update_progress(success=True)
        tracker.update_progress(success=True)
        tracker.update_progress(success=True)
        tracker.update_progress(success=False)
        
        assert tracker.get_success_rate() == 75.0  # 3/4 * 100
    
    @patch('time.time')
    def test_get_elapsed_time(self, mock_time):
        """Test elapsed time calculation."""
        mock_time.side_effect = [100, 150]  # start time, current time
        
        tracker = ProgressTracker(total_items=5)
        elapsed = tracker.get_elapsed_time()
        
        assert elapsed == 50.0  # 150 - 100
    
    def test_format_time_seconds(self):
        """Test time formatting for seconds."""
        tracker = ProgressTracker(total_items=1)
        
        assert tracker.format_time(30) == "30s"
        assert tracker.format_time(59) == "59s"
    
    def test_format_time_minutes(self):
        """Test time formatting for minutes."""
        tracker = ProgressTracker(total_items=1)
        
        assert tracker.format_time(60) == "1m 0s"
        assert tracker.format_time(90) == "1m 30s"
        assert tracker.format_time(3599) == "59m 59s"
    
    def test_format_time_hours(self):
        """Test time formatting for hours."""
        tracker = ProgressTracker(total_items=1)
        
        assert tracker.format_time(3600) == "1h 0m"
        assert tracker.format_time(3900) == "1h 5m"
        assert tracker.format_time(7200) == "2h 0m"
    
    def test_get_progress_bar_default_width(self):
        """Test progress bar generation with default width."""
        tracker = ProgressTracker(total_items=10)
        
        # 0% progress
        bar = tracker.get_progress_bar()
        assert len(bar) == 52  # [50 chars]
        assert bar.startswith("[")
        assert bar.endswith("]")
        assert "░" in bar  # Should contain empty blocks
        
        # 50% progress
        for _ in range(5):
            tracker.update_progress()
        bar = tracker.get_progress_bar()
        assert "█" in bar  # Should contain filled blocks
        assert "░" in bar  # Should contain empty blocks
    
    def test_get_progress_bar_custom_width(self):
        """Test progress bar generation with custom width."""
        tracker = ProgressTracker(total_items=4)
        
        # 50% progress with width 10
        tracker.update_progress()
        tracker.update_progress()
        bar = tracker.get_progress_bar(width=10)
        
        assert len(bar) == 12  # [10 chars]
        # Should have 5 filled and 5 empty (50% of 10)
        inner_bar = bar[1:-1]  # Remove brackets
        filled_count = inner_bar.count("█")
        empty_count = inner_bar.count("░")
        assert filled_count == 5
        assert empty_count == 5
    
    def test_get_progress_bar_invalid_width(self):
        """Test progress bar with invalid width raises error."""
        tracker = ProgressTracker(total_items=1)
        
        with pytest.raises(ValueError, match="Progress bar width must be greater than 0"):
            tracker.get_progress_bar(width=0)
        
        with pytest.raises(ValueError, match="Progress bar width must be greater than 0"):
            tracker.get_progress_bar(width=-5)
    
    def test_get_status_summary(self):
        """Test comprehensive status summary."""
        tracker = ProgressTracker(total_items=5)
        
        # Process some items
        tracker.update_progress(success=True, processing_time=1.0, current_item="Item 1")
        tracker.update_progress(success=True, processing_time=2.0, current_item="Item 2")
        tracker.update_progress(success=False, processing_time=1.5, current_item="Item 3")
        
        summary = tracker.get_status_summary()
        
        assert summary["total_items"] == 5
        assert summary["processed_items"] == 3
        assert summary["successful_items"] == 2
        assert summary["failed_items"] == 1
        assert summary["progress_percentage"] == 60.0
        assert summary["success_rate"] == 66.67  # 2/3 * 100, rounded
        assert summary["average_processing_time"] == 1.5  # (1+2+1.5)/3
        assert summary["current_item"] == "Item 3"
        assert summary["is_complete"] is False
    
    def test_display_progress_with_current_item(self):
        """Test formatted progress display with current item."""
        tracker = ProgressTracker(total_items=10)
        tracker.update_progress(success=True, current_item="Test item", status="Processing...")
        
        display = tracker.display_progress(show_current_item=True)
        
        assert "10.0%" in display
        assert "1/10 items" in display
        assert "Success Rate: 100.0%" in display
        assert "Processing..." in display
        assert "Current: Test item" in display
    
    def test_display_progress_without_current_item(self):
        """Test formatted progress display without current item."""
        tracker = ProgressTracker(total_items=5)
        tracker.update_progress(current_item="Hidden item")
        
        display = tracker.display_progress(show_current_item=False)
        
        assert "Hidden item" not in display
        assert "Current:" not in display
    
    def test_is_complete(self):
        """Test completion status checking."""
        tracker = ProgressTracker(total_items=3)
        
        assert tracker.is_complete() is False
        
        tracker.update_progress()
        tracker.update_progress()
        assert tracker.is_complete() is False
        
        tracker.update_progress()
        assert tracker.is_complete() is True
    
    def test_reset_with_same_total(self):
        """Test resetting progress tracker with same total."""
        tracker = ProgressTracker(total_items=5)
        
        # Process some items
        tracker.update_progress(success=True, processing_time=1.0)
        tracker.update_progress(success=False, processing_time=2.0)
        
        # Reset
        tracker.reset()
        
        assert tracker.total_items == 5  # Unchanged
        assert tracker.processed_items == 0
        assert tracker.successful_items == 0
        assert tracker.failed_items == 0
        assert tracker.processing_times == []
        assert tracker.current_item_text == ""
        assert tracker.current_status == "Initializing..."
    
    def test_reset_with_new_total(self):
        """Test resetting progress tracker with new total."""
        tracker = ProgressTracker(total_items=5)
        tracker.update_progress()
        
        # Reset with new total
        tracker.reset(new_total=10)
        
        assert tracker.total_items == 10
        assert tracker.processed_items == 0
        assert tracker.successful_items == 0
        assert tracker.failed_items == 0
    
    def test_reset_with_invalid_total(self):
        """Test resetting with invalid total raises error."""
        tracker = ProgressTracker(total_items=5)
        
        with pytest.raises(ValueError, match="Total items must be greater than 0"):
            tracker.reset(new_total=0)
        
        with pytest.raises(ValueError, match="Total items must be greater than 0"):
            tracker.reset(new_total=-3)
    
    def test_progress_tracking_workflow(self):
        """Test complete progress tracking workflow."""
        tracker = ProgressTracker(total_items=3)
        
        # Process items one by one
        tracker.update_progress(
            success=True,
            processing_time=1.0,
            current_item="First item",
            status="Processing first item"
        )
        assert abs(tracker.get_progress_percentage() - 33.33) < 0.01
        assert not tracker.is_complete()
        
        tracker.update_progress(
            success=True,
            processing_time=1.5,
            current_item="Second item",
            status="Processing second item"
        )
        assert abs(tracker.get_progress_percentage() - 66.67) < 0.01
        assert not tracker.is_complete()
        
        tracker.update_progress(
            success=False,
            processing_time=0.5,
            current_item="Third item",
            status="Failed to process third item"
        )
        assert tracker.get_progress_percentage() == 100.0
        assert tracker.is_complete()
        assert abs(tracker.get_success_rate() - 66.67) < 0.01  # 2/3 successful
        assert tracker.get_average_processing_time() == 1.0  # (1.0+1.5+0.5)/3
    
    @patch('time.time')
    def test_real_time_updates(self, mock_time):
        """Test that progress tracker updates in real-time."""
        # Simulate time progression: start, update1, update2, update3
        mock_time.side_effect = [0, 0, 1, 1, 3, 3, 6, 6]  # start, last_update, get_elapsed calls
        
        tracker = ProgressTracker(total_items=3)
        
        # First update after 1 second
        tracker.update_progress(processing_time=1.0)
        assert tracker.get_elapsed_time() == 1.0
        
        # Second update after 3 seconds total
        tracker.update_progress(processing_time=2.0)
        assert tracker.get_elapsed_time() == 3.0
        
        # Third update after 6 seconds total
        tracker.update_progress(processing_time=3.0)
        assert tracker.get_elapsed_time() == 6.0
        assert tracker.get_average_processing_time() == 2.0  # (1+2+3)/3