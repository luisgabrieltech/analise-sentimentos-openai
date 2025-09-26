"""
Report generation module for sentiment analysis system.

This module provides the ReportGenerator class that creates formatted output reports,
saves detailed results to JSON files, and provides console output for immediate viewing.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from .models import AnalysisResults, SentimentResult, Summary


class ReportGenerator:
    """
    Handles report generation and output formatting for sentiment analysis results.
    
    Provides methods to create human-readable summary reports, save detailed results
    to JSON files, and format console output for immediate viewing.
    """
    
    def __init__(self, output_directory: str = "output"):
        """
        Initialize the ReportGenerator.
        
        Args:
            output_directory: Directory where output files will be saved
        """
        self.output_directory = output_directory
        self._ensure_output_directory()
    
    def _ensure_output_directory(self):
        """Create the output directory if it doesn't exist."""
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
    
    def create_summary_report(self, results: AnalysisResults) -> str:
        """
        Create a human-readable summary report from analysis results.
        
        Args:
            results: AnalysisResults object containing all analysis data
            
        Returns:
            Formatted summary report as a string
        """
        if not results.individual_results:
            return self._create_empty_report()
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(results.individual_results)
        
        # Build the report
        report_lines = []
        
        # Header
        report_lines.extend(self._create_report_header(results))
        
        # Executive Summary
        report_lines.extend(self._create_executive_summary(summary_stats, results))
        
        # Detailed Statistics
        report_lines.extend(self._create_detailed_statistics(summary_stats))
        
        # Sample Results
        report_lines.extend(self._create_sample_results(results.individual_results))
        
        # Suggestions Summary
        report_lines.extend(self._create_suggestions_summary(results.individual_results))
        
        # Insights and Recommendations
        if hasattr(results, 'summary') and results.summary:
            report_lines.extend(self._create_insights_section(results.summary))
        
        # Processing Information
        report_lines.extend(self._create_processing_info(results))
        
        return "\n".join(report_lines)
    
    def _create_empty_report(self) -> str:
        """Create a report for when no results are available."""
        return """
SENTIMENT ANALYSIS REPORT
========================

No results available for analysis.

This could be due to:
- No responses found in the input file
- All responses failed to process
- Processing was interrupted

Please check your input data and try again.
        """.strip()
    
    def _create_report_header(self, results: AnalysisResults) -> List[str]:
        """Create the report header section."""
        return [
            "=" * 60,
            "SENTIMENT ANALYSIS REPORT",
            "=" * 60,
            f"Generated: {results.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Responses Analyzed: {results.total_processed}",
            f"Success Rate: {results.success_rate:.1%}",
            ""
        ]
    
    def _create_executive_summary(self, summary_stats: Dict[str, Any], results: AnalysisResults) -> List[str]:
        """Create the executive summary section."""
        lines = [
            "EXECUTIVE SUMMARY",
            "-" * 17,
            ""
        ]
        
        # Overall sentiment breakdown
        sentiment_dist = summary_stats['sentiment_distribution']
        total_successful = sum(sentiment_dist.values())
        
        if total_successful > 0:
            lines.extend([
                f"Overall Sentiment Distribution:",
                f"  • Positive: {sentiment_dist['positive']} responses ({sentiment_dist['positive']/total_successful:.1%})",
                f"  • Neutral:  {sentiment_dist['neutral']} responses ({sentiment_dist['neutral']/total_successful:.1%})",
                f"  • Negative: {sentiment_dist['negative']} responses ({sentiment_dist['negative']/total_successful:.1%})",
                ""
            ])
            
            # Key findings
            dominant_sentiment = max(sentiment_dist.items(), key=lambda x: x[1])
            lines.extend([
                f"Key Findings:",
                f"  • {dominant_sentiment[0].title()} sentiment dominates with {dominant_sentiment[1]} responses",
                f"  • Average confidence score: {summary_stats['confidence_stats']['mean']:.2f}",
                f"  • Processing completed with {results.success_rate:.1%} success rate",
                ""
            ])
        
        return lines
    
    def _create_detailed_statistics(self, summary_stats: Dict[str, Any]) -> List[str]:
        """Create the detailed statistics section."""
        lines = [
            "DETAILED STATISTICS",
            "-" * 19,
            ""
        ]
        
        # Sentiment distribution
        sentiment_dist = summary_stats['sentiment_distribution']
        lines.extend([
            "Sentiment Distribution:",
            f"  Positive: {sentiment_dist['positive']:>4} responses",
            f"  Neutral:  {sentiment_dist['neutral']:>4} responses", 
            f"  Negative: {sentiment_dist['negative']:>4} responses",
            f"  Total:    {sum(sentiment_dist.values()):>4} responses",
            ""
        ])
        
        # Confidence statistics
        conf_stats = summary_stats['confidence_stats']
        lines.extend([
            "Confidence Score Statistics:",
            f"  Mean:     {conf_stats['mean']:.3f}",
            f"  Median:   {conf_stats['median']:.3f}",
            f"  Min:      {conf_stats['min']:.3f}",
            f"  Max:      {conf_stats['max']:.3f}",
            f"  Std Dev:  {conf_stats['std_dev']:.3f}",
            ""
        ])
        
        # Processing time statistics
        time_stats = summary_stats['processing_time_stats']
        lines.extend([
            "Processing Time Statistics:",
            f"  Average:  {time_stats['mean']:.2f} seconds per response",
            f"  Total:    {time_stats['total']:.2f} seconds",
            f"  Fastest:  {time_stats['min']:.2f} seconds",
            f"  Slowest:  {time_stats['max']:.2f} seconds",
            ""
        ])
        
        return lines
    
    def _create_sample_results(self, results: List[SentimentResult], max_samples: int = 5) -> List[str]:
        """Create a section showing sample results."""
        lines = [
            "SAMPLE RESULTS",
            "-" * 14,
            ""
        ]
        
        # Show samples from each sentiment category
        sentiments = ['positive', 'negative', 'neutral']
        
        for sentiment in sentiments:
            sentiment_results = [r for r in results if r.sentiment == sentiment and r.success]
            if sentiment_results:
                lines.append(f"{sentiment.title()} Examples:")
                
                # Sort by confidence and take top samples
                top_samples = sorted(sentiment_results, key=lambda x: x.confidence, reverse=True)[:max_samples]
                
                for i, result in enumerate(top_samples, 1):
                    text_preview = result.original_text[:100] + "..." if len(result.original_text) > 100 else result.original_text
                    lines.extend([
                        f"  {i}. Text: \"{text_preview}\"",
                        f"     Confidence: {result.confidence:.3f} | Reasoning: {result.reasoning[:80]}{'...' if len(result.reasoning) > 80 else ''}",
                        ""
                    ])
        
        return lines
    
    def _create_insights_section(self, summary: Summary) -> List[str]:
        """Create the insights and recommendations section."""
        lines = [
            "INSIGHTS & RECOMMENDATIONS",
            "-" * 26,
            ""
        ]
        
        if summary.key_insights:
            lines.append("Key Insights:")
            for insight in summary.key_insights:
                lines.append(f"  • {insight}")
            lines.append("")
        
        if summary.common_themes:
            lines.append("Common Themes:")
            for theme in summary.common_themes:
                lines.append(f"  • {theme}")
            lines.append("")
        
        if summary.recommendations:
            lines.append("Recommendations:")
            for recommendation in summary.recommendations:
                lines.append(f"  • {recommendation}")
            lines.append("")
        
        return lines
    
    def _create_suggestions_summary(self, results: List[SentimentResult]) -> List[str]:
        """Create a summary of the most common suggestions extracted from responses."""
        lines = [
            "SUGESTÕES MAIS CITADAS",
            "-" * 22,
            ""
        ]
        
        # Collect all suggestions from successful results
        all_suggestions = []
        for result in results:
            if result.success and result.suggestions:
                all_suggestions.extend(result.suggestions)
        
        if not all_suggestions:
            lines.extend([
                "Nenhuma sugestão específica foi extraída das respostas.",
                ""
            ])
            return lines
        
        # Count frequency of each suggestion
        suggestion_counts = {}
        for suggestion in all_suggestions:
            # Normalize suggestion for counting (lowercase, strip)
            normalized = suggestion.lower().strip()
            if normalized:
                suggestion_counts[normalized] = suggestion_counts.get(normalized, 0) + 1
        
        # Sort by frequency and take top 10
        top_suggestions = sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        lines.append("Os itens mais citados pelos respondentes foram:")
        for i, (suggestion, count) in enumerate(top_suggestions, 1):
            percentage = (count / len([r for r in results if r.success])) * 100
            lines.append(f"  {chr(96 + i)}) {suggestion.title()} ({count} menções - {percentage:.1f}%)")
        
        lines.append("")
        lines.append(f"Total de sugestões únicas: {len(suggestion_counts)}")
        lines.append(f"Total de menções: {len(all_suggestions)}")
        lines.append("")
        
        return lines
    
    def _create_processing_info(self, results: AnalysisResults) -> List[str]:
        """Create the processing information section."""
        lines = [
            "PROCESSING INFORMATION",
            "-" * 22,
            ""
        ]
        
        # Processing metadata
        if results.processing_metadata:
            lines.append("Processing Details:")
            
            # Basic processing info
            basic_info = ['source_file', 'total_processing_time', 'openai_model', 'processing_mode']
            for key in basic_info:
                if key in results.processing_metadata:
                    value = results.processing_metadata[key]
                    if key == 'total_processing_time':
                        value = f"{value:.2f} seconds"
                    lines.append(f"  {key.replace('_', ' ').title()}: {value}")
            
            # Performance statistics
            if 'client_statistics' in results.processing_metadata:
                client_stats = results.processing_metadata['client_statistics']
                lines.extend([
                    "",
                    "API Performance Statistics:",
                    f"  Total API Requests: {client_stats.get('total_requests', 0)}",
                    f"  Success Rate: {client_stats.get('success_rate_percent', 0):.1f}%",
                    f"  Average Request Time: {client_stats.get('average_request_time', 0):.3f}s",
                    f"  Rate Limit Hits: {client_stats.get('rate_limit_hits', 0)}",
                    f"  Timeout Errors: {client_stats.get('timeout_errors', 0)}"
                ])
                
                # Cache statistics
                if 'cache_stats' in client_stats:
                    cache_stats = client_stats['cache_stats']
                    lines.extend([
                        "",
                        "Cache Performance:",
                        f"  Cache Size: {cache_stats.get('cache_size', 0)} entries",
                        f"  Cache Hits: {cache_stats.get('cache_hits', 0)}",
                        f"  Cache Misses: {cache_stats.get('cache_misses', 0)}",
                        f"  Cache Hit Rate: {cache_stats.get('cache_hit_rate_percent', 0):.1f}%"
                    ])
                
                # Async statistics
                if 'async_stats' in client_stats:
                    async_stats = client_stats['async_stats']
                    if async_stats.get('total_async_requests', 0) > 0:
                        lines.extend([
                            "",
                            "Async Processing Statistics:",
                            f"  Total Async Requests: {async_stats.get('total_async_requests', 0)}",
                            f"  Max Concurrent Requests: {async_stats.get('max_concurrent_requests', 0)}"
                        ])
            
            # Memory optimization info
            if 'chunk_size' in results.processing_metadata:
                lines.extend([
                    "",
                    "Memory Optimization:",
                    f"  Chunk Size: {results.processing_metadata['chunk_size']} responses",
                    f"  Total Chunks: {results.processing_metadata.get('total_chunks', 'Unknown')}"
                ])
            
            lines.append("")
        
        # Failed analyses
        failed_results = [r for r in results.individual_results if not r.success]
        if failed_results:
            lines.extend([
                f"Failed Analyses ({len(failed_results)} total):",
                "  Common error types:"
            ])
            
            # Group errors by type
            error_counts = {}
            for result in failed_results:
                error_type = result.error_message.split(':')[0] if result.error_message else "Unknown error"
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"    • {error_type}: {count} occurrences")
            lines.append("")
        
        lines.extend([
            "=" * 60,
            f"Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60
        ])
        
        return lines
    
    def save_detailed_results(self, results: AnalysisResults, filename: Optional[str] = None) -> str:
        """
        Save detailed analysis results to a JSON file.
        
        Args:
            results: AnalysisResults object to save
            filename: Optional custom filename, defaults to timestamp-based name
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = results.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_directory, filename)
        
        try:
            # Ensure output directory exists
            self._ensure_output_directory()
            
            # Convert results to dictionary with error handling
            try:
                results_dict = results.to_dict()
            except Exception as e:
                # Fallback: create a basic dictionary if to_dict() fails
                results_dict = {
                    "error": f"Failed to serialize results: {str(e)}",
                    "timestamp": results.timestamp.isoformat() if hasattr(results, 'timestamp') else None,
                    "total_processed": getattr(results, 'total_processed', 0),
                    "success_rate": getattr(results, 'success_rate', 0.0)
                }
            
            # Write to temporary file first, then rename (atomic operation)
            temp_filepath = filepath + '.tmp'
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
            
            # Rename temp file to final name
            os.rename(temp_filepath, filepath)
            
            return filepath
            
        except PermissionError as e:
            raise IOError(
                f"Permission denied saving to {filepath}. "
                f"Please check that the directory is writable and the file is not open in another application. "
                f"Error: {str(e)}"
            )
        except OSError as e:
            raise IOError(
                f"Failed to save results to {filepath}. "
                f"Please check available disk space and directory permissions. "
                f"Error: {str(e)}"
            )
        except Exception as e:
            raise IOError(f"Unexpected error saving results to {filepath}: {str(e)}")
    
    def save_summary_report(self, results: AnalysisResults, filename: Optional[str] = None) -> str:
        """
        Save the human-readable summary report to a text file.
        
        Args:
            results: AnalysisResults object to create report from
            filename: Optional custom filename, defaults to timestamp-based name
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = results.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_summary_{timestamp}.txt"
        
        filepath = os.path.join(self.output_directory, filename)
        
        try:
            # Ensure output directory exists
            self._ensure_output_directory()
            
            # Generate report content with error handling
            try:
                report_content = self.create_summary_report(results)
            except Exception as e:
                # Fallback: create a basic error report
                report_content = f"""
SENTIMENT ANALYSIS REPORT - ERROR
================================

An error occurred while generating the full report: {str(e)}

Basic Information:
- Timestamp: {getattr(results, 'timestamp', 'Unknown')}
- Total Processed: {getattr(results, 'total_processed', 'Unknown')}
- Success Rate: {getattr(results, 'success_rate', 'Unknown')}

Please check the detailed JSON results file for more information.
                """.strip()
            
            # Write to temporary file first, then rename
            temp_filepath = filepath + '.tmp'
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Rename temp file to final name
            os.rename(temp_filepath, filepath)
            
            return filepath
            
        except PermissionError as e:
            raise IOError(
                f"Permission denied saving summary to {filepath}. "
                f"Please check directory permissions and ensure the file is not open elsewhere. "
                f"Error: {str(e)}"
            )
        except OSError as e:
            raise IOError(
                f"Failed to save summary report to {filepath}. "
                f"Please check available disk space and directory permissions. "
                f"Error: {str(e)}"
            )
        except Exception as e:
            raise IOError(f"Unexpected error saving summary report to {filepath}: {str(e)}")
    
    def display_console_summary(self, results: AnalysisResults, show_samples: bool = True) -> None:
        """
        Display a formatted summary to the console for immediate viewing.
        
        Args:
            results: AnalysisResults object to display
            show_samples: Whether to include sample results in the output
        """
        print("\n" + "=" * 60)
        print("SENTIMENT ANALYSIS COMPLETE")
        print("=" * 60)
        
        if not results.individual_results:
            print("No results to display.")
            return
        
        # Quick stats
        summary_stats = self._calculate_summary_statistics(results.individual_results)
        sentiment_dist = summary_stats['sentiment_distribution']
        total_successful = sum(sentiment_dist.values())
        
        print(f"Total Processed: {results.total_processed}")
        print(f"Success Rate: {results.success_rate:.1%}")
        print(f"Average Confidence: {summary_stats['confidence_stats']['mean']:.3f}")
        print()
        
        # Sentiment distribution
        print("SENTIMENT BREAKDOWN:")
        if total_successful > 0:
            for sentiment in ['positive', 'neutral', 'negative']:
                count = sentiment_dist[sentiment]
                percentage = count / total_successful * 100
                bar_length = int(percentage / 2)  # Scale to fit in console
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"  {sentiment.title():>8}: {count:>3} ({percentage:>5.1f}%) [{bar}]")
        print()
        
        # Sample results if requested
        if show_samples:
            print("SAMPLE RESULTS:")
            for sentiment in ['positive', 'negative', 'neutral']:
                sentiment_results = [r for r in results.individual_results if r.sentiment == sentiment and r.success]
                if sentiment_results:
                    best_example = max(sentiment_results, key=lambda x: x.confidence)
                    text_preview = best_example.original_text[:80] + "..." if len(best_example.original_text) > 80 else best_example.original_text
                    print(f"  {sentiment.title()}: \"{text_preview}\" (confidence: {best_example.confidence:.3f})")
            print()
        
        print("=" * 60)
    
    def _calculate_summary_statistics(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics from results."""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'confidence_stats': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std_dev': 0},
                'processing_time_stats': {'mean': 0, 'total': 0, 'min': 0, 'max': 0}
            }
        
        # Sentiment distribution
        sentiment_dist = {'positive': 0, 'negative': 0, 'neutral': 0}
        for result in successful_results:
            sentiment_dist[result.sentiment] += 1
        
        # Confidence statistics
        confidences = [r.confidence for r in successful_results]
        confidences.sort()
        n = len(confidences)
        
        confidence_stats = {
            'mean': sum(confidences) / n,
            'median': confidences[n//2] if n % 2 == 1 else (confidences[n//2-1] + confidences[n//2]) / 2,
            'min': min(confidences),
            'max': max(confidences),
            'std_dev': self._calculate_std_dev(confidences)
        }
        
        # Processing time statistics
        processing_times = [r.processing_time for r in results if r.processing_time > 0]
        if processing_times:
            time_stats = {
                'mean': sum(processing_times) / len(processing_times),
                'total': sum(processing_times),
                'min': min(processing_times),
                'max': max(processing_times)
            }
        else:
            time_stats = {'mean': 0, 'total': 0, 'min': 0, 'max': 0}
        
        return {
            'sentiment_distribution': sentiment_dist,
            'confidence_stats': confidence_stats,
            'processing_time_stats': time_stats
        }
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def create_progress_display(self, current: int, total: int, current_item: str = "", 
                              success_rate: float = 1.0, elapsed_time: float = 0.0) -> str:
        """
        Create a formatted progress display string for console output.
        
        Args:
            current: Number of items processed
            total: Total number of items to process
            current_item: Description of current item being processed
            success_rate: Current success rate (0.0 to 1.0)
            elapsed_time: Elapsed processing time in seconds
            
        Returns:
            Formatted progress string
        """
        if total <= 0:
            return "Invalid progress parameters"
        
        percentage = (current / total) * 100
        bar_length = 40
        filled_length = int(bar_length * current / total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Format elapsed time
        if elapsed_time < 60:
            time_str = f"{elapsed_time:.0f}s"
        elif elapsed_time < 3600:
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            time_str = f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            time_str = f"{hours}h {minutes}m"
        
        lines = [
            f"Progress: [{bar}] {percentage:.1f}%",
            f"Items: {current}/{total} | Success Rate: {success_rate:.1%} | Time: {time_str}"
        ]
        
        if current_item:
            item_preview = current_item[:60] + "..." if len(current_item) > 60 else current_item
            lines.append(f"Current: {item_preview}")
        
        return "\n".join(lines)