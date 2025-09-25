#!/usr/bin/env python3
"""
Sentiment Analysis System

This application processes survey responses from an Excel file using OpenAI's GPT API
to analyze sentiment and generate comprehensive reports with insights and trends.
"""

import sys
import os
import argparse
import signal
import logging
import asyncio
from pathlib import Path
from typing import Optional

from src.config_manager import ConfigurationManager, ConfigurationError
from src.excel_reader import ExcelReader, ExcelReaderError
from src.sentiment_analyzer import SentimentAnalyzer, SentimentAnalyzerError
from src.report_generator import ReportGenerator


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    global shutdown_requested
    shutdown_requested = True
    print("\n⚠️  Shutdown requested. Finishing current operation...")


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Set up comprehensive logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter if not verbose else detailed_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # Always detailed in file
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
            
            logging.info(f"Logging to file: {log_file}")
        except Exception as e:
            logging.warning(f"Could not set up file logging: {e}")
    
    # Set specific logger levels to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis System - Analyze survey responses using OpenAI's GPT API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Analyze respostas.xlsx with default settings
  %(prog)s -f survey_data.xlsx      # Analyze custom Excel file
  %(prog)s -o results/              # Save output to custom directory
  %(prog)s --no-save               # Don't save files, only display results
  %(prog)s -v                      # Enable verbose logging
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        default='respostas.xlsx',
        help='Excel file to analyze (default: respostas.xlsx)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        default='output',
        help='Output directory for reports (default: output)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Don\'t save results to files, only display console output'
    )
    
    parser.add_argument(
        '--no-samples',
        action='store_true',
        help='Don\'t show sample results in console output'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-file',
        help='Save detailed logs to specified file'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate input file structure without processing'
    )
    
    parser.add_argument(
        '--max-responses',
        type=int,
        help='Limit processing to first N responses (for testing)'
    )
    
    parser.add_argument(
        '--async-processing',
        action='store_true',
        help='Use async processing for improved performance with concurrent API calls'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of concurrent requests for async processing (default: 10)'
    )
    
    parser.add_argument(
        '--memory-optimized',
        action='store_true',
        help='Use memory-optimized processing for large datasets'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100,
        help='Chunk size for memory-optimized processing (default: 100)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Sentiment Analysis System 1.0.0'
    )
    
    return parser.parse_args()


def validate_input_file(file_path: str) -> bool:
    """Comprehensive validation of the input file."""
    logger = logging.getLogger(__name__)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: Input file '{file_path}' does not exist.")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Available Excel files in current directory:")
        
        # List available Excel files
        excel_files = [f for f in os.listdir('.') if f.lower().endswith(('.xlsx', '.xls'))]
        if excel_files:
            for excel_file in excel_files[:5]:  # Show up to 5 files
                print(f"     - {excel_file}")
            if len(excel_files) > 5:
                print(f"     ... and {len(excel_files) - 5} more")
        else:
            print("     (No Excel files found)")
        
        return False
    
    # Check if it's a file (not a directory)
    if not os.path.isfile(file_path):
        print(f"❌ Error: '{file_path}' is not a file.")
        return False
    
    # Check file extension
    if not file_path.lower().endswith(('.xlsx', '.xls')):
        print(f"❌ Error: '{file_path}' is not an Excel file.")
        print(f"   Supported formats: .xlsx, .xls")
        return False
    
    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        print(f"❌ Error: Cannot read input file '{file_path}'.")
        print(f"   Please check file permissions.")
        return False
    
    # Check file size
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"❌ Error: Input file '{file_path}' is empty.")
            return False
        elif file_size > 100 * 1024 * 1024:  # 100MB
            print(f"⚠️  Warning: Input file '{file_path}' is very large ({file_size / (1024*1024):.1f}MB).")
            print(f"   Processing may take a long time and consume significant resources.")
            response = input("   Continue anyway? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                return False
        
        logger.info(f"Input file validation passed: {file_path} ({file_size} bytes)")
        return True
        
    except OSError as e:
        print(f"❌ Error: Cannot access file '{file_path}': {e}")
        return False


def display_startup_banner(args: argparse.Namespace) -> None:
    """Display the application startup banner."""
    print("=" * 60)
    print("🔍 SENTIMENT ANALYSIS SYSTEM")
    print("=" * 60)
    print(f"📁 Input file: {args.file}")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"💾 Save results: {'No' if args.no_save else 'Yes'}")
    print(f"📊 Show samples: {'No' if args.no_samples else 'Yes'}")
    print(f"⚡ Async processing: {'Yes' if args.async_processing else 'No'}")
    if args.async_processing:
        print(f"📦 Batch size: {args.batch_size}")
    print(f"🧠 Memory optimized: {'Yes' if args.memory_optimized else 'No'}")
    if args.memory_optimized:
        print(f"📊 Chunk size: {args.chunk_size}")
    print("=" * 60)


def main(args: Optional[argparse.Namespace] = None) -> int:
    """
    Main entry point for the sentiment analysis application.
    
    Args:
        args: Optional parsed command-line arguments (for testing)
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    global shutdown_requested
    
    # Parse arguments if not provided
    if args is None:
        args = parse_arguments()
    
    # Set up logging
    setup_logging(args.verbose, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Log startup information
    logger.info("=== Sentiment Analysis System Starting ===")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")
    logger.info("=" * 50)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Display startup banner
    display_startup_banner(args)
    
    # Validate input file
    if not validate_input_file(args.file):
        return 1
    
    # Initialize report generator
    report_generator = None
    if not args.no_save:
        try:
            report_generator = ReportGenerator(args.output_dir)
            logger.info(f"Report generator initialized with output directory: {args.output_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize report generator: {e}")
            print("   Continuing with console output only...")
    
    try:
        # 1. Load configuration and validate API keys
        print("\n🔧 Loading configuration...")
        config_manager = ConfigurationManager()
        
        try:
            config = config_manager.load_configuration()
            print("✅ Configuration loaded successfully")
            
            # Display safe configuration summary
            summary = config_manager.get_safe_config_summary()
            print(f"🔑 API Key configured: {summary['api_key_preview']}")
            print(f"🤖 Model: {summary['model']}")
            print(f"⏱️  Timeout: {summary['timeout']}s")
            print(f"🔄 Max retries: {summary['max_retries']}")
            if summary.get('organization_id'):
                print(f"🏢 Organization ID: {summary['organization_id']}")
            
        except ConfigurationError as e:
            print(f"❌ Configuration Error: {e}")
            config_manager.display_setup_instructions()
            return 1
        
        # Check for shutdown request
        if shutdown_requested:
            print("🛑 Shutdown requested during configuration. Exiting...")
            return 0
        
        # 2. Read survey responses from Excel file
        print(f"\n📊 Loading survey responses from '{args.file}'...")
        excel_reader = ExcelReader()
        
        try:
            # Get file info first for validation
            file_info = excel_reader.get_file_info(args.file)
            logger.info(f"File info: {file_info}")
            
            print(f"📁 File size: {file_info['file_size_mb']} MB")
            print(f"📊 Columns found: {file_info['column_count']} ({', '.join(file_info['columns'][:5])}{'...' if len(file_info['columns']) > 5 else ''})")
            
            # Load responses with enhanced error handling
            responses = excel_reader.load_responses(args.file)
            print(f"✅ Successfully loaded {len(responses)} responses")
            
            # Apply max responses limit if specified
            if args.max_responses and len(responses) > args.max_responses:
                responses = responses[:args.max_responses]
                print(f"🔢 Limited to first {args.max_responses} responses for processing")
                logger.info(f"Applied max_responses limit: {args.max_responses}")
            
            # Display sample of loaded data
            if responses:
                sample_response = responses[0]
                text_preview = sample_response['text_content'][:100]
                if len(sample_response['text_content']) > 100:
                    text_preview += "..."
                print(f"📝 Sample response: \"{text_preview}\"")
                print(f"📋 Data columns: {', '.join(sample_response['headers'])}")
                
                # Log data quality metrics
                total_text_length = sum(len(r['text_content']) for r in responses)
                avg_text_length = total_text_length / len(responses)
                logger.info(f"Data quality: {len(responses)} responses, avg length {avg_text_length:.1f} chars")
            else:
                print("⚠️  No responses found in the file.")
                logger.error("No valid responses found after loading and processing")
                return 1
            
            # If validation-only mode, stop here
            if args.validate_only:
                print(f"\n✅ Validation complete! File structure is valid for sentiment analysis.")
                print(f"📊 Ready to process {len(responses)} responses")
                return 0
            
        except ExcelReaderError as e:
            print(f"❌ Excel Reading Error: {e}")
            logger.error(f"Excel reading failed: {e}")
            
            # Provide helpful suggestions based on error type
            error_str = str(e).lower()
            if "does not exist" in error_str:
                print("\n💡 Suggestions:")
                print("   • Check the file path and name")
                print("   • Ensure the file is in the current directory")
                print("   • Use --help to see usage examples")
            elif "permission denied" in error_str:
                print("\n💡 Suggestions:")
                print("   • Close the Excel file if it's open in another application")
                print("   • Check file permissions")
                print("   • Try copying the file to a different location")
            elif "no data" in error_str or "empty" in error_str:
                print("\n💡 Suggestions:")
                print("   • Ensure the Excel file contains both headers and data rows")
                print("   • Check that responses are in text format, not just numbers")
                print("   • Verify the file isn't corrupted")
            
            return 1
        
        except Exception as e:
            print(f"❌ Unexpected error loading file: {e}")
            logger.error(f"Unexpected error during file loading: {e}", exc_info=True)
            return 1
        
        # Check for shutdown request
        if shutdown_requested:
            print("🛑 Shutdown requested during file loading. Exiting...")
            return 0
        
        # 3. Initialize sentiment analyzer and process responses
        print("\n🤖 Initializing sentiment analyzer...")
        
        try:
            analyzer = SentimentAnalyzer(config)
            logger.info("Sentiment analyzer initialized successfully")
            
            # Determine processing mode
            if args.memory_optimized:
                print(f"🧠 Starting memory-optimized sentiment analysis processing (chunk size: {args.chunk_size})...")
                print("   (Processing in chunks to optimize memory usage)")
                analysis_results = analyzer.process_responses_memory_optimized(args.file, args.chunk_size)
            elif args.async_processing:
                print(f"⚡ Starting async sentiment analysis processing (batch size: {args.batch_size})...")
                print("   (Using concurrent API calls for improved performance)")
                # Run async processing
                analysis_results = asyncio.run(analyzer.process_responses_async(args.file, args.batch_size))
            else:
                print("🔄 Starting standard sentiment analysis processing...")
                print("   (This may take a while depending on the number of responses)")
                # Use pre-loaded responses to respect max_responses limit
                analysis_results = analyzer.process_responses_from_data(responses, args.file)
            
            # Check for shutdown request after processing
            if shutdown_requested:
                print("🛑 Shutdown requested during processing. Saving partial results...")
            
            print(f"\n✅ Analysis completed!")
            print(f"📊 Processed: {analysis_results.total_processed} responses")
            print(f"📈 Success rate: {analysis_results.success_rate:.1%}")
            
            # Show performance statistics
            if 'client_statistics' in analysis_results.processing_metadata:
                client_stats = analysis_results.processing_metadata['client_statistics']
                print(f"⚡ API Performance:")
                print(f"  • Total requests: {client_stats.get('total_requests', 0)}")
                print(f"  • Average response time: {client_stats.get('average_request_time', 0):.3f}s")
                
                # Cache statistics
                if 'cache_stats' in client_stats:
                    cache_stats = client_stats['cache_stats']
                    if cache_stats.get('cache_hits', 0) > 0:
                        print(f"🎯 Cache Performance:")
                        print(f"  • Cache hit rate: {cache_stats.get('cache_hit_rate_percent', 0):.1f}%")
                        print(f"  • Cache entries: {cache_stats.get('cache_size', 0)}")
                
                # Async statistics
                if 'async_stats' in client_stats:
                    async_stats = client_stats['async_stats']
                    if async_stats.get('total_async_requests', 0) > 0:
                        print(f"⚡ Async Performance:")
                        print(f"  • Max concurrent requests: {async_stats.get('max_concurrent_requests', 0)}")
            
            # Show processing time breakdown
            total_time = analysis_results.processing_metadata.get('total_processing_time', 0)
            if total_time > 0:
                print(f"⏱️  Processing time: {total_time:.2f}s")
                if analysis_results.total_processed > 0:
                    avg_time = total_time / analysis_results.total_processed
                    print(f"  • Average per response: {avg_time:.3f}s")
            
            # 4. Generate insights and summary
            print("\n💡 Generating insights and summary...")
            summary = analyzer.generate_insights(analysis_results)
            
            # 5. Display results
            if report_generator:
                report_generator.display_console_summary(
                    analysis_results, 
                    show_samples=not args.no_samples
                )
            else:
                # Fallback console display
                print("\n" + "=" * 60)
                print("SENTIMENT ANALYSIS SUMMARY")
                print("=" * 60)
                
                print(f"\n📊 Sentiment Distribution:")
                for sentiment, count in summary.sentiment_distribution.items():
                    percentage = summary.sentiment_percentages[sentiment]
                    print(f"  {sentiment.capitalize()}: {count} responses ({percentage:.1f}%)")
                
                if summary.key_insights:
                    print(f"\n🎯 Key Insights:")
                    for i, insight in enumerate(summary.key_insights, 1):
                        print(f"  {i}. {insight}")
                
                if summary.recommendations:
                    print(f"\n💡 Recommendations:")
                    for i, recommendation in enumerate(summary.recommendations, 1):
                        print(f"  {i}. {recommendation}")
            
            # 6. Save results to files
            if report_generator and not args.no_save:
                print(f"\n💾 Saving results to '{args.output_dir}'...")
                
                try:
                    # Save detailed JSON results
                    json_path = report_generator.save_detailed_results(analysis_results)
                    print(f"✅ Detailed results saved: {json_path}")
                    
                    # Save human-readable summary
                    summary_path = report_generator.save_summary_report(analysis_results)
                    print(f"✅ Summary report saved: {summary_path}")
                    
                except Exception as e:
                    print(f"⚠️  Warning: Could not save results to files: {e}")
                    print("   Results are still available in console output above.")
            
            print("\n" + "=" * 60)
            print("🎉 Analysis completed successfully!")
            print("=" * 60)
            
        except SentimentAnalyzerError as e:
            print(f"❌ Sentiment Analysis Error: {e}")
            logger.error(f"Sentiment analysis failed: {e}")
            
            # Provide helpful suggestions based on error type
            error_str = str(e).lower()
            if "api" in error_str and ("key" in error_str or "auth" in error_str):
                print("\n💡 Suggestions:")
                print("   • Verify your OpenAI API key is correct and active")
                print("   • Check your account has sufficient credits")
                print("   • Visit https://platform.openai.com/api-keys to manage keys")
            elif "rate limit" in error_str or "quota" in error_str:
                print("\n💡 Suggestions:")
                print("   • Wait a few minutes and try again")
                print("   • Process smaller batches using --max-responses")
                print("   • Consider upgrading your OpenAI plan")
                print("   • Check your usage at https://platform.openai.com/account/usage")
            elif "network" in error_str or "connection" in error_str:
                print("\n💡 Suggestions:")
                print("   • Check your internet connection")
                print("   • Try again in a few minutes")
                print("   • Check if your firewall is blocking the connection")
            
            return 1
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user.")
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)