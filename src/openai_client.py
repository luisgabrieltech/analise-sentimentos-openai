#!/usr/bin/env python3
"""
OpenAI API Client for Sentiment Analysis

This module provides a client for communicating with OpenAI's API to perform
sentiment analysis on text responses. It includes proper error handling,
retry logic, and rate limiting management.
"""

import time
import json
import logging
import random
import asyncio
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import openai
from openai import OpenAI, AsyncOpenAI

from .models import SentimentResult
from .config_manager import Configuration


class OpenAIError(Exception):
    """Custom exception for OpenAI API related errors."""
    pass


class RateLimitError(OpenAIError):
    """Exception raised when API rate limits are exceeded."""
    pass


class AuthenticationError(OpenAIError):
    """Exception raised when API authentication fails."""
    pass


class TimeoutError(OpenAIError):
    """Exception raised when API requests timeout."""
    pass


@dataclass
class APIResponse:
    """Represents a response from the OpenAI API."""
    success: bool
    sentiment: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    suggestions: Optional[List[str]] = None


class OpenAIClient:
    """
    Client for communicating with OpenAI's API for sentiment analysis.
    
    This class handles API communication, prompt engineering, error handling,
    and retry logic with exponential backoff for rate limiting.
    """
    
    def __init__(self, config: Configuration):
        """
        Initialize the OpenAI client with configuration.
        
        Args:
            config: Configuration object containing API settings
        """
        self.config = config
        self.client = OpenAI(
            api_key=config.openai_api_key,
            organization=config.openai_org_id,
            timeout=config.api_timeout
        )
        self.async_client = AsyncOpenAI(
            api_key=config.openai_api_key,
            organization=config.openai_org_id,
            timeout=config.api_timeout
        )
        self.logger = logging.getLogger(__name__)
        
        # Enhanced rate limiting tracking
        self._last_request_time = 0.0
        self._request_count = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._rate_limit_hits = 0
        self._timeout_errors = 0
        self._rate_limit_delay = 0.1  # Start with 100ms delay
        self._consecutive_failures = 0
        self._last_rate_limit_time = 0.0
        
        # Request timing tracking for adaptive rate limiting
        self._request_times: List[float] = []
        self._max_request_history = 100
        
        # Caching system for duplicate text analysis
        self._cache: Dict[str, SentimentResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 1000  # Limit cache size to prevent memory issues
        
        # Async processing controls
        self._semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        self._async_stats = {
            'concurrent_requests': 0,
            'max_concurrent': 0,
            'total_async_requests': 0
        }
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for the given text.
        
        Args:
            text: The text to generate a key for
            
        Returns:
            SHA256 hash of the normalized text
        """
        # Normalize text for consistent caching
        normalized_text = text.strip().lower()
        return hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, text: str) -> Optional[SentimentResult]:
        """
        Retrieve a cached result for the given text.
        
        Args:
            text: The text to look up
            
        Returns:
            Cached SentimentResult or None if not found
        """
        cache_key = self._get_cache_key(text)
        cached_result = self._cache.get(cache_key)
        
        if cached_result:
            self._cache_hits += 1
            self.logger.debug(f"Cache hit for text: {text[:50]}...")
            # Return a copy with updated processing time
            return SentimentResult(
                original_text=text,  # Use original text, not normalized
                sentiment=cached_result.sentiment,
                confidence=cached_result.confidence,
                reasoning=f"[CACHED] {cached_result.reasoning}",
                processing_time=0.001,  # Minimal time for cache retrieval
                success=cached_result.success,
                error_message=cached_result.error_message,
                suggestions=cached_result.suggestions or []
            )
        else:
            self._cache_misses += 1
            return None
    
    def _store_in_cache(self, text: str, result: SentimentResult) -> None:
        """
        Store a result in the cache.
        
        Args:
            text: The original text
            result: The sentiment result to cache
        """
        if not result.success:
            return  # Don't cache failed results
        
        cache_key = self._get_cache_key(text)
        
        # Implement LRU-style cache eviction if cache is full
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.logger.debug("Cache evicted oldest entry due to size limit")
        
        # Store a copy without the original text to save memory
        cached_result = SentimentResult(
            original_text="",  # Don't store original text in cache
            sentiment=result.sentiment,
            confidence=result.confidence,
            reasoning=result.reasoning,
            processing_time=result.processing_time,
            success=result.success,
            error_message=result.error_message,
            suggestions=result.suggestions or []
        )
        
        self._cache[cache_key] = cached_result
        self.logger.debug(f"Cached result for text: {text[:50]}...")
    
    async def analyze_sentiment_async(self, text: str, context: str = None) -> SentimentResult:
        """
        Analyze sentiment of a single text response asynchronously.
        
        Args:
            text: The text to analyze for sentiment
            
        Returns:
            SentimentResult object with analysis results
        """
        # Check cache first
        cached_result = self._get_from_cache(text)
        if cached_result:
            return cached_result
        
        if not text or not text.strip():
            return SentimentResult(
                original_text=text,
                sentiment="neutral",
                confidence=0.0,
                reasoning="Empty or whitespace-only text",
                processing_time=0.0,
                success=False,
                error_message="Cannot analyze empty text",
                suggestions=[]
            )
        
        async with self._semaphore:  # Limit concurrent requests
            self._async_stats['concurrent_requests'] += 1
            self._async_stats['max_concurrent'] = max(
                self._async_stats['max_concurrent'],
                self._async_stats['concurrent_requests']
            )
            self._async_stats['total_async_requests'] += 1
            
            try:
                start_time = time.time()
                
                # Enhanced retry logic with comprehensive error handling
                last_error = None
                for attempt in range(self.config.max_retries + 1):
                    try:
                        self.logger.debug(f"Async attempt {attempt + 1}/{self.config.max_retries + 1} for text analysis")
                        response = await self._make_async_api_request(text)
                        processing_time = time.time() - start_time
                        
                        if response.success:
                            self._successful_requests += 1
                            self._consecutive_failures = 0
                            result = SentimentResult(
                                original_text=text,
                                sentiment=response.sentiment,
                                confidence=response.confidence,
                                reasoning=response.reasoning,
                                processing_time=processing_time,
                                success=True
                            )
                            # Cache successful result
                            self._store_in_cache(text, result)
                            return result
                        else:
                            if attempt == self.config.max_retries:
                                self._failed_requests += 1
                                return SentimentResult(
                                    original_text=text,
                                    sentiment="neutral",
                                    confidence=0.0,
                                    reasoning="Analysis failed",
                                    processing_time=processing_time,
                                    success=False,
                                    error_message=response.error_message,
                                    suggestions=[]
                                )
                    
                    except Exception as e:
                        if attempt < self.config.max_retries:
                            delay = self._calculate_backoff_delay(attempt)
                            await asyncio.sleep(delay)
                            continue
                        else:
                            processing_time = time.time() - start_time
                            self._failed_requests += 1
                            return SentimentResult(
                                original_text=text,
                                sentiment="neutral",
                                confidence=0.0,
                                reasoning="Analysis failed",
                                processing_time=processing_time,
                                success=False,
                                error_message=f"Async error after all retries: {str(e)}",
                                suggestions=[]
                            )
                
            finally:
                self._async_stats['concurrent_requests'] -= 1
    
    async def batch_analyze_async(self, texts: List[str], batch_size: int = 10) -> List[SentimentResult]:
        """
        Analyze multiple texts concurrently with controlled batch processing.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of concurrent requests to process at once
            
        Returns:
            List of SentimentResult objects in the same order as input
        """
        self.logger.info(f"Starting async batch analysis of {len(texts)} texts with batch size {batch_size}")
        
        # Adjust semaphore for batch size
        self._semaphore = asyncio.Semaphore(min(batch_size, 10))  # Cap at 10 concurrent
        
        # Create tasks for all texts
        tasks = [self.analyze_sentiment_async(text) for text in texts]
        
        # Process in batches to avoid overwhelming the API
        results = []
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle any exceptions in the batch
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Batch processing error for text {i+j}: {result}")
                    error_result = SentimentResult(
                        original_text=texts[i+j] if i+j < len(texts) else "",
                        sentiment="neutral",
                        confidence=0.0,
                        reasoning="Batch processing error",
                        processing_time=0.0,
                        success=False,
                        error_message=f"Batch error: {str(result)}",
                        suggestions=[]
                    )
                    results.append(error_result)
                else:
                    results.append(result)
            
            # Small delay between batches to be respectful to the API
            if i + batch_size < len(tasks):
                await asyncio.sleep(0.5)
        
        cache_hit_rate = (self._cache_hits / (self._cache_hits + self._cache_misses)) * 100 if (self._cache_hits + self._cache_misses) > 0 else 0
        self.logger.info(f"Batch analysis completed. Cache hit rate: {cache_hit_rate:.1f}%")
        
        return results
    
    async def _make_async_api_request(self, text: str) -> APIResponse:
        """
        Make an asynchronous API request to analyze sentiment.
        
        Args:
            text: The text to analyze
            
        Returns:
            APIResponse object with the result
        """
        request_start_time = time.time()
        
        try:
            # Create the prompt for sentiment analysis
            prompt = self._create_sentiment_prompt(text, context)
            
            self.logger.debug(f"Making async API request to model {self.config.openai_model}")
            
            # Make the async API call
            response = await self.async_client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "Você é um especialista em análise de sentimentos. Analise o texto fornecido e responda com um objeto JSON contendo sentiment, confidence e reasoning em português."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"},
                timeout=self.config.api_timeout
            )
            
            # Track successful request timing
            request_time = time.time() - request_start_time
            self._track_request_timing(request_time)
            
            # Parse the response
            return self._parse_api_response(response)
            
        except Exception as e:
            # Convert async exceptions to our standard format
            if "rate_limit" in str(e).lower():
                raise openai.RateLimitError(str(e))
            elif "auth" in str(e).lower():
                raise openai.AuthenticationError(str(e))
            elif "timeout" in str(e).lower():
                raise openai.APITimeoutError(str(e))
            else:
                raise openai.APIError(str(e))

    def analyze_sentiment(self, text: str, context: str = None) -> SentimentResult:
        """
        Analyze sentiment of a single text response.
        
        Args:
            text: The text to analyze for sentiment
            
        Returns:
            SentimentResult object with analysis results
            
        Raises:
            OpenAIError: If API communication fails after all retries
        """
        # Check cache first
        cached_result = self._get_from_cache(text)
        if cached_result:
            return cached_result
        
        if not text or not text.strip():
            return SentimentResult(
                original_text=text,
                sentiment="neutral",
                confidence=0.0,
                reasoning="Empty or whitespace-only text",
                processing_time=0.0,
                success=False,
                error_message="Cannot analyze empty text",
                suggestions=[]
            )
        
        start_time = time.time()
        
        # Enhanced retry logic with comprehensive error handling
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.debug(f"Attempt {attempt + 1}/{self.config.max_retries + 1} for text analysis")
                response = self._make_api_request(text, context)
                processing_time = time.time() - start_time
                
                if response.success:
                    self._successful_requests += 1
                    self._consecutive_failures = 0
                    self.logger.debug(f"Successful analysis in {processing_time:.2f}s on attempt {attempt + 1}")
                    result = SentimentResult(
                        original_text=text,
                        sentiment=response.sentiment,
                        confidence=response.confidence,
                        reasoning=response.reasoning,
                        processing_time=processing_time,
                        success=True,
                        suggestions=response.suggestions or []
                    )
                    # Cache successful result
                    self._store_in_cache(text, result)
                    return result
                else:
                    # If this was the last attempt, return failed result
                    if attempt == self.config.max_retries:
                        self._failed_requests += 1
                        self.logger.error(f"Analysis failed after {attempt + 1} attempts: {response.error_message}")
                        return SentimentResult(
                            original_text=text,
                            sentiment="neutral",
                            confidence=0.0,
                            reasoning="Analysis failed",
                            processing_time=processing_time,
                            success=False,
                            error_message=response.error_message,
                            suggestions=[]
                        )
                    else:
                        self.logger.warning(f"Attempt {attempt + 1} failed: {response.error_message}")
                    
            except RateLimitError as e:
                self._rate_limit_hits += 1
                self._consecutive_failures += 1
                last_error = e
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_rate_limit_delay(attempt)
                    self.logger.warning(
                        f"Rate limit hit on attempt {attempt + 1}/{self.config.max_retries + 1}. "
                        f"Waiting {delay:.2f}s before retry. Total rate limit hits: {self._rate_limit_hits}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    processing_time = time.time() - start_time
                    self._failed_requests += 1
                    self.logger.error(
                        f"Rate limit exceeded after all {self.config.max_retries + 1} attempts. "
                        f"Total processing time: {processing_time:.2f}s"
                    )
                    return SentimentResult(
                        original_text=text,
                        sentiment="neutral",
                        confidence=0.0,
                        reasoning="Analysis failed due to rate limiting",
                        processing_time=processing_time,
                        success=False,
                        error_message=f"Rate limit exceeded after all retries: {str(e)}",
                        suggestions=[]
                    )
            
            except TimeoutError as e:
                self._timeout_errors += 1
                self._consecutive_failures += 1
                last_error = e
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    self.logger.warning(
                        f"Timeout on attempt {attempt + 1}/{self.config.max_retries + 1}. "
                        f"Retrying in {delay:.2f}s. Total timeouts: {self._timeout_errors}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    processing_time = time.time() - start_time
                    self._failed_requests += 1
                    self.logger.error(f"Request timed out after all {self.config.max_retries + 1} attempts")
                    return SentimentResult(
                        original_text=text,
                        sentiment="neutral",
                        confidence=0.0,
                        reasoning="Analysis failed due to timeout",
                        processing_time=processing_time,
                        success=False,
                        error_message=f"Request timed out after all retries: {str(e)}",
                        suggestions=[]
                    )
            
            except AuthenticationError as e:
                # Don't retry authentication errors
                processing_time = time.time() - start_time
                self._failed_requests += 1
                self.logger.error(f"Authentication failed: {str(e)}")
                return SentimentResult(
                    original_text=text,
                    sentiment="neutral",
                    confidence=0.0,
                    reasoning="Authentication failed",
                    processing_time=processing_time,
                    success=False,
                    error_message=f"Authentication failed: {str(e)}",
                    suggestions=[]
                )
            
            except OpenAIError as e:
                self._consecutive_failures += 1
                last_error = e
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    self.logger.warning(
                        f"API error on attempt {attempt + 1}/{self.config.max_retries + 1}: {str(e)}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                    continue
                else:
                    processing_time = time.time() - start_time
                    self._failed_requests += 1
                    self.logger.error(f"API error after all {self.config.max_retries + 1} attempts: {str(e)}")
                    return SentimentResult(
                        original_text=text,
                        sentiment="neutral",
                        confidence=0.0,
                        reasoning="Analysis failed",
                        processing_time=processing_time,
                        success=False,
                        error_message=f"API error after all retries: {str(e)}",
                        suggestions=[]
                    )
        
        # This should never be reached, but just in case
        processing_time = time.time() - start_time
        return SentimentResult(
            original_text=text,
            sentiment="neutral",
            confidence=0.0,
            reasoning="Analysis failed",
            processing_time=processing_time,
            success=False,
            error_message="Unexpected error in retry loop",
            suggestions=[]
        )
    
    def _make_api_request(self, text: str, context: str = None) -> APIResponse:
        """
        Make a single API request to analyze sentiment.
        
        Args:
            text: The text to analyze
            context: Optional context information (e.g., column headers)
            
        Returns:
            APIResponse object with the result
            
        Raises:
            RateLimitError: If rate limits are exceeded
            AuthenticationError: If authentication fails
            OpenAIError: For other API errors
        """
        request_start_time = time.time()
        
        try:
            # Implement enhanced rate limiting
            self._enforce_rate_limit()
            
            # Create the prompt for sentiment analysis
            prompt = self._create_sentiment_prompt(text, context)
            
            self.logger.debug(f"Making API request to model {self.config.openai_model}")
            
            # Make the API call with timeout handling
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "Você é um especialista em análise de sentimentos. Analise o texto fornecido e responda com um objeto JSON contendo sentiment, confidence e reasoning em português."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"},
                timeout=self.config.api_timeout
            )
            
            # Track successful request timing
            request_time = time.time() - request_start_time
            self._track_request_timing(request_time)
            self.logger.debug(f"API request completed in {request_time:.2f}s")
            
            # Parse the response
            return self._parse_api_response(response)
            
        except openai.RateLimitError as e:
            self._handle_rate_limit()
            error_msg = self._create_user_friendly_rate_limit_message(e)
            self.logger.warning(f"Rate limit error: {str(e)}")
            raise RateLimitError(error_msg)
        
        except openai.AuthenticationError as e:
            error_msg = self._create_user_friendly_auth_message(e)
            self.logger.error(f"Authentication error: {str(e)}")
            raise AuthenticationError(error_msg)
        
        except openai.APITimeoutError as e:
            request_time = time.time() - request_start_time
            error_msg = self._create_user_friendly_timeout_message(e, request_time)
            self.logger.warning(f"Request timed out after {request_time:.2f}s: {str(e)}")
            raise TimeoutError(error_msg)
        
        except openai.APIConnectionError as e:
            error_msg = self._create_user_friendly_connection_message(e)
            self.logger.warning(f"Connection error: {str(e)}")
            raise OpenAIError(error_msg)
        
        except openai.APIError as e:
            error_msg = self._create_user_friendly_api_message(e)
            self.logger.warning(f"API error: {str(e)}")
            raise OpenAIError(error_msg)
        
        except Exception as e:
            error_msg = f"An unexpected error occurred while communicating with OpenAI: {str(e)}"
            self.logger.error(f"Unexpected error during API request: {str(e)}")
            raise OpenAIError(error_msg)
    
    def _create_sentiment_prompt(self, text: str, context: str = None) -> str:
        """
        Create a well-engineered prompt for sentiment analysis with suggestion extraction.
        
        Args:
            text: The text to analyze
            context: Optional context information (e.g., column headers)
            
        Returns:
            Formatted prompt string
        """
        # Truncate text if it's too long to avoid token limits
        max_text_length = 2000  # Conservative limit
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        
        # Build context section if provided
        context_section = ""
        if context and context.strip():
            # Clean and format context
            clean_context = context.replace('_', ' ').replace('-', ' ').title()
            context_section = f"""
CONTEXTO DA PERGUNTA/CAMPO:
"{clean_context}"

Use este contexto para entender melhor o propósito da resposta e fornecer análise mais específica e sugestões mais relevantes.

"""

        prompt = f"""
Analise o sentimento do seguinte texto e responda com um objeto JSON contendo:
- "sentiment": um de "positive", "negative", ou "neutral"
- "confidence": um número entre 0.0 e 1.0 indicando a confiança na classificação
- "reasoning": uma breve explicação em português de por que este sentimento foi escolhido
- "suggestions": uma lista de sugestões ou temas mencionados no texto (máximo 5 itens)

Diretrizes:
- "positive": Texto expressa satisfação, felicidade, aprovação ou outras emoções positivas
- "negative": Texto expressa insatisfação, frustração, crítica ou outras emoções negativas
- "neutral": Texto é factual, equilibrado ou não expressa sentimento emocional claro
- Considere contexto, tom e mensagem geral
- Seja conservador com pontuações de confiança - use pontuações menores para texto ambíguo
- IMPORTANTE: A explicação (reasoning) deve ser sempre em português

Para "suggestions":
- Extraia temas, sugestões de melhoria, problemas mencionados ou pontos importantes
- Use frases curtas e claras (máximo 50 caracteres cada)
- Foque em aspectos acionáveis ou categorias relevantes
- Se não houver sugestões claras, retorne uma lista vazia []
- Considere o contexto fornecido para sugestões mais específicas e relevantes
- Exemplos: ["Melhorar atendimento", "Mais informações", "Sistema mais rápido"]

{context_section}Texto para analisar:
"{text}"

Responda apenas com JSON válido:
"""
        return prompt.strip()
    
    def _parse_api_response(self, response) -> APIResponse:
        """
        Parse the OpenAI API response into our APIResponse format.
        
        Args:
            response: OpenAI API response object
            
        Returns:
            APIResponse object with parsed data
        """
        try:
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            if not content or content.strip() == "":
                return APIResponse(
                    success=False,
                    error_message="Empty response from API"
                )
            
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the whole content
                data = json.loads(content)
            
            # Validate required fields
            sentiment = data.get("sentiment", "").lower()
            if sentiment not in ["positive", "negative", "neutral"]:
                return APIResponse(
                    success=False,
                    error_message=f"Invalid sentiment value: {sentiment}"
                )
            
            confidence = data.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
                return APIResponse(
                    success=False,
                    error_message=f"Invalid confidence value: {confidence}"
                )
            
            reasoning = data.get("reasoning", "")
            if not isinstance(reasoning, str):
                reasoning = str(reasoning)
            
            # Process suggestions field
            suggestions = data.get("suggestions", [])
            if not isinstance(suggestions, list):
                suggestions = []
            else:
                # Ensure all suggestions are strings and limit length
                suggestions = [str(s)[:50] for s in suggestions if s][:5]
            
            return APIResponse(
                success=True,
                sentiment=sentiment,
                confidence=float(confidence),
                reasoning=reasoning,
                suggestions=suggestions
            )
            
        except json.JSONDecodeError as e:
            return APIResponse(
                success=False,
                error_message=f"Failed to parse JSON response: {e}"
            )
        
        except (KeyError, IndexError) as e:
            return APIResponse(
                success=False,
                error_message=f"Unexpected response format: {e}"
            )
        
        except Exception as e:
            return APIResponse(
                success=False,
                error_message=f"Error parsing response: {e}"
            )
    
    def _enforce_rate_limit(self) -> None:
        """
        Enforce enhanced rate limiting to avoid hitting API limits.
        Uses adaptive rate limiting based on recent request patterns.
        """
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        # Calculate adaptive delay based on recent failures and rate limits
        base_delay = self._rate_limit_delay
        
        # Increase delay if we've had consecutive failures
        if self._consecutive_failures > 0:
            base_delay *= (1 + self._consecutive_failures * 0.5)
        
        # Increase delay if we've hit rate limits recently
        time_since_rate_limit = current_time - self._last_rate_limit_time
        if time_since_rate_limit < 60.0:  # Within last minute
            base_delay *= 2.0
        
        # Apply minimum delay
        min_delay = max(base_delay, 0.1)  # At least 100ms between requests
        
        if time_since_last_request < min_delay:
            sleep_time = min_delay - time_since_last_request
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f}s")
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
        self._request_count += 1
    
    def _handle_rate_limit(self) -> None:
        """
        Handle rate limit by increasing the delay for future requests.
        Uses more sophisticated backoff strategy.
        """
        self._last_rate_limit_time = time.time()
        
        # Exponential backoff with jitter
        self._rate_limit_delay = min(self._rate_limit_delay * 2.0, 30.0)  # Cap at 30 seconds
        
        self.logger.warning(
            f"Rate limit hit (#{self._rate_limit_hits}), increasing base delay to {self._rate_limit_delay:.2f}s"
        )
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay for retries with jitter.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds with jitter applied
        """
        base_delay = 1.0
        max_delay = 60.0
        
        # Exponential backoff
        delay = min(base_delay * (2 ** attempt), max_delay)
        
        # Add jitter to avoid thundering herd
        jitter = random.uniform(0.1, 0.3) * delay
        final_delay = delay + jitter
        
        return min(final_delay, max_delay)
    
    def _calculate_rate_limit_delay(self, attempt: int) -> float:
        """
        Calculate delay specifically for rate limit errors.
        Uses longer delays than regular backoff.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Start with longer base delay for rate limits
        base_delay = 5.0
        max_delay = 120.0  # Up to 2 minutes for rate limits
        
        # Exponential backoff with higher multiplier
        delay = min(base_delay * (3 ** attempt), max_delay)
        
        # Add jitter
        jitter = random.uniform(0.2, 0.5) * delay
        final_delay = delay + jitter
        
        return min(final_delay, max_delay)
    
    def _track_request_timing(self, request_time: float) -> None:
        """
        Track request timing for adaptive rate limiting.
        
        Args:
            request_time: Time taken for the request in seconds
        """
        self._request_times.append(request_time)
        
        # Keep only recent request times
        if len(self._request_times) > self._max_request_history:
            self._request_times.pop(0)
        
        # Adjust rate limiting based on average request time
        if len(self._request_times) >= 10:
            avg_time = sum(self._request_times[-10:]) / 10
            if avg_time > 5.0:  # If requests are taking too long
                self._rate_limit_delay = max(self._rate_limit_delay, avg_time * 0.1)
    
    def get_client_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the client usage.
        
        Returns:
            Dictionary with detailed client statistics
        """
        success_rate = (self._successful_requests / max(self._request_count, 1)) * 100
        avg_request_time = (sum(self._request_times) / len(self._request_times)) if self._request_times else 0.0
        cache_hit_rate = (self._cache_hits / max(self._cache_hits + self._cache_misses, 1)) * 100
        
        return {
            "total_requests": self._request_count,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate_percent": round(success_rate, 2),
            "rate_limit_hits": self._rate_limit_hits,
            "timeout_errors": self._timeout_errors,
            "consecutive_failures": self._consecutive_failures,
            "current_rate_limit_delay": round(self._rate_limit_delay, 3),
            "average_request_time": round(avg_request_time, 3),
            "model": self.config.openai_model,
            "timeout": self.config.api_timeout,
            "max_retries": self.config.max_retries,
            "last_request_time": self._last_request_time,
            "last_rate_limit_time": self._last_rate_limit_time,
            "cache_stats": {
                "cache_size": len(self._cache),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "max_cache_size": self._max_cache_size
            },
            "async_stats": {
                "total_async_requests": self._async_stats['total_async_requests'],
                "max_concurrent_requests": self._async_stats['max_concurrent'],
                "current_concurrent": self._async_stats['concurrent_requests']
            }
        }
    
    def clear_cache(self) -> None:
        """
        Clear the sentiment analysis cache.
        """
        cache_size = len(self._cache)
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger.info(f"Cache cleared ({cache_size} entries removed)")
    
    def reset_stats(self) -> None:
        """
        Reset all client statistics.
        Useful for testing or starting fresh tracking.
        """
        self._request_count = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._rate_limit_hits = 0
        self._timeout_errors = 0
        self._consecutive_failures = 0
        self._request_times.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._async_stats = {
            'concurrent_requests': 0,
            'max_concurrent': 0,
            'total_async_requests': 0
        }
        self.logger.info("Client statistics reset")
    
    def log_performance_summary(self) -> None:
        """
        Log a comprehensive performance summary.
        """
        stats = self.get_client_stats()
        
        self.logger.info("=== OpenAI Client Performance Summary ===")
        self.logger.info(f"Total requests: {stats['total_requests']}")
        self.logger.info(f"Success rate: {stats['success_rate_percent']}%")
        self.logger.info(f"Rate limit hits: {stats['rate_limit_hits']}")
        self.logger.info(f"Timeout errors: {stats['timeout_errors']}")
        self.logger.info(f"Average request time: {stats['average_request_time']}s")
        self.logger.info(f"Current rate limit delay: {stats['current_rate_limit_delay']}s")
        self.logger.info("===========================================")
    
    def _create_user_friendly_rate_limit_message(self, error: openai.RateLimitError) -> str:
        """Create a user-friendly message for rate limit errors."""
        base_msg = "OpenAI API rate limit exceeded. "
        
        # Check if it's a quota issue vs rate limit issue
        error_str = str(error).lower()
        if "quota" in error_str or "billing" in error_str:
            return (
                f"{base_msg}This appears to be a quota/billing issue. "
                "Please check your OpenAI account billing and usage limits at "
                "https://platform.openai.com/account/billing"
            )
        elif "requests per minute" in error_str or "rpm" in error_str:
            return (
                f"{base_msg}You're sending requests too quickly. "
                "The system will automatically retry with delays, but you may want to "
                "process smaller batches or upgrade your OpenAI plan for higher limits."
            )
        elif "tokens per minute" in error_str or "tpm" in error_str:
            return (
                f"{base_msg}Token usage limit reached. "
                "Consider processing shorter texts or upgrading your OpenAI plan."
            )
        else:
            return (
                f"{base_msg}The system will automatically retry with exponential backoff. "
                "If this persists, consider upgrading your OpenAI plan or processing smaller batches."
            )
    
    def _create_user_friendly_auth_message(self, error: openai.AuthenticationError) -> str:
        """Create a user-friendly message for authentication errors."""
        error_str = str(error).lower()
        
        if "invalid api key" in error_str or "incorrect api key" in error_str:
            return (
                "Invalid OpenAI API key. Please check that your API key is correct and active. "
                "You can find your API keys at https://platform.openai.com/api-keys"
            )
        elif "organization" in error_str:
            return (
                "OpenAI organization ID issue. Please verify your organization ID is correct "
                "or remove it from your configuration if not needed."
            )
        else:
            return (
                f"OpenAI authentication failed: {error}. "
                "Please verify your API key and organization settings."
            )
    
    def _create_user_friendly_timeout_message(self, error: openai.APITimeoutError, request_time: float) -> str:
        """Create a user-friendly message for timeout errors."""
        return (
            f"Request to OpenAI timed out after {request_time:.1f} seconds. "
            "This may be due to network issues or high API load. "
            "The system will automatically retry, but you may want to check your internet connection."
        )
    
    def _create_user_friendly_connection_message(self, error: openai.APIConnectionError) -> str:
        """Create a user-friendly message for connection errors."""
        return (
            f"Failed to connect to OpenAI API: {error}. "
            "This may be due to network issues, firewall restrictions, or temporary API outages. "
            "Please check your internet connection and try again."
        )
    
    def _create_user_friendly_api_message(self, error: openai.APIError) -> str:
        """Create a user-friendly message for general API errors."""
        error_str = str(error).lower()
        
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return (
                f"The specified OpenAI model '{self.config.openai_model}' is not available. "
                "Please check your model name or try a different model like 'gpt-3.5-turbo'."
            )
        elif "context length" in error_str or "token" in error_str:
            return (
                "The text is too long for the OpenAI model to process. "
                "The system will automatically truncate long texts, but you may want to "
                "break very long responses into smaller parts."
            )
        elif "server error" in error_str or "internal error" in error_str:
            return (
                "OpenAI is experiencing server issues. The system will automatically retry, "
                "but you may need to wait a few minutes if the problem persists."
            )
        else:
            return f"OpenAI API error: {error}. The system will retry automatically."