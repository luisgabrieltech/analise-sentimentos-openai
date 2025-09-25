#!/usr/bin/env python3
"""
Configuration Manager for Sentiment Analysis System

This module handles secure loading and validation of configuration settings
from environment variables, including OpenAI API credentials and other
application settings.
"""

import os
import sys
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Configuration:
    """
    Data class to hold all configuration settings for the application.
    """
    openai_api_key: str
    openai_org_id: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    api_timeout: int = 30
    max_retries: int = 3


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class ConfigurationManager:
    """
    Manages application configuration by loading and validating environment variables.
    
    This class handles secure loading of API keys and other settings from environment
    variables, with proper validation and error handling.
    """
    
    def __init__(self, env_file: str = ".env"):
        """
        Initialize the configuration manager.
        
        Args:
            env_file: Path to the .env file to load (default: ".env")
        """
        self.env_file = env_file
        self._config: Optional[Configuration] = None
        
    def load_configuration(self) -> Configuration:
        """
        Load and validate configuration from environment variables.
        
        Returns:
            Configuration object with all validated settings
            
        Raises:
            ConfigurationError: If required configuration is missing or invalid
        """
        # Load environment variables from .env file if it exists
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
        
        # Load required settings
        api_key = self._load_api_key()
        
        # Load optional settings with defaults
        org_id = os.getenv("OPENAI_ORG_ID")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        timeout = self._load_int_setting("API_TIMEOUT", 30)
        max_retries = self._load_int_setting("MAX_RETRIES", 3)
        
        # Validate settings
        self._validate_configuration(api_key, model, timeout, max_retries)
        
        # Create and cache configuration
        self._config = Configuration(
            openai_api_key=api_key,
            openai_org_id=org_id,
            openai_model=model,
            api_timeout=timeout,
            max_retries=max_retries
        )
        
        return self._config
    
    def get_configuration(self) -> Configuration:
        """
        Get the current configuration, loading it if not already loaded.
        
        Returns:
            Configuration object with all settings
        """
        if self._config is None:
            return self.load_configuration()
        return self._config
    
    def _load_api_key(self) -> str:
        """
        Load and validate the OpenAI API key from environment variables.
        
        Returns:
            The API key string
            
        Raises:
            ConfigurationError: If API key is missing or invalid
        """
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key is required but not found. "
                "Please set the OPENAI_API_KEY environment variable or create a .env file. "
                "See .env.example for the required format."
            )
        
        # Basic validation of API key format
        api_key = api_key.strip()
        if len(api_key) < 10:  # OpenAI keys are much longer
            raise ConfigurationError(
                "OpenAI API key appears to be invalid (too short). "
                "Please check your API key and try again."
            )
        
        if api_key == "your_openai_api_key_here":
            raise ConfigurationError(
                "Please replace the placeholder API key with your actual OpenAI API key. "
                "You can get one from https://platform.openai.com/api-keys"
            )
        
        return api_key
    
    def _load_int_setting(self, env_var: str, default: int) -> int:
        """
        Load an integer setting from environment variables with validation.
        
        Args:
            env_var: Environment variable name
            default: Default value if not set
            
        Returns:
            The integer value
            
        Raises:
            ConfigurationError: If the value is not a valid integer
        """
        value_str = os.getenv(env_var)
        if value_str is None:
            return default
        
        try:
            value = int(value_str)
            if value <= 0:
                raise ValueError("Value must be positive")
            return value
        except ValueError as e:
            raise ConfigurationError(
                f"Invalid value for {env_var}: '{value_str}'. "
                f"Expected a positive integer. Error: {e}"
            )
    
    def _validate_configuration(self, api_key: str, model: str, timeout: int, max_retries: int) -> None:
        """
        Validate all configuration settings.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model name
            timeout: API timeout in seconds
            max_retries: Maximum number of retries
            
        Raises:
            ConfigurationError: If any setting is invalid
        """
        # Validate model name
        valid_models = [
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
            "gpt-4", "gpt-4-32k", "gpt-4-turbo-preview",
            "gpt-4o", "gpt-4o-mini",
            "gpt-5-nano"  # Latest GPT-5 nano model
        ]
        if model not in valid_models:
            print(f"Warning: Model '{model}' is not in the list of known models. "
                  f"Valid models: {', '.join(valid_models)}")
        
        # Validate timeout
        if timeout < 5 or timeout > 300:
            raise ConfigurationError(
                f"API timeout must be between 5 and 300 seconds, got {timeout}"
            )
        
        # Validate max retries
        if max_retries < 0 or max_retries > 10:
            raise ConfigurationError(
                f"Max retries must be between 0 and 10, got {max_retries}"
            )
    
    def display_setup_instructions(self) -> None:
        """
        Display setup instructions for configuring the application.
        """
        print("\n" + "=" * 60)
        print("CONFIGURATION SETUP REQUIRED")
        print("=" * 60)
        print("\nTo use this application, you need to configure your OpenAI API key:")
        print("\n1. Get an API key from: https://platform.openai.com/api-keys")
        print("2. Copy .env.example to .env:")
        print("   cp .env.example .env")
        print("3. Edit .env and replace 'your_openai_api_key_here' with your actual API key")
        print("4. Run the application again")
        print("\nAlternatively, you can set the environment variable directly:")
        print("   export OPENAI_API_KEY=your_actual_api_key")
        print("\n" + "=" * 60)
    
    def get_safe_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of configuration settings with sensitive data masked.
        
        Returns:
            Dictionary with configuration summary (API key masked)
        """
        if self._config is None:
            return {"status": "not_loaded"}
        
        return {
            "api_key_configured": bool(self._config.openai_api_key),
            "api_key_preview": f"{self._config.openai_api_key[:8]}..." if self._config.openai_api_key else None,
            "organization_id": self._config.openai_org_id,
            "model": self._config.openai_model,
            "timeout": self._config.api_timeout,
            "max_retries": self._config.max_retries
        }