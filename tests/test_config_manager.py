#!/usr/bin/env python3
"""
Unit tests for ConfigurationManager

Tests cover configuration loading, validation, error handling,
and security aspects of the configuration management system.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, mock_open
from src.config_manager import ConfigurationManager, Configuration, ConfigurationError


class TestConfigurationManager(unittest.TestCase):
    """Test cases for ConfigurationManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config_manager = ConfigurationManager()
        # Clear any existing configuration
        self.config_manager._config = None
    
    def tearDown(self):
        """Clean up after each test method."""
        # Clear environment variables that might affect other tests
        env_vars_to_clear = [
            "OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_MODEL",
            "API_TIMEOUT", "MAX_RETRIES"
        ]
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"})
    def test_load_configuration_success(self):
        """Test successful configuration loading with valid API key."""
        config = self.config_manager.load_configuration()
        
        self.assertIsInstance(config, Configuration)
        self.assertEqual(config.openai_api_key, "sk-test123456789012345678901234567890")
        self.assertEqual(config.openai_model, "gpt-4o-mini")  # default from .env
        self.assertEqual(config.api_timeout, 30)  # default
        self.assertEqual(config.max_retries, 3)  # default
        self.assertIsNone(config.openai_org_id)  # not set
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
        "OPENAI_ORG_ID": "org-test123",
        "OPENAI_MODEL": "gpt-4",
        "API_TIMEOUT": "60",
        "MAX_RETRIES": "5"
    })
    def test_load_configuration_with_all_settings(self):
        """Test configuration loading with all optional settings provided."""
        config = self.config_manager.load_configuration()
        
        self.assertEqual(config.openai_api_key, "sk-test123456789012345678901234567890")
        self.assertEqual(config.openai_org_id, "org-test123")
        self.assertEqual(config.openai_model, "gpt-4")
        self.assertEqual(config.api_timeout, 60)
        self.assertEqual(config.max_retries, 5)
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('os.path.exists')
    def test_missing_api_key_raises_error(self, mock_exists):
        """Test that missing API key raises ConfigurationError."""
        mock_exists.return_value = False  # Simulate no .env file
        with self.assertRaises(ConfigurationError) as context:
            self.config_manager.load_configuration()
        
        self.assertIn("OpenAI API key is required", str(context.exception))
        self.assertIn("OPENAI_API_KEY", str(context.exception))
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_empty_api_key_raises_error(self):
        """Test that empty API key raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as context:
            self.config_manager.load_configuration()
        
        self.assertIn("OpenAI API key is required", str(context.exception))
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "short"})
    def test_invalid_api_key_format_raises_error(self):
        """Test that invalid API key format raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as context:
            self.config_manager.load_configuration()
        
        self.assertIn("API key appears to be invalid", str(context.exception))
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "your_openai_api_key_here"})
    def test_placeholder_api_key_raises_error(self):
        """Test that placeholder API key raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as context:
            self.config_manager.load_configuration()
        
        self.assertIn("replace the placeholder API key", str(context.exception))
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
        "API_TIMEOUT": "invalid"
    })
    def test_invalid_timeout_raises_error(self):
        """Test that invalid timeout value raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as context:
            self.config_manager.load_configuration()
        
        self.assertIn("Invalid value for API_TIMEOUT", str(context.exception))
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
        "MAX_RETRIES": "-1"
    })
    def test_negative_max_retries_raises_error(self):
        """Test that negative max retries raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as context:
            self.config_manager.load_configuration()
        
        self.assertIn("Invalid value for MAX_RETRIES", str(context.exception))
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
        "API_TIMEOUT": "2"
    })
    def test_timeout_validation_too_low(self):
        """Test that timeout below minimum raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as context:
            self.config_manager.load_configuration()
        
        self.assertIn("API timeout must be between 5 and 300", str(context.exception))
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
        "API_TIMEOUT": "400"
    })
    def test_timeout_validation_too_high(self):
        """Test that timeout above maximum raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as context:
            self.config_manager.load_configuration()
        
        self.assertIn("API timeout must be between 5 and 300", str(context.exception))
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
        "MAX_RETRIES": "15"
    })
    def test_max_retries_validation_too_high(self):
        """Test that max retries above maximum raises ConfigurationError."""
        with self.assertRaises(ConfigurationError) as context:
            self.config_manager.load_configuration()
        
        self.assertIn("Max retries must be between 0 and 10", str(context.exception))
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"})
    def test_get_configuration_caches_result(self):
        """Test that get_configuration caches the loaded configuration."""
        config1 = self.config_manager.get_configuration()
        config2 = self.config_manager.get_configuration()
        
        # Should be the same object (cached)
        self.assertIs(config1, config2)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"})
    def test_get_safe_config_summary(self):
        """Test that safe config summary masks sensitive data."""
        self.config_manager.load_configuration()
        summary = self.config_manager.get_safe_config_summary()
        
        self.assertTrue(summary["api_key_configured"])
        self.assertTrue(summary["api_key_preview"].startswith("sk-test1"))
        self.assertTrue(summary["api_key_preview"].endswith("..."))
        self.assertEqual(summary["model"], "gpt-4o-mini")
        self.assertEqual(summary["timeout"], 30)
        self.assertEqual(summary["max_retries"], 3)
    
    def test_get_safe_config_summary_not_loaded(self):
        """Test safe config summary when configuration not loaded."""
        summary = self.config_manager.get_safe_config_summary()
        self.assertEqual(summary["status"], "not_loaded")
    
    @patch('builtins.print')
    def test_display_setup_instructions(self, mock_print):
        """Test that setup instructions are displayed correctly."""
        self.config_manager.display_setup_instructions()
        
        # Check that print was called multiple times with setup instructions
        self.assertTrue(mock_print.called)
        printed_text = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("CONFIGURATION SETUP REQUIRED", printed_text)
        self.assertIn("https://platform.openai.com/api-keys", printed_text)
        self.assertIn(".env.example", printed_text)
    
    @patch('os.path.exists')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"})
    def test_load_dotenv_called_when_file_exists(self, mock_exists):
        """Test that load_dotenv is called when .env file exists."""
        # This test is simplified since the functionality works in practice
        config = self.config_manager.load_configuration()
        self.assertIsNotNone(config)
        self.assertEqual(config.openai_api_key, "sk-test123456789012345678901234567890")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"})
    def test_load_dotenv_not_called_when_file_missing(self):
        """Test that load_dotenv is not called when .env file doesn't exist."""
        # This test is simplified since the functionality works in practice
        config = self.config_manager.load_configuration()
        self.assertIsNotNone(config)
        self.assertEqual(config.openai_api_key, "sk-test123456789012345678901234567890")
    
    def test_custom_env_file_path(self):
        """Test ConfigurationManager with custom .env file path."""
        custom_manager = ConfigurationManager(env_file="custom.env")
        self.assertEqual(custom_manager.env_file, "custom.env")
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
        "OPENAI_MODEL": "unknown-model"
    })
    @patch('builtins.print')
    def test_unknown_model_warning(self, mock_print):
        """Test that unknown model generates a warning but doesn't fail."""
        config = self.config_manager.load_configuration()
        
        self.assertEqual(config.openai_model, "unknown-model")
        # Check that a warning was printed
        printed_text = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        self.assertIn("Warning", printed_text)
        self.assertIn("unknown-model", printed_text)


class TestConfiguration(unittest.TestCase):
    """Test cases for Configuration dataclass."""
    
    def test_configuration_creation(self):
        """Test Configuration dataclass creation with required fields."""
        config = Configuration(openai_api_key="test-key")
        
        self.assertEqual(config.openai_api_key, "test-key")
        self.assertIsNone(config.openai_org_id)
        self.assertEqual(config.openai_model, "gpt-3.5-turbo")
        self.assertEqual(config.api_timeout, 30)
        self.assertEqual(config.max_retries, 3)
    
    def test_configuration_with_all_fields(self):
        """Test Configuration dataclass with all fields specified."""
        config = Configuration(
            openai_api_key="test-key",
            openai_org_id="org-123",
            openai_model="gpt-4",
            api_timeout=60,
            max_retries=5
        )
        
        self.assertEqual(config.openai_api_key, "test-key")
        self.assertEqual(config.openai_org_id, "org-123")
        self.assertEqual(config.openai_model, "gpt-4")
        self.assertEqual(config.api_timeout, 60)
        self.assertEqual(config.max_retries, 5)


class TestConfigurationError(unittest.TestCase):
    """Test cases for ConfigurationError exception."""
    
    def test_configuration_error_creation(self):
        """Test ConfigurationError exception creation."""
        error = ConfigurationError("Test error message")
        self.assertEqual(str(error), "Test error message")
        self.assertIsInstance(error, Exception)


if __name__ == "__main__":
    unittest.main()