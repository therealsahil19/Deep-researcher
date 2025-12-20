"""
Unit tests for the Deep Research Agent utility functions.

Tests cover:
- extract_date_range_from_query: Date parsing from natural language
- generate_pdf: PDF generation and validity
- check_and_update_limit: Rate limiting logic
"""
import pytest
from unittest.mock import patch, MagicMock
import os
import json
import tempfile


# Import the module under test
import utils
import config


class TestExtractDateRangeFromQuery:
    """Tests for the extract_date_range_from_query function."""
    
    def test_month_year_format(self):
        """Test extraction of 'Month Year' format."""
        start, end = utils.extract_date_range_from_query("LLMs released in November 2025")
        assert start == "2025-11-01"
        assert end == "2025-11-30"
    
    def test_month_year_lowercase(self):
        """Test case-insensitive month parsing."""
        start, end = utils.extract_date_range_from_query("events in january 2024")
        assert start == "2024-01-01"
        assert end == "2024-01-31"
    
    def test_year_only(self):
        """Test extraction of year-only queries."""
        start, end = utils.extract_date_range_from_query("AI developments in 2025")
        assert start == "2025-01-01"
        assert end == "2025-12-31"
    
    def test_february_leap_year(self):
        """Test February in a leap year."""
        start, end = utils.extract_date_range_from_query("February 2024")
        assert start == "2024-02-01"
        assert end == "2024-02-29"  # 2024 is a leap year
    
    def test_february_non_leap_year(self):
        """Test February in a non-leap year."""
        start, end = utils.extract_date_range_from_query("February 2025")
        assert start == "2025-02-01"
        assert end == "2025-02-28"
    
    def test_no_date_in_query(self):
        """Test query with no date returns None."""
        start, end = utils.extract_date_range_from_query("What are the best AI models?")
        assert start is None
        assert end is None
    
    def test_date_extraction_with_slashes(self):
        """Test query with date containing slashes extracts the year."""
        # The regex r'\b(202[4-9]|203\d)\b' matches 2025 in "11/2025" because "/" acts as a separator.

        start, end = utils.extract_date_range_from_query("Release in 11/2025")
        # Now we expect it to find 2025
        assert start == "2025-01-01"
        assert end == "2025-12-31"


class TestGeneratePdf:
    """Tests for the generate_pdf function."""
    
    def test_generates_valid_pdf(self):
        """Test that output starts with PDF header."""
        test_content = "# Test Report\n\nThis is a test."
        pdf_bytes = utils.generate_pdf(test_content)
        
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:5] == b"%PDF-"
    
    def test_pdf_has_content(self):
        """Test that generated PDF has reasonable size."""
        test_content = "# Test Report\n\nThis is a comprehensive test with some content."
        pdf_bytes = utils.generate_pdf(test_content)
        
        # A valid PDF with content should be at least 1KB
        assert len(pdf_bytes) > 1000
    
    def test_handles_unicode(self):
        """Test that PDF handles Unicode characters."""
        test_content = "Test with Unicode: é, ñ, 中文, 日本語"
        # Should not raise an exception
        pdf_bytes = utils.generate_pdf(test_content)
        assert isinstance(pdf_bytes, bytes)
    
    def test_handles_empty_string(self):
        """Test PDF generation with empty string."""
        pdf_bytes = utils.generate_pdf("")
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:5] == b"%PDF-"


class TestCheckAndUpdateLimit:
    """Tests for the check_and_update_limit function."""
    
    @pytest.fixture
    def temp_usage_file(self, tmp_path):
        """Create a temporary usage file for testing."""
        usage_file = tmp_path / "test_usage.json"
        # Temporarily override the config
        original_file = config.RATE_LIMIT_CONFIG.usage_file
        config.RATE_LIMIT_CONFIG.usage_file = str(usage_file)
        yield usage_file
        # Restore original
        config.RATE_LIMIT_CONFIG.usage_file = original_file
    
    def test_allows_request_within_limit(self, temp_usage_file):
        """Test that requests within limit are allowed."""
        config.RATE_LIMIT_CONFIG.enabled = True
        allowed, message = utils.check_and_update_limit("tavily")
        assert allowed is True
        assert message == ""
    
    def test_creates_usage_file(self, temp_usage_file):
        """Test that usage file is created if it doesn't exist."""
        config.RATE_LIMIT_CONFIG.enabled = True
        utils.check_and_update_limit("tavily")
        assert temp_usage_file.exists()
    
    def test_increments_count(self, temp_usage_file):
        """Test that usage count is incremented."""
        config.RATE_LIMIT_CONFIG.enabled = True
        utils.check_and_update_limit("exa")
        
        with open(temp_usage_file, "r") as f:
            data = json.load(f)
        
        assert data["exa"]["daily_count"] == 1
        assert data["exa"]["monthly_count"] == 1
    
    def test_disabled_rate_limiting(self, temp_usage_file):
        """Test that disabled rate limiting always allows."""
        config.RATE_LIMIT_CONFIG.enabled = False
        allowed, message = utils.check_and_update_limit("tavily")
        assert allowed is True
        # Re-enable for other tests
        config.RATE_LIMIT_CONFIG.enabled = True


class TestConfigModule:
    """Tests for the config module."""
    
    def test_available_models_not_empty(self):
        """Test that there are available models."""
        assert len(config.AVAILABLE_MODELS) > 0
    
    def test_default_model_exists(self):
        """Test that default model is in available models."""
        model = config.get_model_by_id(config.DEFAULT_MODEL_ID)
        assert model is not None
    
    def test_get_model_choices_returns_dict(self):
        """Test that get_model_choices returns a valid dict."""
        choices = config.get_model_choices()
        assert isinstance(choices, dict)
        assert len(choices) == len(config.AVAILABLE_MODELS)
    
    def test_search_config_defaults(self):
        """Test search config has valid defaults."""
        assert config.SEARCH_CONFIG.min_results > 0
        assert config.SEARCH_CONFIG.max_results >= config.SEARCH_CONFIG.min_results
        assert config.SEARCH_CONFIG.default_results >= config.SEARCH_CONFIG.min_results
        assert config.SEARCH_CONFIG.default_results <= config.SEARCH_CONFIG.max_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
