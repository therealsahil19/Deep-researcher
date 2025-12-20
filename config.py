"""
Configuration settings for Deep Research Agent.

This module centralizes all configurable parameters for the application,
including model settings, API endpoints, and search parameters.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for LLM models available via OpenRouter."""
    id: str
    name: str
    description: str
    is_free: bool = False

# Available models for selection
AVAILABLE_MODELS: List[ModelConfig] = [
    ModelConfig(
        id="alibaba/tongyi-deepresearch-30b-a3b:free",
        name="Tongyi DeepResearch 30B",
        description="Optimized for deep research tasks",
        is_free=True
    ),
    ModelConfig(
        id="openai/gpt-4o",
        name="GPT-4o",
        description="OpenAI's flagship model"
    ),
    ModelConfig(
        id="anthropic/claude-3.5-sonnet",
        name="Claude 3.5 Sonnet",
        description="Anthropic's balanced model"
    ),
    ModelConfig(
        id="meta-llama/llama-3.1-70b-instruct",
        name="Llama 3.1 70B",
        description="Meta's open-weight model"
    ),
    ModelConfig(
        id="google/gemini-2.0-flash-001",
        name="Gemini 2.0 Flash",
        description="Google's fast reasoning model"
    ),
]

DEFAULT_MODEL_ID = "alibaba/tongyi-deepresearch-30b-a3b:free"

def get_model_by_id(model_id: str) -> Optional[ModelConfig]:
    """Get a model configuration by its ID."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    return None

def get_model_choices() -> Dict[str, str]:
    """Returns a dict of {display_name: model_id} for UI dropdowns."""
    return {f"{m.name} {'(Free)' if m.is_free else ''}".strip(): m.id for m in AVAILABLE_MODELS}

# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================

@dataclass
class SearchConfig:
    """Configuration for search tools."""
    min_results: int = 3
    max_results: int = 10
    default_results: int = 5
    
    # Date range options for UI
    date_range_options: Dict[str, Optional[int]] = field(default_factory=lambda: {
        "Any Time": None,
        "Last 24 Hours": 1,
        "Last Week": 7,
        "Last Month": 30,
        "Last Year": 365,
    })

SEARCH_CONFIG = SearchConfig()

# =============================================================================
# RATE LIMITING
# =============================================================================

@dataclass
class RateLimitConfig:
    """Configuration for API rate limiting."""
    enabled: bool = True
    usage_file: str = "usage.json"
    daily_limit: int = 30
    monthly_limit: int = 1000

RATE_LIMIT_CONFIG = RateLimitConfig()

# =============================================================================
# API CONFIGURATION
# =============================================================================

@dataclass
class APIConfig:
    """Configuration for external API endpoints."""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    app_referer: str = "http://localhost:8501"
    app_title: str = "Deep Research Agent"
    request_timeout: int = 300  # 5 minutes timeout for long research tasks

API_CONFIG = APIConfig()

# =============================================================================
# REACT LOOP CONFIGURATION
# =============================================================================

@dataclass
class ReActConfig:
    """Configuration for the ReAct reasoning loop."""
    max_steps: int = 15
    
REACT_CONFIG = ReActConfig()

# =============================================================================
# ENVIRONMENT VARIABLE HELPERS
# =============================================================================

def get_env_api_key(key_name: str) -> Optional[str]:
    """Get an API key from environment variables."""
    value = os.environ.get(key_name)
    return value.strip() if value else None

# Environment variable names
ENV_OPENROUTER_KEY = "OPENROUTER_API_KEY"
ENV_TAVILY_KEY = "TAVILY_API_KEY"
ENV_EXA_KEY = "EXA_API_KEY"
