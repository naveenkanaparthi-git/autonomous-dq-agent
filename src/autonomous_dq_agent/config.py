"""Application settings loaded from environment variables."""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the autonomous DQ agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude AI suggestions",
    )
    claude_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model identifier to use",
    )
    claude_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens per Claude response",
    )
    null_rate_critical_threshold: float = Field(
        default=0.50,
        description="Null rate above which an issue is CRITICAL severity",
    )
    null_rate_high_threshold: float = Field(
        default=0.20,
        description="Null rate above which an issue is HIGH severity",
    )
    null_rate_medium_threshold: float = Field(
        default=0.05,
        description="Null rate above which an issue is MEDIUM severity",
    )
    outlier_iqr_multiplier: float = Field(
        default=1.5,
        description="IQR fence multiplier for outlier detection",
    )
    high_cardinality_threshold: float = Field(
        default=0.95,
        description="Cardinality ratio above which a column is 'high cardinality'",
    )
    skewness_threshold: float = Field(
        default=2.0,
        description="Absolute skewness above which distribution is flagged",
    )
    duplicate_rate_threshold: float = Field(
        default=0.01,
        description="Duplicate row rate above which an issue is raised",
    )
    correlation_threshold: float = Field(
        default=0.95,
        description="Pearson |r| above which columns are flagged as highly correlated",
    )
    max_sample_values: int = Field(
        default=5,
        description="Number of sample values to store per column",
    )
    report_output_dir: str = Field(
        default="reports",
        description="Directory to write HTML/JSON quality reports",
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level",
    )


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
