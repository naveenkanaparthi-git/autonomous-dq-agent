"""Shared pytest fixtures for autonomous DQ agent tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autonomous_dq_agent.config import Settings
from autonomous_dq_agent.core.profiler import DataProfiler
from autonomous_dq_agent.core.validator import DataValidator
from autonomous_dq_agent.models.profile import (
    DataProfile,
)
from autonomous_dq_agent.models.validation import (
    ExpectationSuite,
)
from autonomous_dq_agent.services.suite_builder import SuiteBuilder


@pytest.fixture
def settings() -> Settings:
    """Return Settings with safe test defaults (no API key)."""
    return Settings(
        anthropic_api_key=None,
        null_rate_critical_threshold=0.50,
        null_rate_high_threshold=0.20,
        null_rate_medium_threshold=0.05,
        outlier_iqr_multiplier=1.5,
        high_cardinality_threshold=0.95,
        skewness_threshold=2.0,
        duplicate_rate_threshold=0.01,
        correlation_threshold=0.95,
        max_sample_values=5,
    )


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A clean DataFrame with no quality issues."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "age": rng.integers(18, 80, size=n).astype(float),
            "income": rng.uniform(20_000, 150_000, size=n),
            "status": rng.choice(["active", "inactive", "pending"], size=n),
            "score": rng.uniform(0.0, 1.0, size=n),
            "region": rng.choice(["north", "south", "east", "west"], size=n),
        }
    )


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """A DataFrame with deliberate quality issues: nulls, outliers, duplicates."""
    rng = np.random.default_rng(7)
    n = 200
    age = rng.integers(18, 80, size=n).astype(float)
    age[rng.choice(n, size=30, replace=False)] = np.nan  # 15% null
    income = rng.uniform(20_000, 100_000, size=n)
    income[0] = 9_999_999  # extreme outlier
    income[1] = -500.0  # negative outlier
    status = rng.choice(["active", "inactive"], size=n).tolist()
    status[10] = None  # type: ignore[call-overload]

    df = pd.DataFrame(
        {
            "id": list(range(1, n + 1)),
            "age": age,
            "income": income,
            "status": status,
            "constant_col": ["X"] * n,
        }
    )
    # Inject 10 duplicates
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    return df


@pytest.fixture
def numeric_series() -> pd.Series:
    """A clean numeric series for unit tests."""
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(50.0, 10.0, 200), name="value")


@pytest.fixture
def categorical_series() -> pd.Series:
    """A clean low-cardinality categorical series."""
    return pd.Series(
        ["a", "b", "c", "a", "b", "a", "c", "b", "a", "b"] * 20,
        name="category",
    )


@pytest.fixture
def profiler(settings: Settings) -> DataProfiler:
    """DataProfiler with test settings."""
    return DataProfiler(settings=settings)


@pytest.fixture
def validator() -> DataValidator:
    """Default DataValidator instance."""
    return DataValidator()


@pytest.fixture
def sample_profile(profiler: DataProfiler, clean_df: pd.DataFrame) -> DataProfile:
    """DataProfile computed from clean_df."""
    return profiler.profile(clean_df, dataset_name="test_clean")


@pytest.fixture
def dirty_profile(profiler: DataProfiler, dirty_df: pd.DataFrame) -> DataProfile:
    """DataProfile computed from dirty_df."""
    return profiler.profile(dirty_df, dataset_name="test_dirty")


@pytest.fixture
def simple_suite() -> ExpectationSuite:
    """A minimal ExpectationSuite for unit tests."""
    return (
        SuiteBuilder("test_suite", "test_clean")
        .row_count_between(100, 10_000)
        .not_null("id")
        .unique("id")
        .between("age", 0, 120)
        .between("income", 0, 200_000)
        .in_set("status", ["active", "inactive", "pending"])
        .build()
    )


@pytest.fixture
def null_series() -> pd.Series:
    """A series with 60% null values."""
    vals = [1.0, 2.0, None, None, None, None, None, None, 3.0, 4.0]
    return pd.Series(vals * 10, name="mostly_null")
