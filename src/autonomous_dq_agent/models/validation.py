"""Validation and expectation models."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

__version__ = "0.1.0"


class ExpectationType(str, enum.Enum):
    """Supported expectation types."""

    NOT_NULL = "expect_column_values_to_not_be_null"
    NULL_RATE_BELOW = "expect_column_null_rate_to_be_below"
    BETWEEN = "expect_column_values_to_be_between"
    IN_SET = "expect_column_values_to_be_in_set"
    MATCH_REGEX = "expect_column_values_to_match_regex"
    MEAN_BETWEEN = "expect_column_mean_to_be_between"
    MIN_GTE = "expect_column_min_to_be_gte"
    MAX_LTE = "expect_column_max_to_be_lte"
    UNIQUE = "expect_column_values_to_be_unique"
    ROW_COUNT_BETWEEN = "expect_table_row_count_to_be_between"
    COLUMN_EXISTS = "expect_column_to_exist"
    DISTINCT_COUNT_GTE = "expect_column_distinct_count_to_be_gte"
    STD_BETWEEN = "expect_column_std_to_be_between"
    QUANTILE_BETWEEN = "expect_column_quantile_values_to_be_between"
    NOT_IN_SET = "expect_column_values_to_not_be_in_set"


class Expectation(BaseModel):
    """A single data quality expectation."""

    expectation_type: ExpectationType
    column: Optional[str] = Field(
        None, description="Target column (None for table-level)"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Expectation parameters"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata (notes, owner, severity)"
    )

    @property
    def description(self) -> str:
        """Human-readable expectation description."""
        col = f"[{self.column}]" if self.column else "[table]"
        return f"{col} {self.expectation_type.value} {self.kwargs}"


class ExpectationSuite(BaseModel):
    """Named collection of expectations for a dataset."""

    suite_name: str = Field(default="default_suite")
    dataset_name: str = Field(default="dataset")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expectations: List[Expectation] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

    def add_expectation(self, expectation: Expectation) -> None:
        """Append an expectation to the suite."""
        self.expectations.append(expectation)

    def filter_by_column(self, column: str) -> List[Expectation]:
        """Return expectations targeting a specific column."""
        return [e for e in self.expectations if e.column == column]

    def filter_by_type(self, etype: ExpectationType) -> List[Expectation]:
        """Return expectations of a specific type."""
        return [e for e in self.expectations if e.expectation_type == etype]


class ExpectationResult(BaseModel):
    """Result of evaluating a single expectation."""

    expectation: Expectation
    success: bool
    observed_value: Optional[Any] = None
    element_count: int = Field(default=0)
    unexpected_count: int = Field(default=0)
    unexpected_percent: float = Field(default=0.0)
    unexpected_values: List[Any] = Field(default_factory=list)
    error_message: Optional[str] = None

    @property
    def pass_rate(self) -> float:
        """Fraction of values that passed the expectation."""
        if self.element_count == 0:
            return 1.0
        return 1.0 - (self.unexpected_count / self.element_count)


class ValidationResult(BaseModel):
    """Aggregated result of running an ExpectationSuite against a dataset."""

    suite_name: str
    dataset_name: str
    validated_at: datetime = Field(default_factory=datetime.utcnow)
    success: bool
    statistics: Dict[str, Any] = Field(default_factory=dict)
    results: List[ExpectationResult] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

    @property
    def evaluated_expectations(self) -> int:
        """Total number of expectations evaluated."""
        return len(self.results)

    @property
    def successful_expectations(self) -> int:
        """Number of expectations that passed."""
        return sum(1 for r in self.results if r.success)

    @property
    def failed_expectations(self) -> int:
        """Number of expectations that failed."""
        return sum(1 for r in self.results if not r.success)

    @property
    def success_percent(self) -> float:
        """Fraction of expectations that passed."""
        if not self.results:
            return 100.0
        return (self.successful_expectations / self.evaluated_expectations) * 100.0

    def failed_results(self) -> List[ExpectationResult]:
        """Return only failed expectation results."""
        return [r for r in self.results if not r.success]
