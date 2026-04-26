"""SuiteBuilder — fluent builder for constructing ExpectationSuites."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from autonomous_dq_agent.models.validation import (
    Expectation,
    ExpectationSuite,
    ExpectationType,
)


class SuiteBuilder:
    """Fluent API for building ExpectationSuites programmatically.

    Usage::

        suite = (
            SuiteBuilder("my_suite", "orders")
            .not_null("order_id")
            .unique("order_id")
            .between("amount", 0.01, 100_000)
            .in_set("status", ["pending", "shipped", "delivered", "cancelled"])
            .row_count_between(1_000, 10_000_000)
            .build()
        )
    """

    def __init__(
        self, suite_name: str = "default_suite", dataset_name: str = "dataset"
    ) -> None:
        """Initialize builder with suite and dataset names."""
        self._suite_name = suite_name
        self._dataset_name = dataset_name
        self._expectations: List[Expectation] = []

    def _add(
        self,
        expectation_type: ExpectationType,
        column: Optional[str],
        kwargs: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> "SuiteBuilder":
        """Append an expectation and return self for chaining."""
        self._expectations.append(
            Expectation(
                expectation_type=expectation_type,
                column=column,
                kwargs=kwargs,
                meta=meta or {},
            )
        )
        return self

    def not_null(self, column: str) -> "SuiteBuilder":
        """Expect no null values in column."""
        return self._add(ExpectationType.NOT_NULL, column, {})

    def null_rate_below(self, column: str, max_null_rate: float) -> "SuiteBuilder":
        """Expect null rate <= max_null_rate."""
        return self._add(
            ExpectationType.NULL_RATE_BELOW, column, {"max_null_rate": max_null_rate}
        )

    def between(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> "SuiteBuilder":
        """Expect column values within [min_value, max_value]."""
        kwargs: Dict[str, Any] = {}
        if min_value is not None:
            kwargs["min_value"] = min_value
        if max_value is not None:
            kwargs["max_value"] = max_value
        return self._add(ExpectationType.BETWEEN, column, kwargs)

    def in_set(self, column: str, value_set: List[Any]) -> "SuiteBuilder":
        """Expect all non-null values to be within value_set."""
        return self._add(ExpectationType.IN_SET, column, {"value_set": value_set})

    def not_in_set(self, column: str, forbidden_set: List[Any]) -> "SuiteBuilder":
        """Expect no values in forbidden_set."""
        return self._add(
            ExpectationType.NOT_IN_SET, column, {"forbidden_set": forbidden_set}
        )

    def match_regex(self, column: str, regex: str) -> "SuiteBuilder":
        """Expect all non-null string values to match regex."""
        return self._add(ExpectationType.MATCH_REGEX, column, {"regex": regex})

    def mean_between(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> "SuiteBuilder":
        """Expect column mean within [min_value, max_value]."""
        kwargs: Dict[str, Any] = {}
        if min_value is not None:
            kwargs["min_value"] = min_value
        if max_value is not None:
            kwargs["max_value"] = max_value
        return self._add(ExpectationType.MEAN_BETWEEN, column, kwargs)

    def min_gte(self, column: str, min_value: float) -> "SuiteBuilder":
        """Expect column minimum >= min_value."""
        return self._add(ExpectationType.MIN_GTE, column, {"min_value": min_value})

    def max_lte(self, column: str, max_value: float) -> "SuiteBuilder":
        """Expect column maximum <= max_value."""
        return self._add(ExpectationType.MAX_LTE, column, {"max_value": max_value})

    def unique(self, column: str) -> "SuiteBuilder":
        """Expect all non-null values to be unique."""
        return self._add(ExpectationType.UNIQUE, column, {})

    def row_count_between(
        self, min_value: Optional[int] = None, max_value: Optional[int] = None
    ) -> "SuiteBuilder":
        """Expect total row count within [min_value, max_value]."""
        kwargs: Dict[str, Any] = {}
        if min_value is not None:
            kwargs["min_value"] = min_value
        if max_value is not None:
            kwargs["max_value"] = max_value
        return self._add(ExpectationType.ROW_COUNT_BETWEEN, None, kwargs)

    def column_exists(self, column: str) -> "SuiteBuilder":
        """Expect column to exist in the dataset."""
        return self._add(ExpectationType.COLUMN_EXISTS, column, {})

    def distinct_count_gte(self, column: str, min_count: int) -> "SuiteBuilder":
        """Expect distinct non-null value count >= min_count."""
        return self._add(
            ExpectationType.DISTINCT_COUNT_GTE, column, {"min_count": min_count}
        )

    def std_between(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> "SuiteBuilder":
        """Expect column std dev within [min_value, max_value]."""
        kwargs: Dict[str, Any] = {}
        if min_value is not None:
            kwargs["min_value"] = min_value
        if max_value is not None:
            kwargs["max_value"] = max_value
        return self._add(ExpectationType.STD_BETWEEN, column, kwargs)

    def quantile_between(
        self,
        column: str,
        quantile: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> "SuiteBuilder":
        """Expect a given quantile to be within [min_value, max_value]."""
        kwargs: Dict[str, Any] = {"quantile": quantile}
        if min_value is not None:
            kwargs["min_value"] = min_value
        if max_value is not None:
            kwargs["max_value"] = max_value
        return self._add(ExpectationType.QUANTILE_BETWEEN, column, kwargs)

    def build(self) -> ExpectationSuite:
        """Finalize and return the ExpectationSuite."""
        return ExpectationSuite(
            suite_name=self._suite_name,
            dataset_name=self._dataset_name,
            expectations=list(self._expectations),
        )

    def expectation_count(self) -> int:
        """Return the number of expectations added so far."""
        return len(self._expectations)

    def columns_covered(self) -> Set[str]:
        """Return set of column names that have at least one expectation."""
        return {e.column for e in self._expectations if e.column is not None}
