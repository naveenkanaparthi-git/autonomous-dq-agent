"""DataValidator — runs ExpectationSuites against DataFrames."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from autonomous_dq_agent.models.validation import (
    Expectation,
    ExpectationResult,
    ExpectationSuite,
    ExpectationType,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class DataValidator:
    """Evaluates an ExpectationSuite against a pandas DataFrame.

    Each expectation type is dispatched to a dedicated handler method.
    Results are aggregated into a ValidationResult.
    """

    def validate(self, df: pd.DataFrame, suite: ExpectationSuite) -> ValidationResult:
        """Run all expectations in the suite against df.

        Args:
            df: DataFrame to validate.
            suite: ExpectationSuite containing expectations to evaluate.

        Returns:
            ValidationResult with per-expectation outcomes.
        """
        logger.info(
            "Validating '%s' against suite '%s' (%d expectations)",
            suite.dataset_name,
            suite.suite_name,
            len(suite.expectations),
        )
        results: List[ExpectationResult] = []
        for expectation in suite.expectations:
            result = self._evaluate(df, expectation)
            results.append(result)

        success = all(r.success for r in results)
        total = len(results)
        passed = sum(1 for r in results if r.success)
        failed = total - passed

        return ValidationResult(
            suite_name=suite.suite_name,
            dataset_name=suite.dataset_name,
            validated_at=datetime.utcnow(),
            success=success,
            statistics={
                "evaluated_expectations": total,
                "successful_expectations": passed,
                "unsuccessful_expectations": failed,
                "success_percent": round((passed / max(total, 1)) * 100, 2),
            },
            results=results,
        )

    def _evaluate(self, df: pd.DataFrame, exp: Expectation) -> ExpectationResult:
        """Dispatch a single expectation to its handler."""
        try:
            handler_map = {
                ExpectationType.NOT_NULL: self._expect_not_null,
                ExpectationType.NULL_RATE_BELOW: self._expect_null_rate_below,
                ExpectationType.BETWEEN: self._expect_values_between,
                ExpectationType.IN_SET: self._expect_values_in_set,
                ExpectationType.NOT_IN_SET: self._expect_values_not_in_set,
                ExpectationType.MATCH_REGEX: self._expect_match_regex,
                ExpectationType.MEAN_BETWEEN: self._expect_mean_between,
                ExpectationType.MIN_GTE: self._expect_min_gte,
                ExpectationType.MAX_LTE: self._expect_max_lte,
                ExpectationType.UNIQUE: self._expect_unique,
                ExpectationType.ROW_COUNT_BETWEEN: self._expect_row_count_between,
                ExpectationType.COLUMN_EXISTS: self._expect_column_exists,
                ExpectationType.DISTINCT_COUNT_GTE: self._expect_distinct_count_gte,
                ExpectationType.STD_BETWEEN: self._expect_std_between,
                ExpectationType.QUANTILE_BETWEEN: self._expect_quantile_between,
            }
            handler = handler_map.get(exp.expectation_type)
            if handler is None:
                return ExpectationResult(
                    expectation=exp,
                    success=False,
                    error_message=f"Unknown expectation type: {exp.expectation_type}",
                )
            return handler(df, exp)
        except Exception as exc:
            logger.warning(
                "Expectation %s failed with error: %s", exp.expectation_type, exc
            )
            return ExpectationResult(
                expectation=exp,
                success=False,
                error_message=str(exc),
            )

    def _get_col(self, df: pd.DataFrame, exp: Expectation) -> pd.Series:
        """Retrieve series by column name, raising ValueError if missing."""
        col = exp.column
        if col is None:
            raise ValueError("Expectation requires a column name but none provided.")
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        return df[col]

    def _expect_not_null(self, df: pd.DataFrame, exp: Expectation) -> ExpectationResult:
        """All values must be non-null."""
        series = self._get_col(df, exp)
        null_mask = series.isna()
        unexpected = null_mask.sum()
        return ExpectationResult(
            expectation=exp,
            success=bool(unexpected == 0),
            observed_value=int(unexpected),
            element_count=len(series),
            unexpected_count=int(unexpected),
            unexpected_percent=round(float(unexpected / max(len(series), 1)) * 100, 4),
        )

    def _expect_null_rate_below(
        self, df: pd.DataFrame, exp: Expectation
    ) -> ExpectationResult:
        """Null rate must be below max_null_rate."""
        series = self._get_col(df, exp)
        max_rate = float(exp.kwargs.get("max_null_rate", 0.0))
        null_rate = series.isna().mean()
        return ExpectationResult(
            expectation=exp,
            success=bool(null_rate <= max_rate),
            observed_value=round(float(null_rate), 6),
            element_count=len(series),
            unexpected_count=int(series.isna().sum()),
            unexpected_percent=round(float(null_rate) * 100, 4),
        )

    def _expect_values_between(
        self, df: pd.DataFrame, exp: Expectation
    ) -> ExpectationResult:
        """Numeric values must be within [min_value, max_value]."""
        series = self._get_col(df, exp)
        min_val = exp.kwargs.get("min_value")
        max_val = exp.kwargs.get("max_value")
        non_null = series.dropna()
        if len(non_null) == 0:
            return ExpectationResult(expectation=exp, success=True, element_count=0)

        numeric = pd.to_numeric(non_null, errors="coerce")
        mask = pd.Series(True, index=numeric.index)
        if min_val is not None:
            mask &= numeric >= min_val
        if max_val is not None:
            mask &= numeric <= max_val
        unexpected = int((~mask).sum())
        unexpected_vals = non_null[~mask].head(10).tolist()
        return ExpectationResult(
            expectation=exp,
            success=bool(unexpected == 0),
            observed_value={"min": float(numeric.min()), "max": float(numeric.max())},
            element_count=len(non_null),
            unexpected_count=unexpected,
            unexpected_percent=round(unexpected / max(len(non_null), 1) * 100, 4),
            unexpected_values=unexpected_vals,
        )

    def _expect_values_in_set(
        self, df: pd.DataFrame, exp: Expectation
    ) -> ExpectationResult:
        """Non-null values must be in value_set."""
        series = self._get_col(df, exp)
        value_set = set(exp.kwargs.get("value_set", []))
        non_null = series.dropna()
        mask = non_null.isin(value_set)
        unexpected = int((~mask).sum())
        unexpected_vals = non_null[~mask].head(10).tolist()
        return ExpectationResult(
            expectation=exp,
            success=bool(unexpected == 0),
            observed_value=non_null.nunique(),
            element_count=len(non_null),
            unexpected_count=unexpected,
            unexpected_percent=round(unexpected / max(len(non_null), 1) * 100, 4),
            unexpected_values=unexpected_vals,
        )

    def _expect_values_not_in_set(
        self, df: pd.DataFrame, exp: Expectation
    ) -> ExpectationResult:
        """Non-null values must NOT be in forbidden_set."""
        series = self._get_col(df, exp)
        forbidden = set(exp.kwargs.get("forbidden_set", []))
        non_null = series.dropna()
        mask = non_null.isin(forbidden)
        unexpected = int(mask.sum())
        unexpected_vals = non_null[mask].head(10).tolist()
        return ExpectationResult(
            expectation=exp,
            success=bool(unexpected == 0),
            element_count=len(non_null),
            unexpected_count=unexpected,
            unexpected_percent=round(unexpected / max(len(non_null), 1) * 100, 4),
            unexpected_values=unexpected_vals,
        )

    def _expect_match_regex(
        self, df: pd.DataFrame, exp: Expectation
    ) -> ExpectationResult:
        """Non-null string values must match regex pattern."""
        series = self._get_col(df, exp)
        pattern = str(exp.kwargs.get("regex", ".*"))
        compiled = re.compile(pattern)
        non_null = series.dropna().astype(str)
        mask = non_null.str.match(compiled)
        unexpected = int((~mask).sum())
        return ExpectationResult(
            expectation=exp,
            success=bool(unexpected == 0),
            observed_value=pattern,
            element_count=len(non_null),
            unexpected_count=unexpected,
            unexpected_percent=round(unexpected / max(len(non_null), 1) * 100, 4),
            unexpected_values=non_null[~mask].head(10).tolist(),
        )

    def _expect_mean_between(
        self, df: pd.DataFrame, exp: Expectation
    ) -> ExpectationResult:
        """Column mean must be within [min_value, max_value]."""
        series = self._get_col(df, exp)
        min_val = exp.kwargs.get("min_value")
        max_val = exp.kwargs.get("max_value")
        numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(numeric) == 0:
            return ExpectationResult(
                expectation=exp, success=False, error_message="No numeric values."
            )
        mean_val = float(numeric.mean())
        success = True
        if min_val is not None:
            success = success and mean_val >= min_val
        if max_val is not None:
            success = success and mean_val <= max_val
        return ExpectationResult(
            expectation=exp,
            success=success,
            observed_value=round(mean_val, 6),
            element_count=len(numeric),
        )

    def _expect_min_gte(self, df: pd.DataFrame, exp: Expectation) -> ExpectationResult:
        """Column minimum must be >= min_value."""
        series = self._get_col(df, exp)
        min_val = float(exp.kwargs.get("min_value", float("-inf")))
        numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(numeric) == 0:
            return ExpectationResult(
                expectation=exp, success=False, error_message="No numeric values."
            )
        actual_min = float(numeric.min())
        return ExpectationResult(
            expectation=exp,
            success=actual_min >= min_val,
            observed_value=actual_min,
            element_count=len(numeric),
        )

    def _expect_max_lte(self, df: pd.DataFrame, exp: Expectation) -> ExpectationResult:
        """Column maximum must be <= max_value."""
        series = self._get_col(df, exp)
        max_val = float(exp.kwargs.get("max_value", float("inf")))
        numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(numeric) == 0:
            return ExpectationResult(
                expectation=exp, success=False, error_message="No numeric values."
            )
        actual_max = float(numeric.max())
        return ExpectationResult(
            expectation=exp,
            success=actual_max <= max_val,
            observed_value=actual_max,
            element_count=len(numeric),
        )

    def _expect_unique(self, df: pd.DataFrame, exp: Expectation) -> ExpectationResult:
        """All non-null values must be unique."""
        series = self._get_col(df, exp)
        non_null = series.dropna()
        dups = non_null[non_null.duplicated()]
        unexpected = int(len(dups))
        return ExpectationResult(
            expectation=exp,
            success=bool(unexpected == 0),
            observed_value=non_null.nunique(),
            element_count=len(non_null),
            unexpected_count=unexpected,
            unexpected_percent=round(unexpected / max(len(non_null), 1) * 100, 4),
            unexpected_values=dups.head(10).tolist(),
        )

    def _expect_row_count_between(
        self, df: pd.DataFrame, exp: Expectation
    ) -> ExpectationResult:
        """Row count must be within [min_value, max_value]."""
        min_val = exp.kwargs.get("min_value")
        max_val = exp.kwargs.get("max_value")
        row_count = len(df)
        success = True
        if min_val is not None:
            success = success and row_count >= min_val
        if max_val is not None:
            success = success and row_count <= max_val
        return ExpectationResult(
            expectation=exp,
            success=success,
            observed_value=row_count,
            element_count=row_count,
        )

    def _expect_column_exists(
        self, df: pd.DataFrame, exp: Expectation
    ) -> ExpectationResult:
        """Column must exist in the DataFrame."""
        col = exp.column or exp.kwargs.get("column")
        exists = col in df.columns if col else False
        return ExpectationResult(
            expectation=exp,
            success=exists,
            observed_value=col,
            element_count=len(df),
        )

    def _expect_distinct_count_gte(
        self, df: pd.DataFrame, exp: Expectation
    ) -> ExpectationResult:
        """Number of distinct values must be >= min_count."""
        series = self._get_col(df, exp)
        min_count = int(exp.kwargs.get("min_count", 1))
        distinct = int(series.dropna().nunique())
        return ExpectationResult(
            expectation=exp,
            success=distinct >= min_count,
            observed_value=distinct,
            element_count=len(series),
        )

    def _expect_std_between(
        self, df: pd.DataFrame, exp: Expectation
    ) -> ExpectationResult:
        """Column standard deviation must be within [min_value, max_value]."""
        series = self._get_col(df, exp)
        min_val = exp.kwargs.get("min_value")
        max_val = exp.kwargs.get("max_value")
        numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(numeric) < 2:
            return ExpectationResult(
                expectation=exp,
                success=False,
                error_message="Insufficient data for std.",
            )
        std_val = float(numeric.std(ddof=1))
        success = True
        if min_val is not None:
            success = success and std_val >= min_val
        if max_val is not None:
            success = success and std_val <= max_val
        return ExpectationResult(
            expectation=exp,
            success=success,
            observed_value=round(std_val, 6),
            element_count=len(numeric),
        )

    def _expect_quantile_between(
        self, df: pd.DataFrame, exp: Expectation
    ) -> ExpectationResult:
        """A specified quantile must be within [min_value, max_value]."""
        series = self._get_col(df, exp)
        quantile = float(exp.kwargs.get("quantile", 0.5))
        min_val = exp.kwargs.get("min_value")
        max_val = exp.kwargs.get("max_value")
        numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(numeric) == 0:
            return ExpectationResult(
                expectation=exp, success=False, error_message="No numeric values."
            )
        q_val = float(np.percentile(numeric.values, quantile * 100))
        success = True
        if min_val is not None:
            success = success and q_val >= min_val
        if max_val is not None:
            success = success and q_val <= max_val
        return ExpectationResult(
            expectation=exp,
            success=success,
            observed_value=round(q_val, 6),
            element_count=len(numeric),
        )
