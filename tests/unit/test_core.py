"""Unit tests for DataProfiler, DataValidator, QualityReporter, SuiteBuilder."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from autonomous_dq_agent.core.profiler import DataProfiler
from autonomous_dq_agent.core.reporter import QualityReporter
from autonomous_dq_agent.core.validator import DataValidator
from autonomous_dq_agent.models.profile import ColumnType, IssueType
from autonomous_dq_agent.models.validation import (
    Expectation,
    ExpectationSuite,
    ExpectationType,
)
from autonomous_dq_agent.services.suite_builder import SuiteBuilder

# ───────────────────────────────────────────────────────────────────────────────
# DataProfiler — column type detection
# ───────────────────────────────────────────────────────────────────────────────


class TestColumnTypeDetection:
    """Tests for DataProfiler._detect_column_type."""

    def test_numeric_int(self, profiler: DataProfiler) -> None:
        """Integer series detected as NUMERIC."""
        s = pd.Series([1, 2, 3, 4, 5], name="x")
        assert profiler._detect_column_type(s) == ColumnType.NUMERIC

    def test_numeric_float(self, profiler: DataProfiler) -> None:
        """Float series detected as NUMERIC."""
        s = pd.Series([1.1, 2.2, 3.3], name="x")
        assert profiler._detect_column_type(s) == ColumnType.NUMERIC

    def test_boolean(self, profiler: DataProfiler) -> None:
        """Bool series detected as BOOLEAN."""
        s = pd.Series([True, False, True], name="flag")
        assert profiler._detect_column_type(s) == ColumnType.BOOLEAN

    def test_datetime(self, profiler: DataProfiler) -> None:
        """Datetime64 series detected as DATETIME."""
        s = pd.to_datetime(pd.Series(["2024-01-01", "2024-06-15"]))
        assert profiler._detect_column_type(s) == ColumnType.DATETIME

    def test_categorical_low_cardinality(self, profiler: DataProfiler) -> None:
        """Low-cardinality object series detected as CATEGORICAL."""
        s = pd.Series(["a", "b", "a", "c", "b"] * 20, name="cat")
        assert profiler._detect_column_type(s) == ColumnType.CATEGORICAL

    def test_text_high_cardinality(self, profiler: DataProfiler) -> None:
        """High-cardinality long string series detected as TEXT."""
        s = pd.Series(
            [f"long description of item number {i} in the dataset" for i in range(100)],
            name="desc",
        )
        assert profiler._detect_column_type(s) == ColumnType.TEXT

    def test_unknown_all_null(self, profiler: DataProfiler) -> None:
        """All-null object series detected as UNKNOWN."""
        s = pd.Series([None, None, None], dtype=object, name="x")
        assert profiler._detect_column_type(s) == ColumnType.UNKNOWN


# ───────────────────────────────────────────────────────────────────────────────
# DataProfiler — numeric stats
# ───────────────────────────────────────────────────────────────────────────────


class TestNumericStats:
    """Tests for DataProfiler._compute_numeric_stats."""

    def test_basic_stats(
        self, profiler: DataProfiler, numeric_series: pd.Series
    ) -> None:
        """Mean, std, min, max should match pandas."""
        stats = profiler._compute_numeric_stats(numeric_series)
        assert math.isclose(stats.mean, float(numeric_series.mean()), rel_tol=1e-4)
        assert math.isclose(stats.std, float(numeric_series.std(ddof=1)), rel_tol=1e-4)
        assert stats.min == float(numeric_series.min())
        assert stats.max == float(numeric_series.max())

    def test_iqr_computed(
        self, profiler: DataProfiler, numeric_series: pd.Series
    ) -> None:
        """IQR should equal p75 - p25."""
        stats = profiler._compute_numeric_stats(numeric_series)
        assert math.isclose(stats.iqr, stats.p75 - stats.p25, rel_tol=1e-6)

    def test_outlier_count_with_spike(self, profiler: DataProfiler) -> None:
        """Extreme value should be counted as outlier."""
        s = pd.Series([10.0] * 100 + [10_000.0])  # clear outlier
        stats = profiler._compute_numeric_stats(s)
        assert stats.outlier_count >= 1

    def test_no_outliers_normal(
        self, profiler: DataProfiler, numeric_series: pd.Series
    ) -> None:
        """Normal distribution should have few outliers."""
        stats = profiler._compute_numeric_stats(numeric_series)
        outlier_pct = stats.outlier_count / len(numeric_series)
        assert outlier_pct < 0.10

    def test_zero_count(self, profiler: DataProfiler) -> None:
        """Zero values counted correctly."""
        s = pd.Series([0.0, 1.0, 2.0, 0.0, 3.0])
        stats = profiler._compute_numeric_stats(s)
        assert stats.zero_count == 2

    def test_negative_count(self, profiler: DataProfiler) -> None:
        """Negative values counted correctly."""
        s = pd.Series([-1.0, 0.0, 1.0, -2.0, 3.0])
        stats = profiler._compute_numeric_stats(s)
        assert stats.negative_count == 2

    def test_skewness_positive(self, profiler: DataProfiler) -> None:
        """Right-skewed data has positive skewness."""
        s = pd.Series([1.0] * 50 + [100.0, 200.0, 300.0])
        stats = profiler._compute_numeric_stats(s)
        assert stats.skewness > 0

    def test_percentiles_ordered(
        self, profiler: DataProfiler, numeric_series: pd.Series
    ) -> None:
        """Percentile ordering: p25 <= median <= p75 <= p95 <= p99."""
        stats = profiler._compute_numeric_stats(numeric_series)
        assert stats.p25 <= stats.median <= stats.p75 <= stats.p95 <= stats.p99


# ───────────────────────────────────────────────────────────────────────────────
# DataProfiler — categorical stats
# ───────────────────────────────────────────────────────────────────────────────


class TestCategoricalStats:
    """Tests for DataProfiler._compute_categorical_stats."""

    def test_unique_count(
        self, profiler: DataProfiler, categorical_series: pd.Series
    ) -> None:
        """Unique count matches series nunique."""
        stats = profiler._compute_categorical_stats(
            categorical_series, ColumnType.CATEGORICAL
        )
        assert stats.unique_count == categorical_series.nunique()

    def test_most_frequent(self, profiler: DataProfiler) -> None:
        """Most frequent value correctly identified."""
        s = pd.Series(["a", "a", "a", "b", "c"])
        stats = profiler._compute_categorical_stats(s, ColumnType.CATEGORICAL)
        assert stats.most_frequent == "a"

    def test_top_values_limited(self, profiler: DataProfiler) -> None:
        """Top values dict capped at 10 entries."""
        s = pd.Series([str(i % 50) for i in range(500)])
        stats = profiler._compute_categorical_stats(s, ColumnType.CATEGORICAL)
        assert len(stats.top_values) <= 10

    def test_cardinality_ratio(self, profiler: DataProfiler) -> None:
        """Cardinality ratio = unique / total."""
        s = pd.Series(["a", "b", "c", "d"])
        stats = profiler._compute_categorical_stats(s, ColumnType.CATEGORICAL)
        assert math.isclose(stats.cardinality_ratio, 1.0, rel_tol=1e-6)

    def test_text_avg_length_computed(self, profiler: DataProfiler) -> None:
        """Average length computed for TEXT columns."""
        s = pd.Series(["hello", "world", "foo"])
        stats = profiler._compute_categorical_stats(s, ColumnType.TEXT)
        assert stats.avg_length is not None
        assert stats.avg_length > 0


# ───────────────────────────────────────────────────────────────────────────────
# DataProfiler — full profile
# ───────────────────────────────────────────────────────────────────────────────


class TestDataProfiler:
    """Tests for DataProfiler.profile."""

    def test_row_column_count(self, sample_profile, clean_df: pd.DataFrame) -> None:
        """Profile row and column counts match DataFrame shape."""
        assert sample_profile.row_count == len(clean_df)
        assert sample_profile.column_count == len(clean_df.columns)

    def test_all_columns_profiled(self, sample_profile, clean_df: pd.DataFrame) -> None:
        """Every DataFrame column appears in profile.columns."""
        for col in clean_df.columns:
            assert col in sample_profile.columns

    def test_null_rate_zero_for_clean_data(self, sample_profile) -> None:
        """Clean DataFrame has zero null rates."""
        for col in sample_profile.columns.values():
            assert col.null_rate == 0.0

    def test_quality_score_range(self, sample_profile, dirty_profile) -> None:
        """Quality score is in [0, 100]."""
        assert 0.0 <= sample_profile.overall_quality_score <= 100.0
        assert 0.0 <= dirty_profile.overall_quality_score <= 100.0

    def test_dirty_score_lower_than_clean(self, sample_profile, dirty_profile) -> None:
        """Dirty data has lower quality score than clean data."""
        assert (
            dirty_profile.overall_quality_score < sample_profile.overall_quality_score
        )

    def test_duplicate_detection(self, dirty_profile) -> None:
        """Duplicates detected in dirty DataFrame."""
        assert dirty_profile.duplicate_row_count > 0

    def test_null_issues_detected(self, dirty_profile) -> None:
        """Null rate issues detected for age column."""
        null_issues = [
            i
            for i in dirty_profile.quality_issues
            if i.issue_type == IssueType.NULL_RATE
        ]
        assert len(null_issues) > 0

    def test_constant_column_detected(self, dirty_profile) -> None:
        """Constant column detected as CONSTANT_COLUMN issue."""
        constant_issues = [
            i
            for i in dirty_profile.quality_issues
            if i.issue_type == IssueType.CONSTANT_COLUMN
        ]
        assert len(constant_issues) > 0

    def test_correlations_computed(self, profiler: DataProfiler) -> None:
        """Highly correlated columns produce a correlation pair."""
        df = pd.DataFrame(
            {
                "x": range(100),
                "y": [i * 2 for i in range(100)],
            }
        )
        profile = profiler.profile(df)
        assert len(profile.correlations) == 1
        assert abs(profile.correlations[0].correlation - 1.0) < 1e-6

    def test_pk_candidate_detected(self, sample_profile) -> None:
        """Unique non-null column marked as PK candidate."""
        id_col = sample_profile.columns.get("id")
        assert id_col is not None
        assert id_col.is_primary_key_candidate

    def test_profile_dataset_name(
        self, profiler: DataProfiler, clean_df: pd.DataFrame
    ) -> None:
        """dataset_name propagated to profile."""
        p = profiler.profile(clean_df, dataset_name="my_table")
        assert p.dataset_name == "my_table"

    def test_memory_mb_positive(self, sample_profile) -> None:
        """Memory usage is positive."""
        assert sample_profile.memory_mb > 0

    def test_high_null_columns(self, dirty_profile) -> None:
        """high_null_columns returns age (15% nulls)."""
        high_null = dirty_profile.high_null_columns(threshold=0.10)
        assert "age" in high_null

    def test_numeric_columns_list(self, sample_profile) -> None:
        """numeric_columns returns at least age, income, score."""
        num_cols = sample_profile.numeric_columns()
        assert "age" in num_cols
        assert "income" in num_cols

    def test_categorical_columns_list(self, sample_profile) -> None:
        """categorical_columns returns status and region."""
        cat_cols = sample_profile.categorical_columns()
        assert "status" in cat_cols or "region" in cat_cols


# ───────────────────────────────────────────────────────────────────────────────
# DataValidator — individual expectations
# ───────────────────────────────────────────────────────────────────────────────


class TestDataValidator:
    """Tests for DataValidator expectation handlers."""

    def test_not_null_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """NOT_NULL passes when column has no nulls."""
        exp = Expectation(expectation_type=ExpectationType.NOT_NULL, column="id")
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_not_null_fails(
        self, validator: DataValidator, dirty_df: pd.DataFrame
    ) -> None:
        """NOT_NULL fails when column has nulls."""
        exp = Expectation(expectation_type=ExpectationType.NOT_NULL, column="age")
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(dirty_df, suite)
        assert not result.results[0].success

    def test_null_rate_below_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """NULL_RATE_BELOW passes for zero-null column."""
        exp = Expectation(
            expectation_type=ExpectationType.NULL_RATE_BELOW,
            column="id",
            kwargs={"max_null_rate": 0.1},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_null_rate_below_fails(
        self, validator: DataValidator, null_series: pd.Series
    ) -> None:
        """NULL_RATE_BELOW fails for high-null series."""
        df = pd.DataFrame({"col": null_series})
        exp = Expectation(
            expectation_type=ExpectationType.NULL_RATE_BELOW,
            column="col",
            kwargs={"max_null_rate": 0.05},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(df, suite)
        assert not result.results[0].success

    def test_between_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """BETWEEN passes when all values in range."""
        exp = Expectation(
            expectation_type=ExpectationType.BETWEEN,
            column="score",
            kwargs={"min_value": 0.0, "max_value": 1.0},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_between_fails(
        self, validator: DataValidator, dirty_df: pd.DataFrame
    ) -> None:
        """BETWEEN fails when outlier income exceeds max_value."""
        exp = Expectation(
            expectation_type=ExpectationType.BETWEEN,
            column="income",
            kwargs={"min_value": 0.0, "max_value": 200_000.0},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(dirty_df, suite)
        # income[0] = 9_999_999 exceeds max — should fail
        assert not result.results[0].success

    def test_in_set_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """IN_SET passes when all values in the set."""
        exp = Expectation(
            expectation_type=ExpectationType.IN_SET,
            column="status",
            kwargs={"value_set": ["active", "inactive", "pending"]},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_in_set_fails(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """IN_SET fails when value not in set."""
        exp = Expectation(
            expectation_type=ExpectationType.IN_SET,
            column="status",
            kwargs={"value_set": ["active"]},  # missing inactive, pending
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert not result.results[0].success

    def test_not_in_set_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """NOT_IN_SET passes when no forbidden values present."""
        exp = Expectation(
            expectation_type=ExpectationType.NOT_IN_SET,
            column="status",
            kwargs={"forbidden_set": ["deleted", "banned"]},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_not_in_set_fails(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """NOT_IN_SET fails when a forbidden value is present."""
        exp = Expectation(
            expectation_type=ExpectationType.NOT_IN_SET,
            column="status",
            kwargs={"forbidden_set": ["active"]},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert not result.results[0].success

    def test_match_regex_passes(self, validator: DataValidator) -> None:
        """MATCH_REGEX passes when all values match pattern."""
        df = pd.DataFrame({"email": ["a@b.com", "x@y.org", "hello@world.net"]})
        exp = Expectation(
            expectation_type=ExpectationType.MATCH_REGEX,
            column="email",
            kwargs={"regex": r".+@.+\..+"},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(df, suite)
        assert result.results[0].success

    def test_match_regex_fails(self, validator: DataValidator) -> None:
        """MATCH_REGEX fails when some values don't match."""
        df = pd.DataFrame({"email": ["a@b.com", "not_an_email"]})
        exp = Expectation(
            expectation_type=ExpectationType.MATCH_REGEX,
            column="email",
            kwargs={"regex": r".+@.+\..+"},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(df, suite)
        assert not result.results[0].success

    def test_mean_between_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """MEAN_BETWEEN passes when mean within range."""
        exp = Expectation(
            expectation_type=ExpectationType.MEAN_BETWEEN,
            column="score",
            kwargs={"min_value": 0.0, "max_value": 1.0},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_mean_between_fails(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """MEAN_BETWEEN fails when mean out of range."""
        exp = Expectation(
            expectation_type=ExpectationType.MEAN_BETWEEN,
            column="income",
            kwargs={"min_value": 0.0, "max_value": 100.0},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert not result.results[0].success

    def test_min_gte_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """MIN_GTE passes when actual min >= threshold."""
        exp = Expectation(
            expectation_type=ExpectationType.MIN_GTE,
            column="age",
            kwargs={"min_value": 0.0},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_max_lte_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """MAX_LTE passes when actual max <= threshold."""
        exp = Expectation(
            expectation_type=ExpectationType.MAX_LTE,
            column="score",
            kwargs={"max_value": 1.0},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_unique_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """UNIQUE passes for id column (all values unique)."""
        exp = Expectation(expectation_type=ExpectationType.UNIQUE, column="id")
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_unique_fails(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """UNIQUE fails for status column (repeated values)."""
        exp = Expectation(expectation_type=ExpectationType.UNIQUE, column="status")
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert not result.results[0].success

    def test_row_count_between_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """ROW_COUNT_BETWEEN passes for correct row count range."""
        exp = Expectation(
            expectation_type=ExpectationType.ROW_COUNT_BETWEEN,
            kwargs={"min_value": 100, "max_value": 10_000},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_row_count_between_fails(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """ROW_COUNT_BETWEEN fails when count outside range."""
        exp = Expectation(
            expectation_type=ExpectationType.ROW_COUNT_BETWEEN,
            kwargs={"min_value": 10_000, "max_value": 100_000},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert not result.results[0].success

    def test_column_exists_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """COLUMN_EXISTS passes for existing column."""
        exp = Expectation(
            expectation_type=ExpectationType.COLUMN_EXISTS,
            column="id",
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_column_exists_fails(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """COLUMN_EXISTS fails for missing column."""
        exp = Expectation(
            expectation_type=ExpectationType.COLUMN_EXISTS,
            column="nonexistent_column",
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert not result.results[0].success

    def test_distinct_count_gte_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """DISTINCT_COUNT_GTE passes when distinct count sufficient."""
        exp = Expectation(
            expectation_type=ExpectationType.DISTINCT_COUNT_GTE,
            column="status",
            kwargs={"min_count": 2},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_std_between_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """STD_BETWEEN passes when std in range."""
        exp = Expectation(
            expectation_type=ExpectationType.STD_BETWEEN,
            column="score",
            kwargs={"min_value": 0.0, "max_value": 1.0},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_quantile_between_passes(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """QUANTILE_BETWEEN passes for median in range."""
        exp = Expectation(
            expectation_type=ExpectationType.QUANTILE_BETWEEN,
            column="score",
            kwargs={"quantile": 0.5, "min_value": 0.0, "max_value": 1.0},
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert result.results[0].success

    def test_missing_column_returns_error(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """Expectation on missing column returns error result."""
        exp = Expectation(
            expectation_type=ExpectationType.NOT_NULL,
            column="does_not_exist",
        )
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        assert not result.results[0].success
        assert result.results[0].error_message is not None

    def test_validation_aggregation(
        self,
        validator: DataValidator,
        clean_df: pd.DataFrame,
        simple_suite: ExpectationSuite,
    ) -> None:
        """ValidationResult statistics match individual results."""
        result = validator.validate(clean_df, simple_suite)
        assert result.evaluated_expectations == len(simple_suite.expectations)
        assert (
            result.successful_expectations + result.failed_expectations
            == result.evaluated_expectations
        )

    def test_pass_rate_property(
        self, validator: DataValidator, clean_df: pd.DataFrame
    ) -> None:
        """ExpectationResult.pass_rate is in [0, 1]."""
        exp = Expectation(expectation_type=ExpectationType.NOT_NULL, column="id")
        suite = ExpectationSuite(suite_name="s", dataset_name="d", expectations=[exp])
        result = validator.validate(clean_df, suite)
        for r in result.results:
            assert 0.0 <= r.pass_rate <= 1.0


# ───────────────────────────────────────────────────────────────────────────────
# SuiteBuilder
# ───────────────────────────────────────────────────────────────────────────────


class TestSuiteBuilder:
    """Tests for the fluent SuiteBuilder API."""

    def test_build_returns_suite(self) -> None:
        """build() returns an ExpectationSuite."""
        suite = SuiteBuilder("s", "d").not_null("x").build()
        assert isinstance(suite, ExpectationSuite)

    def test_suite_name_propagated(self) -> None:
        """Suite name set on SuiteBuilder appears in suite."""
        suite = SuiteBuilder("my_suite", "my_data").build()
        assert suite.suite_name == "my_suite"
        assert suite.dataset_name == "my_data"

    def test_chaining_accumulates_expectations(self) -> None:
        """Each chained call adds one expectation."""
        builder = SuiteBuilder("s", "d")
        builder.not_null("a").unique("b").between("c", 0, 100).row_count_between(
            10, 1000
        )
        assert builder.expectation_count() == 4

    def test_columns_covered(self) -> None:
        """columns_covered returns columns with at least one expectation."""
        builder = SuiteBuilder("s", "d")
        builder.not_null("col1").unique("col2").row_count_between(1, 1000)
        covered = builder.columns_covered()
        assert "col1" in covered
        assert "col2" in covered

    def test_quantile_between(self) -> None:
        """quantile_between adds QUANTILE_BETWEEN expectation with correct kwargs."""
        suite = (
            SuiteBuilder("s", "d")
            .quantile_between("x", 0.75, min_value=5.0, max_value=10.0)
            .build()
        )
        exp = suite.expectations[0]
        assert exp.expectation_type == ExpectationType.QUANTILE_BETWEEN
        assert exp.kwargs["quantile"] == 0.75

    def test_filter_by_column(self) -> None:
        """filter_by_column returns only expectations for that column."""
        suite = SuiteBuilder("s", "d").not_null("a").unique("a").not_null("b").build()
        a_exps = suite.filter_by_column("a")
        assert len(a_exps) == 2

    def test_filter_by_type(self) -> None:
        """filter_by_type returns only expectations of that type."""
        suite = SuiteBuilder("s", "d").not_null("a").not_null("b").unique("c").build()
        nn_exps = suite.filter_by_type(ExpectationType.NOT_NULL)
        assert len(nn_exps) == 2

    def test_all_builder_methods_produce_expectations(self) -> None:
        """Every builder method appends exactly one expectation."""
        builder = SuiteBuilder("s", "d")
        [
            builder.not_null("col"),
            builder.null_rate_below("col", 0.1),
            builder.between("col", 0, 100),
            builder.in_set("col", ["a", "b"]),
            builder.not_in_set("col", ["x"]),
            builder.match_regex("col", r"\d+"),
            builder.mean_between("col", 0, 100),
            builder.min_gte("col", 0),
            builder.max_lte("col", 100),
            builder.unique("col"),
            builder.row_count_between(1, 1000),
            builder.column_exists("col"),
            builder.distinct_count_gte("col", 2),
            builder.std_between("col", 0, 50),
            builder.quantile_between("col", 0.5, 0, 100),
        ]
        assert builder.expectation_count() == 15


# ───────────────────────────────────────────────────────────────────────────────
# QualityReporter
# ───────────────────────────────────────────────────────────────────────────────


class TestQualityReporter:
    """Tests for QualityReporter output generation."""

    def test_text_report_contains_dataset_name(self, sample_profile) -> None:
        """Text report includes dataset name."""
        reporter = QualityReporter()
        text = reporter.generate_text_report(sample_profile)
        assert sample_profile.dataset_name in text

    def test_text_report_contains_score(self, sample_profile) -> None:
        """Text report contains quality score."""
        reporter = QualityReporter()
        text = reporter.generate_text_report(sample_profile)
        assert str(int(sample_profile.overall_quality_score)) in text

    def test_json_report_is_dict(self, sample_profile) -> None:
        """JSON report returns a dictionary."""
        reporter = QualityReporter()
        data = reporter.generate_json_report(sample_profile)
        assert isinstance(data, dict)
        assert data["dataset_name"] == sample_profile.dataset_name

    def test_json_report_serializable(self, sample_profile) -> None:
        """JSON report can be serialized via json.dumps."""
        reporter = QualityReporter()
        data = reporter.generate_json_report(sample_profile)
        text = json.dumps(data)
        assert len(text) > 0

    def test_html_report_contains_doctype(self, sample_profile) -> None:
        """HTML report starts with DOCTYPE."""
        reporter = QualityReporter()
        html = reporter.generate_html_report(sample_profile)
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_html_report_contains_dataset_name(self, sample_profile) -> None:
        """HTML report contains dataset name."""
        reporter = QualityReporter()
        html = reporter.generate_html_report(sample_profile)
        assert sample_profile.dataset_name in html

    def test_save_json_creates_file(self, sample_profile, tmp_path: Path) -> None:
        """save_json_report writes a file to disk."""
        reporter = QualityReporter(output_dir=str(tmp_path))
        path = reporter.save_json_report(sample_profile, filename="test_report.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["dataset_name"] == sample_profile.dataset_name

    def test_save_html_creates_file(self, sample_profile, tmp_path: Path) -> None:
        """save_html_report writes an HTML file to disk."""
        reporter = QualityReporter(output_dir=str(tmp_path))
        path = reporter.save_html_report(sample_profile, filename="test_report.html")
        assert path.exists()
        content = path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_validation_text_report_contains_pass(
        self,
        validator: DataValidator,
        clean_df: pd.DataFrame,
        simple_suite: ExpectationSuite,
    ) -> None:
        """Validation text report contains PASS for clean data."""
        result = validator.validate(clean_df, simple_suite)
        reporter = QualityReporter()
        text = reporter.generate_validation_text_report(result)
        assert "PASS" in text or "FAIL" in text

    def test_dirty_report_contains_issues(self, dirty_profile) -> None:
        """Dirty profile text report mentions issues."""
        reporter = QualityReporter()
        text = reporter.generate_text_report(dirty_profile)
        assert "CRITICAL" in text or "HIGH" in text or "MEDIUM" in text
