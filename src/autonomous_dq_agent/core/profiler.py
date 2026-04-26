"""DataProfiler — computes statistical profiles for pandas DataFrames."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from autonomous_dq_agent.config import Settings, get_settings
from autonomous_dq_agent.models.profile import (
    CategoricalStats,
    ColumnProfile,
    ColumnType,
    CorrelationPair,
    DataProfile,
    IssueSeverity,
    IssueType,
    NumericStats,
    QualityIssue,
)

logger = logging.getLogger(__name__)


class DataProfiler:
    """Produces a comprehensive statistical profile of a pandas DataFrame.

    The profiler computes per-column statistics, detects data quality issues,
    computes pairwise correlations among numeric columns, and derives an
    overall quality score.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize profiler with optional custom settings."""
        self.settings = settings or get_settings()

    def profile(self, df: pd.DataFrame, dataset_name: str = "dataset") -> DataProfile:
        """Profile a DataFrame and return a DataProfile.

        Args:
            df: Input DataFrame to profile.
            dataset_name: Logical name for the dataset.

        Returns:
            DataProfile with column stats, issues, and quality score.
        """
        logger.info(
            "Profiling dataset '%s' (%d rows, %d cols)",
            dataset_name,
            len(df),
            len(df.columns),
        )

        dup_count = int(df.duplicated().sum())
        dup_rate = dup_count / max(len(df), 1)

        columns: Dict[str, ColumnProfile] = {}
        for col in df.columns:
            columns[col] = self._profile_column(df[col])

        correlations = self._compute_correlations(df)
        issues = self._detect_quality_issues(
            df, columns, dup_count, dup_rate, correlations
        )
        score = self._compute_quality_score(issues, len(df.columns))

        return DataProfile(
            dataset_name=dataset_name,
            row_count=len(df),
            column_count=len(df.columns),
            memory_mb=round(df.memory_usage(deep=True).sum() / 1024 / 1024, 4),
            duplicate_row_count=dup_count,
            duplicate_row_rate=round(dup_rate, 6),
            columns=columns,
            correlations=correlations,
            quality_issues=issues,
            overall_quality_score=score,
        )

    def _detect_column_type(self, series: pd.Series) -> ColumnType:
        """Infer semantic column type from pandas dtype and content."""
        dtype = series.dtype
        if pd.api.types.is_bool_dtype(dtype):
            return ColumnType.BOOLEAN
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return ColumnType.DATETIME
        if pd.api.types.is_numeric_dtype(dtype):
            return ColumnType.NUMERIC
        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            non_null = series.dropna()
            if len(non_null) == 0:
                return ColumnType.UNKNOWN
            unique_ratio = non_null.nunique() / len(non_null)
            avg_len = non_null.astype(str).str.len().mean()
            if avg_len > 50 or unique_ratio > 0.8:
                return ColumnType.TEXT
            return ColumnType.CATEGORICAL
        return ColumnType.UNKNOWN

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        """Compute per-column profile statistics."""
        col_type = self._detect_column_type(series)
        row_count = len(series)
        null_count = int(series.isna().sum())
        null_rate = null_count / max(row_count, 1)
        non_null = series.dropna()
        distinct_count = int(non_null.nunique())

        sample_vals: List[Any] = []
        if len(non_null) > 0:
            n_sample = min(self.settings.max_sample_values, len(non_null))
            sample_vals = non_null.sample(n=n_sample, random_state=42).tolist()

        numeric_stats: Optional[NumericStats] = None
        categorical_stats: Optional[CategoricalStats] = None

        if col_type == ColumnType.NUMERIC and len(non_null) > 0:
            numeric_stats = self._compute_numeric_stats(non_null)
        elif col_type in (ColumnType.CATEGORICAL, ColumnType.TEXT, ColumnType.BOOLEAN):
            categorical_stats = self._compute_categorical_stats(non_null, col_type)

        is_constant = distinct_count <= 1
        is_pk_candidate = null_count == 0 and distinct_count == row_count

        return ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            column_type=col_type,
            row_count=row_count,
            null_count=null_count,
            null_rate=round(null_rate, 6),
            distinct_count=distinct_count,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
            sample_values=sample_vals,
            is_constant=is_constant,
            is_primary_key_candidate=is_pk_candidate,
        )

    def _compute_numeric_stats(self, series: pd.Series) -> NumericStats:
        """Compute descriptive statistics for a numeric series."""
        arr = series.astype(float).values
        p25 = float(np.percentile(arr, 25))
        p75 = float(np.percentile(arr, 75))
        iqr = p75 - p25
        fence_lo = p25 - self.settings.outlier_iqr_multiplier * iqr
        fence_hi = p75 + self.settings.outlier_iqr_multiplier * iqr
        outlier_count = int(np.sum((arr < fence_lo) | (arr > fence_hi)))

        skewness = float(scipy_stats.skew(arr))
        kurtosis = float(scipy_stats.kurtosis(arr))

        return NumericStats(
            mean=round(float(np.mean(arr)), 6),
            median=round(float(np.median(arr)), 6),
            std=round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 6),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            p25=round(p25, 6),
            p75=round(p75, 6),
            p95=round(float(np.percentile(arr, 95)), 6),
            p99=round(float(np.percentile(arr, 99)), 6),
            skewness=round(skewness, 6),
            kurtosis=round(kurtosis, 6),
            iqr=round(iqr, 6),
            outlier_count=outlier_count,
            zero_count=int(np.sum(arr == 0)),
            negative_count=int(np.sum(arr < 0)),
        )

    def _compute_categorical_stats(
        self, series: pd.Series, col_type: ColumnType
    ) -> CategoricalStats:
        """Compute frequency statistics for a categorical/text series."""
        str_series = series.astype(str)
        unique_count = int(series.nunique())
        total = max(len(series), 1)
        cardinality_ratio = unique_count / total

        value_counts = series.value_counts()
        top_values: Dict[str, int] = {
            str(k): int(v) for k, v in value_counts.head(10).items()
        }
        most_frequent = str(value_counts.index[0]) if len(value_counts) > 0 else None
        most_frequent_pct = (
            float(value_counts.iloc[0]) / total if len(value_counts) > 0 else 0.0
        )

        avg_len: Optional[float] = None
        max_len: Optional[int] = None
        if col_type == ColumnType.TEXT:
            lengths = str_series.str.len()
            avg_len = round(float(lengths.mean()), 2)
            max_len = int(lengths.max())

        return CategoricalStats(
            unique_count=unique_count,
            top_values=top_values,
            cardinality_ratio=round(cardinality_ratio, 6),
            most_frequent=most_frequent,
            most_frequent_pct=round(most_frequent_pct, 6),
            avg_length=avg_len,
            max_length=max_len,
        )

    def _compute_correlations(self, df: pd.DataFrame) -> List[CorrelationPair]:
        """Compute pairwise Pearson correlations for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return []

        corr_matrix = df[numeric_cols].corr(method="pearson")
        pairs: List[CorrelationPair] = []
        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1 :]:
                val = corr_matrix.loc[col_a, col_b]
                if not np.isnan(val):
                    pairs.append(
                        CorrelationPair(
                            col_a=col_a, col_b=col_b, correlation=round(float(val), 6)
                        )
                    )
        return pairs

    def _detect_quality_issues(
        self,
        df: pd.DataFrame,
        columns: Dict[str, ColumnProfile],
        dup_count: int,
        dup_rate: float,
        correlations: List[CorrelationPair],
    ) -> List[QualityIssue]:
        """Detect data quality issues across all columns and dataset-level."""
        issues: List[QualityIssue] = []
        cfg = self.settings

        # Dataset-level duplicate check
        if dup_rate > cfg.duplicate_rate_threshold:
            issues.append(
                QualityIssue(
                    issue_type=IssueType.DUPLICATE,
                    severity=(
                        IssueSeverity.HIGH if dup_rate > 0.10 else IssueSeverity.MEDIUM
                    ),
                    column=None,
                    description=f"{dup_count} duplicate rows detected ({dup_rate:.1%})",
                    metric_value=dup_rate,
                    threshold=cfg.duplicate_rate_threshold,
                    recommendation="De-duplicate with df.drop_duplicates() before loading.",
                )
            )

        for col_name, col in columns.items():
            # Null rate issues
            if col.null_rate >= cfg.null_rate_critical_threshold:
                issues.append(
                    QualityIssue(
                        issue_type=IssueType.NULL_RATE,
                        severity=IssueSeverity.CRITICAL,
                        column=col_name,
                        description=f"Column '{col_name}' has {col.null_rate:.1%} null rate (>= critical threshold {cfg.null_rate_critical_threshold:.0%})",
                        metric_value=col.null_rate,
                        threshold=cfg.null_rate_critical_threshold,
                        recommendation=f"Investigate source for '{col_name}'; consider imputation or dropping column.",
                    )
                )
            elif col.null_rate >= cfg.null_rate_high_threshold:
                issues.append(
                    QualityIssue(
                        issue_type=IssueType.NULL_RATE,
                        severity=IssueSeverity.HIGH,
                        column=col_name,
                        description=f"Column '{col_name}' has {col.null_rate:.1%} null rate",
                        metric_value=col.null_rate,
                        threshold=cfg.null_rate_high_threshold,
                        recommendation=f"Impute or flag nulls in '{col_name}'.",
                    )
                )
            elif col.null_rate >= cfg.null_rate_medium_threshold:
                issues.append(
                    QualityIssue(
                        issue_type=IssueType.NULL_RATE,
                        severity=IssueSeverity.MEDIUM,
                        column=col_name,
                        description=f"Column '{col_name}' has {col.null_rate:.1%} null rate",
                        metric_value=col.null_rate,
                        threshold=cfg.null_rate_medium_threshold,
                        recommendation=f"Monitor null rate trend for '{col_name}'.",
                    )
                )

            # Constant column
            if col.is_constant and col.null_rate < 1.0:
                issues.append(
                    QualityIssue(
                        issue_type=IssueType.CONSTANT_COLUMN,
                        severity=IssueSeverity.LOW,
                        column=col_name,
                        description=f"Column '{col_name}' has only one distinct value — likely a constant.",
                        metric_value=float(col.distinct_count),
                        recommendation=f"Drop constant column '{col_name}' to reduce schema noise.",
                    )
                )

            # Numeric-specific issues
            if col.numeric_stats is not None:
                ns = col.numeric_stats
                # Outliers
                if ns.outlier_count > 0:
                    outlier_pct = ns.outlier_count / max(
                        col.row_count - col.null_count, 1
                    )
                    sev = (
                        IssueSeverity.HIGH
                        if outlier_pct > 0.05
                        else IssueSeverity.MEDIUM
                    )
                    issues.append(
                        QualityIssue(
                            issue_type=IssueType.OUTLIER,
                            severity=sev,
                            column=col_name,
                            description=f"Column '{col_name}' has {ns.outlier_count} IQR-fence outliers ({outlier_pct:.1%})",
                            metric_value=outlier_pct,
                            recommendation=f"Cap or clip outliers in '{col_name}' using IQR fencing.",
                        )
                    )
                # Distribution skew
                if abs(ns.skewness) > cfg.skewness_threshold:
                    issues.append(
                        QualityIssue(
                            issue_type=IssueType.DISTRIBUTION_SKEW,
                            severity=IssueSeverity.LOW,
                            column=col_name,
                            description=f"Column '{col_name}' is highly skewed (skewness={ns.skewness:.2f})",
                            metric_value=ns.skewness,
                            threshold=cfg.skewness_threshold,
                            recommendation=f"Apply log/sqrt transform to '{col_name}' to reduce skew.",
                        )
                    )

            # Categorical-specific issues
            if col.categorical_stats is not None:
                cs = col.categorical_stats
                if cs.cardinality_ratio > cfg.high_cardinality_threshold:
                    issues.append(
                        QualityIssue(
                            issue_type=IssueType.HIGH_CARDINALITY,
                            severity=IssueSeverity.INFO,
                            column=col_name,
                            description=f"Column '{col_name}' has very high cardinality ({cs.unique_count} unique values, {cs.cardinality_ratio:.1%} of rows)",
                            metric_value=cs.cardinality_ratio,
                            threshold=cfg.high_cardinality_threshold,
                            recommendation=f"Consider hashing or embedding '{col_name}' instead of one-hot encoding.",
                        )
                    )

        # High correlation pairs
        for pair in correlations:
            if abs(pair.correlation) >= cfg.correlation_threshold:
                issues.append(
                    QualityIssue(
                        issue_type=IssueType.CARDINALITY,
                        severity=IssueSeverity.INFO,
                        column=f"{pair.col_a},{pair.col_b}",
                        description=f"Columns '{pair.col_a}' and '{pair.col_b}' are highly correlated (r={pair.correlation:.3f})",
                        metric_value=abs(pair.correlation),
                        threshold=cfg.correlation_threshold,
                        recommendation=f"Consider dropping one of '{pair.col_a}' or '{pair.col_b}' to reduce multicollinearity.",
                    )
                )

        return issues

    def _compute_quality_score(
        self, issues: List[QualityIssue], col_count: int
    ) -> float:
        """Derive a composite quality score [0, 100] from detected issues."""
        penalty_map = {
            IssueSeverity.CRITICAL: 20.0,
            IssueSeverity.HIGH: 10.0,
            IssueSeverity.MEDIUM: 5.0,
            IssueSeverity.LOW: 2.0,
            IssueSeverity.INFO: 0.5,
        }
        total_penalty = sum(penalty_map[i.severity] for i in issues)
        score = max(0.0, 100.0 - total_penalty)
        return round(score, 2)
