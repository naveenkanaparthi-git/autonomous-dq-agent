"""Data profile models for autonomous DQ agent."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

__version__ = "0.1.0"


class ColumnType(str, enum.Enum):
    """Detected column data type."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class IssueSeverity(str, enum.Enum):
    """Severity level for data quality issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueType(str, enum.Enum):
    """Category of data quality issue."""

    NULL_RATE = "null_rate"
    OUTLIER = "outlier"
    CARDINALITY = "cardinality"
    DISTRIBUTION_SKEW = "distribution_skew"
    DUPLICATE = "duplicate"
    SCHEMA_DRIFT = "schema_drift"
    CONSTANT_COLUMN = "constant_column"
    HIGH_CARDINALITY = "high_cardinality"
    MIXED_TYPES = "mixed_types"
    SUSPICIOUS_VALUE = "suspicious_value"


class NumericStats(BaseModel):
    """Statistical summary for numeric columns."""

    mean: float = Field(description="Arithmetic mean")
    median: float = Field(description="Median value (50th percentile)")
    std: float = Field(description="Standard deviation")
    min: float = Field(description="Minimum value")
    max: float = Field(description="Maximum value")
    p25: float = Field(description="25th percentile")
    p75: float = Field(description="75th percentile")
    p95: float = Field(description="95th percentile")
    p99: float = Field(description="99th percentile")
    skewness: float = Field(description="Distribution skewness")
    kurtosis: float = Field(description="Distribution kurtosis")
    iqr: float = Field(description="Interquartile range")
    outlier_count: int = Field(default=0, description="IQR-fence outlier count")
    zero_count: int = Field(default=0, description="Count of zero values")
    negative_count: int = Field(default=0, description="Count of negative values")


class CategoricalStats(BaseModel):
    """Statistical summary for categorical/text columns."""

    unique_count: int = Field(description="Number of distinct values")
    top_values: Dict[str, int] = Field(
        default_factory=dict, description="Top 10 values by frequency"
    )
    cardinality_ratio: float = Field(
        description="Ratio of unique values to total non-null rows"
    )
    most_frequent: Optional[str] = Field(None, description="Most frequent value")
    most_frequent_pct: float = Field(
        default=0.0, description="Fraction of rows with most frequent value"
    )
    avg_length: Optional[float] = Field(
        None, description="Average string length (text columns)"
    )
    max_length: Optional[int] = Field(None, description="Max string length")


class ColumnProfile(BaseModel):
    """Complete profile for a single column."""

    name: str = Field(description="Column name")
    dtype: str = Field(description="Pandas dtype string")
    column_type: ColumnType = Field(description="Detected semantic type")
    row_count: int = Field(description="Total rows including nulls")
    null_count: int = Field(description="Number of null/NaN values")
    null_rate: float = Field(description="Fraction of null values [0,1]")
    distinct_count: int = Field(description="Number of distinct non-null values")
    numeric_stats: Optional[NumericStats] = None
    categorical_stats: Optional[CategoricalStats] = None
    sample_values: List[Any] = Field(
        default_factory=list, description="Up to 5 representative sample values"
    )
    is_constant: bool = Field(
        default=False, description="True if column has only one distinct value"
    )
    is_primary_key_candidate: bool = Field(
        default=False, description="True if column has no nulls and all values unique"
    )


class CorrelationPair(BaseModel):
    """Pearson correlation between two numeric columns."""

    col_a: str
    col_b: str
    correlation: float = Field(description="Pearson r [-1, 1]")


class QualityIssue(BaseModel):
    """A detected data quality problem."""

    issue_type: IssueType
    severity: IssueSeverity
    column: Optional[str] = Field(
        None, description="Affected column (None=dataset-level)"
    )
    description: str = Field(description="Human-readable issue description")
    metric_value: Optional[float] = Field(
        None, description="The measured metric that triggered the issue"
    )
    threshold: Optional[float] = Field(None, description="Threshold that was breached")
    recommendation: str = Field(description="Suggested remediation action")


class DataProfile(BaseModel):
    """Full dataset profile produced by DataProfiler."""

    dataset_name: str = Field(default="dataset")
    profiled_at: datetime = Field(default_factory=datetime.utcnow)
    row_count: int
    column_count: int
    memory_mb: float = Field(default=0.0)
    duplicate_row_count: int = Field(default=0)
    duplicate_row_rate: float = Field(default=0.0)
    columns: Dict[str, ColumnProfile] = Field(default_factory=dict)
    correlations: List[CorrelationPair] = Field(default_factory=list)
    quality_issues: List[QualityIssue] = Field(default_factory=list)
    overall_quality_score: float = Field(
        default=100.0,
        description="Composite quality score [0, 100] — higher is better",
    )

    def get_issues_by_severity(self, severity: IssueSeverity) -> List[QualityIssue]:
        """Return issues filtered by severity level."""
        return [i for i in self.quality_issues if i.severity == severity]

    def critical_issue_count(self) -> int:
        """Return number of critical severity issues."""
        return len(self.get_issues_by_severity(IssueSeverity.CRITICAL))

    def high_null_columns(self, threshold: float = 0.20) -> List[str]:
        """Return column names with null rate above threshold."""
        return [name for name, col in self.columns.items() if col.null_rate > threshold]

    def numeric_columns(self) -> List[str]:
        """Return names of numeric columns."""
        return [
            name
            for name, col in self.columns.items()
            if col.column_type == ColumnType.NUMERIC
        ]

    def categorical_columns(self) -> List[str]:
        """Return names of categorical columns."""
        return [
            name
            for name, col in self.columns.items()
            if col.column_type == ColumnType.CATEGORICAL
        ]
