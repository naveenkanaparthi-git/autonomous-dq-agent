# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2026-04-26

### Added

- `DataProfiler` with per-column type detection (NUMERIC, CATEGORICAL, TEXT, DATETIME, BOOLEAN)
- Numeric statistics: mean, median, std, percentiles (p25/p50/p75/p95/p99), skewness, kurtosis, IQR-fence outlier detection
- Categorical statistics: unique count, top-k values, cardinality ratio, average string length
- Dataset-level checks: duplicate row detection, Pearson correlation matrix
- Quality issue detection: NULL_RATE, OUTLIER, CONSTANT_COLUMN, DISTRIBUTION_SKEW, HIGH_CARDINALITY, DUPLICATE
- Composite quality score [0-100] with severity-weighted penalties
- 15-type `DataValidator` expectation engine
- `ExpectationSuite` and `ValidationResult` Pydantic v2 models
- `SuiteBuilder` fluent API for programmatic suite construction
- `ClaudeAIAgent` with Claude API integration and full heuristic fallback
- `QualityReporter` with plain-text, JSON, and self-contained HTML output
- Typer CLI: `profile`, `suggest`, `validate`, `run-all`, `version` commands
- 78+ unit tests and 15 integration tests
- GitHub Actions CI with black/ruff/mypy/pytest
