# autonomous-dq-agent

[![CI](https://github.com/naveenkanaparthi-git/autonomous-dq-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/naveenkanaparthi-git/autonomous-dq-agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Autonomous Data Quality Agent** — a production-grade, LLM-powered data quality platform that profiles datasets, auto-generates expectation suites, validates data contracts, and emits rich HTML/JSON reports. Powered by Anthropic Claude for intelligent expectation generation with a full heuristic fallback for air-gapped environments.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [CLI Usage](#cli-usage)
- [Python API](#python-api)
- [Configuration](#configuration)
- [Expectation Types](#expectation-types)
- [Quality Score](#quality-score)
- [Reports](#reports)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Features

| Feature | Detail |
|---------|--------|
| **Statistical Profiling** | Null rates, percentiles (p25/p50/p75/p95/p99), skewness, kurtosis, IQR-fence outlier detection |
| **15 Expectation Types** | NOT_NULL, BETWEEN, IN_SET, MATCH_REGEX, MEAN_BETWEEN, UNIQUE, ROW_COUNT, and more |
| **AI Expectation Generation** | Claude API analyzes profiles and suggests context-aware expectations |
| **Heuristic Fallback** | Full rule-based suggestion engine — no API key required |
| **Rich HTML Reports** | Self-contained, shareable quality reports with color-coded issue tables |
| **JSON Reports** | Machine-readable profiles for CI/CD data quality gates |
| **Fluent Suite Builder** | Programmatic expectation builder with method chaining |
| **Quality Score** | Composite [0-100] score with severity-weighted penalties |
| **SQL Fix Generator** | Auto-generate UPDATE/DELETE SQL to remediate detected issues |
| **CLI + Python API** | Typer CLI with 5 commands + importable Python classes |
| **Type-safe** | Full Pydantic v2 models throughout, mypy-clean |
| **Multi-format Input** | CSV, JSON, JSONL, Parquet via pandas |

---

## Architecture

```
+-------------------------------------------------------------+
|                    CLI / Python API                          |
|         profile  |  suggest  |  validate  |  run-all        |
+------------+----------+-----------+-----------+-------------+
             |          |           |           |
             v          v           v           v
      +----------+  +----------+  +----------+  +----------+
      |DataPro-  |  |ClaudeAI  |  |DataVali- |  |Quality   |
      |filer     |  |Agent     |  |dator     |  |Reporter  |
      +----------+  +----------+  +----------+  +----------+
           |              |             |              |
           v              v             v              v
      DataProfile   ExpectationSuite  ValidationResult  HTML/JSON
      (Pydantic)    (Pydantic)        (Pydantic)        Reports
```

### Data Flow

1. **Profile** — `DataProfiler` scans a pandas DataFrame, computing per-column statistics (numeric: mean/std/IQR/outliers, categorical: top-k/cardinality, plus correlation matrix and issue detection).
2. **Suggest** — `ClaudeAIAgent` sends the condensed profile JSON to Claude (`claude-sonnet-4-6`) which returns a JSON array of expectations. Falls back to deterministic heuristics when offline.
3. **Validate** — `DataValidator` evaluates each expectation against the live DataFrame using 15 specialized handlers.
4. **Report** — `QualityReporter` renders plain-text, JSON, and self-contained HTML outputs.

---

## Quickstart

```bash
# Install
pip install -r requirements.txt

# Profile a CSV file
dq-agent profile data/sample/sample_data.csv

# Generate an expectation suite (heuristic mode -- no API key needed)
dq-agent suggest data/sample/sample_data.csv --output suite.json

# Validate the data against the generated suite
dq-agent validate data/sample/sample_data.csv suite.json

# Full pipeline: profile -> suggest -> validate -> save reports
dq-agent run-all data/sample/sample_data.csv --output-dir reports/
```

With Claude AI:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
dq-agent suggest data/sample/sample_data.csv --output ai_suite.json
```

---

## Installation

### From source

```bash
git clone https://github.com/naveenkanaparthi-git/autonomous-dq-agent.git
cd autonomous-dq-agent
pip install -e ".[dev]"
```

### Dependencies only

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.11+
- pandas 2.0+
- numpy 1.26+
- scipy 1.11+
- pydantic 2.0+
- pydantic-settings 2.0+
- typer 0.12+
- rich 13.0+
- anthropic 0.25+ (optional -- for Claude AI mode)

---

## CLI Usage

### `dq-agent profile`

Profile a data file and display statistics with quality issues.

```bash
dq-agent profile <input_file> [OPTIONS]

Options:
  --name, -n TEXT       Dataset logical name
  --html PATH           Save HTML report to this path
  --json PATH           Save JSON report to this path
  --issues/--no-issues  Show/hide quality issues table (default: show)
```

Example output:

```
Dataset: sample_data
Rows: 20  Columns: 7  Memory: 0.01 MB
Duplicates: 0 (0.00%)
Quality Score: 95.0/100

Column Profiles
+----------+-------------+--------+----------+----------------------------+
| Column   | Type        | Null % | Distinct | Stats                      |
+----------+-------------+--------+----------+----------------------------+
| id       | numeric     | 0.0%   | 20       | mean=10.50 std=5.92        |
| age      | numeric     | 0.0%   | 20       | mean=38.10 std=9.21        |
| income   | numeric     | 0.0%   | 20       | mean=83450.00 std=27018.40 |
| status   | categorical | 0.0%   | 3        | top='active'               |
| region   | categorical | 0.0%   | 4        | top='north'                |
| score    | numeric     | 0.0%   | 20       | mean=0.79 std=0.11         |
+----------+-------------+--------+----------+----------------------------+
```

### `dq-agent suggest`

Generate an expectation suite from a data file.

```bash
dq-agent suggest <input_file> [OPTIONS]

Options:
  --name, -n TEXT    Dataset logical name
  --output, -o PATH  Save suite JSON to this path
```

### `dq-agent validate`

Validate a data file against an existing expectation suite.

```bash
dq-agent validate <input_file> <suite_file> [OPTIONS]

Options:
  --name, -n TEXT  Dataset logical name
  --fail/--no-fail Exit 1 if validation fails (default: no-fail)
```

### `dq-agent run-all`

Full pipeline: profile -> suggest -> validate -> save reports.

```bash
dq-agent run-all <input_file> [OPTIONS]

Options:
  --name, -n TEXT        Dataset logical name
  --output-dir, -o PATH  Reports output directory (default: reports/)
  --fail/--no-fail       Exit 1 if validation fails
```

### `dq-agent version`

Print the installed version.

```bash
dq-agent version
# dq-agent v0.1.0
```

---

## Python API

### DataProfiler

```python
import pandas as pd
from autonomous_dq_agent.core.profiler import DataProfiler

df = pd.read_csv("data/sample/sample_data.csv")
profiler = DataProfiler()
profile = profiler.profile(df, dataset_name="customers")

print(f"Quality Score: {profile.overall_quality_score}")
print(f"Issues: {len(profile.quality_issues)}")
print(f"Critical: {profile.critical_issue_count()}")
print(f"High-null columns: {profile.high_null_columns(threshold=0.05)}")
```

### DataValidator

```python
from autonomous_dq_agent.core.validator import DataValidator
from autonomous_dq_agent.services.suite_builder import SuiteBuilder

suite = (
    SuiteBuilder("production_suite", "customers")
    .not_null("customer_id")
    .unique("customer_id")
    .between("age", 0, 120)
    .between("ltv", 0, 1_000_000)
    .in_set("status", ["active", "churned", "trial"])
    .match_regex("email", r".+@.+\..+")
    .row_count_between(1_000, 10_000_000)
    .build()
)

validator = DataValidator()
result = validator.validate(df, suite)

print(f"Pass: {result.success}")
print(f"Rate: {result.success_percent:.1f}%")
for failure in result.failed_results():
    print(f"  FAIL: {failure.expectation.description}")
```

### ClaudeAIAgent

```python
from autonomous_dq_agent.services.ai_agent import ClaudeAIAgent
import os

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

agent = ClaudeAIAgent()
suite = agent.suggest_expectations(profile)

# Analyze issues
recommendations = agent.analyze_quality_issues(profile.quality_issues)
for rec in recommendations:
    print(rec)

# Generate SQL fixes
sql = agent.generate_fix_sql(profile)
print(sql)
```

### QualityReporter

```python
from autonomous_dq_agent.core.reporter import QualityReporter

reporter = QualityReporter(output_dir="reports")

# Plain text
print(reporter.generate_text_report(profile))

# Save HTML
html_path = reporter.save_html_report(profile)

# Save JSON
json_path = reporter.save_json_report(profile)

# Validation report
print(reporter.generate_validation_text_report(result))
```

---

## Configuration

All settings can be set via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | `None` | Anthropic API key (optional) |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model ID |
| `CLAUDE_MAX_TOKENS` | `4096` | Max tokens per Claude response |
| `NULL_RATE_CRITICAL_THRESHOLD` | `0.50` | Null rate threshold for CRITICAL severity |
| `NULL_RATE_HIGH_THRESHOLD` | `0.20` | Null rate threshold for HIGH severity |
| `NULL_RATE_MEDIUM_THRESHOLD` | `0.05` | Null rate threshold for MEDIUM severity |
| `OUTLIER_IQR_MULTIPLIER` | `1.5` | IQR fence multiplier for outlier detection |
| `HIGH_CARDINALITY_THRESHOLD` | `0.95` | Cardinality ratio for HIGH_CARDINALITY flag |
| `SKEWNESS_THRESHOLD` | `2.0` | Absolute skewness for DISTRIBUTION_SKEW flag |
| `DUPLICATE_RATE_THRESHOLD` | `0.01` | Duplicate row rate for DUPLICATE issue |
| `CORRELATION_THRESHOLD` | `0.95` | Pearson r threshold for correlation flag |
| `REPORT_OUTPUT_DIR` | `reports` | Default report output directory |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Expectation Types

| Expectation | Column | Key Parameters |
|-------------|--------|----------------|
| `expect_column_values_to_not_be_null` | yes | -- |
| `expect_column_null_rate_to_be_below` | yes | `max_null_rate` |
| `expect_column_values_to_be_between` | yes | `min_value`, `max_value` |
| `expect_column_values_to_be_in_set` | yes | `value_set` |
| `expect_column_values_to_not_be_in_set` | yes | `forbidden_set` |
| `expect_column_values_to_match_regex` | yes | `regex` |
| `expect_column_mean_to_be_between` | yes | `min_value`, `max_value` |
| `expect_column_min_to_be_gte` | yes | `min_value` |
| `expect_column_max_to_be_lte` | yes | `max_value` |
| `expect_column_values_to_be_unique` | yes | -- |
| `expect_table_row_count_to_be_between` | no | `min_value`, `max_value` |
| `expect_column_to_exist` | yes | -- |
| `expect_column_distinct_count_to_be_gte` | yes | `min_count` |
| `expect_column_std_to_be_between` | yes | `min_value`, `max_value` |
| `expect_column_quantile_values_to_be_between` | yes | `quantile`, `min_value`, `max_value` |

---

## Quality Score

The overall quality score starts at **100** and deducts points per detected issue:

| Severity | Penalty |
|----------|---------|
| CRITICAL | -20 pts |
| HIGH | -10 pts |
| MEDIUM | -5 pts |
| LOW | -2 pts |
| INFO | -0.5 pts |

Score is clamped to [0, 100]. A score >= 80 is considered healthy.

### Issue Types

| Issue Type | Description |
|------------|-------------|
| `null_rate` | Column has excessive null values |
| `outlier` | Column has IQR-fence outliers |
| `cardinality` | Columns are highly correlated |
| `distribution_skew` | Distribution is highly skewed |
| `duplicate` | Dataset has duplicate rows |
| `constant_column` | Column has only one distinct value |
| `high_cardinality` | Categorical column has too many unique values |

---

## Reports

### HTML Report

Self-contained HTML file with:
- Quality score badge (color-coded: green >= 80, orange >= 50, red < 50)
- Dataset overview cards (rows, columns, duplicates, issues, critical count)
- Color-coded issue table with recommendations
- Full column profile table with per-column stats

### JSON Report

Full `DataProfile` serialized as JSON -- suitable for:
- CI/CD quality gates
- Dashboard ingestion
- Trend comparison across pipeline runs

---

## Project Structure

```
autonomous-dq-agent/
+-- .github/workflows/ci.yml       # GitHub Actions CI
+-- src/autonomous_dq_agent/
|   +-- __init__.py
|   +-- config.py                  # Settings (pydantic-settings)
|   +-- cli.py                     # Typer CLI (5 commands)
|   +-- core/
|   |   +-- profiler.py            # DataProfiler
|   |   +-- validator.py           # DataValidator (15 expectation types)
|   |   +-- reporter.py            # QualityReporter (text/JSON/HTML)
|   +-- models/
|   |   +-- profile.py             # DataProfile, ColumnProfile, QualityIssue
|   |   +-- validation.py          # ExpectationSuite, ValidationResult
|   +-- services/
|       +-- ai_agent.py            # ClaudeAIAgent (AI + heuristic modes)
|       +-- suite_builder.py       # SuiteBuilder (fluent API)
+-- tests/
|   +-- conftest.py                # Shared fixtures
|   +-- unit/test_core.py          # 78+ unit tests
|   +-- integration/test_integration.py  # 15 integration tests
+-- data/sample/sample_data.csv    # Sample dataset
+-- docs/architecture.md           # Mermaid C4 diagrams
+-- pyproject.toml
+-- requirements.txt
+-- Makefile
```

---

## Development

```bash
# Install dev dependencies
make install

# Auto-format code
make format

# Lint
make lint

# Type check
make typecheck

# Run tests with coverage
make test

# Full pre-push check
make all
```

---

## Testing

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

The test suite includes:

- **Unit tests** -- 78+ tests covering column type detection, numeric/categorical stats, all 15 expectation handlers, SuiteBuilder fluent API, and all reporter output formats
- **Integration tests** -- 15 tests covering the full pipeline (profile -> suggest -> validate -> report) with realistic e-commerce data, edge cases (empty DataFrame, all-null columns, single rows), and JSON roundtrip serialization

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

## License

MIT License -- see [LICENSE](LICENSE).

---

## Author

Built by [NEXUS Portfolio Agent](https://github.com/naveenkanaparthi-git) -- an autonomous AI/ML portfolio generation system.

Contact: naveenkanaparthi9@gmail.com
