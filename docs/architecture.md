# Architecture — Autonomous DQ Agent

## System Overview

```mermaid
C4Context
    title Autonomous DQ Agent — System Context

    Person(user, "Data Engineer", "Runs DQ checks on datasets")
    System(dqa, "Autonomous DQ Agent", "Profiles data, generates expectations, validates quality")
    System_Ext(claude, "Anthropic Claude API", "LLM for AI-powered suggestions")
    System_Ext(storage, "File System / Data Lake", "CSV, Parquet, JSON datasets")

    Rel(user, dqa, "Uses CLI or Python API")
    Rel(dqa, claude, "Sends profile summaries, receives expectation JSON")
    Rel(dqa, storage, "Reads datasets, writes HTML/JSON reports")
```

## Component Diagram

```mermaid
C4Component
    title Autonomous DQ Agent — Component View

    Container(cli, "CLI (Typer)", "Python", "profile / suggest / validate / run-all commands")

    Component(profiler, "DataProfiler", "Python", "Computes per-column stats, detects issues, scores quality")
    Component(validator, "DataValidator", "Python", "Evaluates ExpectationSuites against DataFrames")
    Component(reporter, "QualityReporter", "Python", "Generates text, JSON, HTML reports")
    Component(agent, "ClaudeAIAgent", "Python + Anthropic SDK", "Calls Claude API or falls back to heuristics")
    Component(builder, "SuiteBuilder", "Python", "Fluent builder for ExpectationSuites")
    Component(models, "Pydantic Models", "Python", "DataProfile, ValidationResult, ExpectationSuite")

    Rel(cli, profiler, "profile(df)")
    Rel(cli, agent, "suggest_expectations(profile)")
    Rel(cli, validator, "validate(df, suite)")
    Rel(cli, reporter, "generate reports")
    Rel(profiler, models, "returns DataProfile")
    Rel(validator, models, "returns ValidationResult")
    Rel(agent, builder, "uses SuiteBuilder internally")
```

## Data Flow

```mermaid
flowchart LR
    A[Raw Data File] --> B[DataProfiler]
    B --> C[DataProfile]
    C --> D{AI Available?}
    D -- Yes --> E[Claude API]
    D -- No --> F[Heuristic Rules]
    E --> G[ExpectationSuite]
    F --> G
    G --> H[DataValidator]
    C --> H
    H --> I[ValidationResult]
    C --> J[QualityReporter]
    I --> J
    J --> K[HTML Report]
    J --> L[JSON Report]
```

## Quality Score Formula

The overall quality score starts at 100 and deducts penalties per issue:

| Severity | Penalty |
|----------|---------|
| CRITICAL | −20 pts |
| HIGH     | −10 pts |
| MEDIUM   | −5 pts  |
| LOW      | −2 pts  |
| INFO     | −0.5 pts |

Score is clamped to [0, 100].

## Expectation Types

15 expectation types are supported:

| Type | Description |
|------|-------------|
| `expect_column_values_to_not_be_null` | No nulls in column |
| `expect_column_null_rate_to_be_below` | Null rate ≤ threshold |
| `expect_column_values_to_be_between` | Values in [min, max] |
| `expect_column_values_to_be_in_set` | Values in allowed set |
| `expect_column_values_to_not_be_in_set` | Values not in forbidden set |
| `expect_column_values_to_match_regex` | Values match regex |
| `expect_column_mean_to_be_between` | Mean in [min, max] |
| `expect_column_min_to_be_gte` | Min ≥ threshold |
| `expect_column_max_to_be_lte` | Max ≤ threshold |
| `expect_column_values_to_be_unique` | All values unique |
| `expect_table_row_count_to_be_between` | Row count in [min, max] |
| `expect_column_to_exist` | Column present in schema |
| `expect_column_distinct_count_to_be_gte` | Distinct count ≥ min |
| `expect_column_std_to_be_between` | Std dev in [min, max] |
| `expect_column_quantile_values_to_be_between` | Quantile in [min, max] |
