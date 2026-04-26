"""Integration tests — full pipeline from DataFrame to reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autonomous_dq_agent.core.profiler import DataProfiler
from autonomous_dq_agent.core.reporter import QualityReporter
from autonomous_dq_agent.core.validator import DataValidator
from autonomous_dq_agent.services.ai_agent import ClaudeAIAgent
from autonomous_dq_agent.services.suite_builder import SuiteBuilder


@pytest.fixture
def ecommerce_df() -> pd.DataFrame:
    """Realistic e-commerce orders DataFrame."""
    rng = np.random.default_rng(99)
    n = 1000
    order_ids = list(range(10001, 10001 + n))
    amounts = rng.uniform(5.0, 500.0, n)
    amounts[rng.choice(n, 5, replace=False)] = np.nan
    statuses = rng.choice(
        ["pending", "shipped", "delivered", "cancelled", "refunded"], n
    )
    regions = rng.choice(["us-east", "us-west", "eu-central", "ap-south"], n)
    quantities = rng.integers(1, 20, n)
    discount_pct = rng.uniform(0.0, 0.5, n)
    return pd.DataFrame(
        {
            "order_id": order_ids,
            "amount": amounts,
            "status": statuses,
            "region": regions,
            "quantity": quantities,
            "discount_pct": discount_pct,
        }
    )


class TestFullPipeline:
    """Integration tests for the full profile -> suggest -> validate pipeline."""

    def test_profile_ecommerce(self, ecommerce_df: pd.DataFrame) -> None:
        """Profiling e-commerce data produces expected results."""
        profiler = DataProfiler()
        profile = profiler.profile(ecommerce_df, dataset_name="orders")

        assert profile.dataset_name == "orders"
        assert profile.row_count == 1000
        assert profile.column_count == 6
        assert "order_id" in profile.columns
        assert "amount" in profile.columns
        assert profile.overall_quality_score > 0

    def test_heuristic_suite_generation(self, ecommerce_df: pd.DataFrame) -> None:
        """Heuristic agent generates a non-empty expectation suite."""
        profiler = DataProfiler()
        profile = profiler.profile(ecommerce_df, dataset_name="orders")

        agent = ClaudeAIAgent()
        suite = agent.suggest_expectations(profile)
        assert len(suite.expectations) > 0
        assert suite.dataset_name == "orders"

    def test_heuristic_suite_covers_all_columns(
        self, ecommerce_df: pd.DataFrame
    ) -> None:
        """Generated suite covers every column in the DataFrame."""
        profiler = DataProfiler()
        profile = profiler.profile(ecommerce_df, dataset_name="orders")
        agent = ClaudeAIAgent()
        suite = agent.suggest_expectations(profile)

        covered = {e.column for e in suite.expectations if e.column}
        for col in ecommerce_df.columns:
            assert col in covered, f"Column {col} not covered by any expectation"

    def test_validate_clean_data_high_pass_rate(
        self, ecommerce_df: pd.DataFrame
    ) -> None:
        """Manual suite built from clean data statistics should mostly pass."""
        builder = (
            SuiteBuilder("ecommerce_suite", "orders")
            .row_count_between(500, 5000)
            .column_exists("order_id")
            .column_exists("amount")
            .column_exists("status")
            .unique("order_id")
            .in_set(
                "status", ["pending", "shipped", "delivered", "cancelled", "refunded"]
            )
            .between("quantity", 1, 100)
            .between("discount_pct", 0.0, 1.0)
        )
        suite = builder.build()

        validator = DataValidator()
        result = validator.validate(ecommerce_df, suite)

        assert result.success_percent >= 80.0, (
            f"Expected >= 80% pass rate, got {result.success_percent:.1f}%\n"
            f"Failed: {[r.expectation.description for r in result.failed_results()]}"
        )

    def test_reports_saved_to_disk(
        self, ecommerce_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """HTML and JSON reports are written to the filesystem."""
        profiler = DataProfiler()
        profile = profiler.profile(ecommerce_df, dataset_name="orders")
        reporter = QualityReporter(output_dir=str(tmp_path))

        html_path = reporter.save_html_report(profile, filename="orders.html")
        json_path = reporter.save_json_report(profile, filename="orders.json")

        assert html_path.exists()
        assert json_path.exists()

        html = html_path.read_text()
        assert "<!DOCTYPE html>" in html
        assert "orders" in html

        data = json.loads(json_path.read_text())
        assert data["row_count"] == 1000

    def test_suite_roundtrip_json(
        self, ecommerce_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """ExpectationSuite serializes and deserializes without data loss."""
        from autonomous_dq_agent.models.validation import ExpectationSuite

        suite = (
            SuiteBuilder("roundtrip_suite", "orders")
            .not_null("order_id")
            .between("amount", 0, 1000)
            .in_set("status", ["pending", "shipped", "delivered"])
            .build()
        )
        suite_path = tmp_path / "suite.json"
        suite_path.write_text(suite.model_dump_json(indent=2))

        loaded = ExpectationSuite.model_validate(json.loads(suite_path.read_text()))
        assert loaded.suite_name == suite.suite_name
        assert len(loaded.expectations) == len(suite.expectations)
        assert (
            loaded.expectations[0].expectation_type
            == suite.expectations[0].expectation_type
        )

    def test_analyze_issues_heuristic(self, ecommerce_df: pd.DataFrame) -> None:
        """ClaudeAIAgent.analyze_quality_issues returns strings in heuristic mode."""
        profiler = DataProfiler()
        profile = profiler.profile(ecommerce_df, dataset_name="orders")
        agent = ClaudeAIAgent()
        recs = agent.analyze_quality_issues(profile.quality_issues)
        assert isinstance(recs, list)
        assert all(isinstance(r, str) for r in recs)

    def test_generate_fix_sql_heuristic(self, ecommerce_df: pd.DataFrame) -> None:
        """ClaudeAIAgent.generate_fix_sql returns SQL string in heuristic mode."""
        profiler = DataProfiler()
        profile = profiler.profile(ecommerce_df, dataset_name="orders")
        agent = ClaudeAIAgent()
        sql = agent.generate_fix_sql(profile)
        assert isinstance(sql, str)
        assert "orders" in sql

    def test_no_issues_analyze(self) -> None:
        """analyze_quality_issues with no issues returns positive message."""
        agent = ClaudeAIAgent()
        recs = agent.analyze_quality_issues([])
        assert len(recs) >= 1
        assert "clean" in recs[0].lower() or "no" in recs[0].lower()

    def test_profile_with_all_null_column(self) -> None:
        """Profiling a DataFrame with an all-null column does not raise."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "empty": [None, None, None],
                "value": [1.0, 2.0, 3.0],
            }
        )
        profiler = DataProfiler()
        profile = profiler.profile(df, dataset_name="test_nulls")
        assert profile.row_count == 3
        assert "empty" in profile.columns

    def test_profile_single_row(self) -> None:
        """Profiler handles a single-row DataFrame."""
        df = pd.DataFrame({"x": [42.0], "y": ["hello"]})
        profiler = DataProfiler()
        profile = profiler.profile(df, dataset_name="tiny")
        assert profile.row_count == 1

    def test_validate_empty_suite(self, ecommerce_df: pd.DataFrame) -> None:
        """Validating against empty suite returns success with zero expectations."""
        from autonomous_dq_agent.models.validation import ExpectationSuite

        suite = ExpectationSuite(suite_name="empty", dataset_name="orders")
        validator = DataValidator()
        result = validator.validate(ecommerce_df, suite)
        assert result.success
        assert result.evaluated_expectations == 0

    def test_profiler_memory_mb(self, ecommerce_df: pd.DataFrame) -> None:
        """Memory usage reported is positive."""
        profiler = DataProfiler()
        profile = profiler.profile(ecommerce_df, dataset_name="orders")
        assert profile.memory_mb > 0.0

    def test_quality_score_degrades_with_issues(self) -> None:
        """Adding more quality issues lowers the score."""
        rng = np.random.default_rng(0)
        n = 500

        clean = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.choice(["a", "b"], n)})
        bad = clean.copy()
        bad.loc[rng.choice(n, 200, replace=False), "x"] = np.nan

        profiler = DataProfiler()
        clean_score = profiler.profile(clean).overall_quality_score
        bad_score = profiler.profile(bad).overall_quality_score
        assert bad_score < clean_score
