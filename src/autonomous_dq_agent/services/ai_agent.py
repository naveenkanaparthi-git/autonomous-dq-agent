"""ClaudeAIAgent — uses Anthropic Claude to analyze profiles and suggest expectations."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from autonomous_dq_agent.config import Settings, get_settings
from autonomous_dq_agent.models.profile import DataProfile, QualityIssue
from autonomous_dq_agent.models.validation import (
    Expectation,
    ExpectationSuite,
    ExpectationType,
)

logger = logging.getLogger(__name__)


class ClaudeAIAgent:
    """Wraps the Anthropic Claude API to provide AI-powered DQ suggestions.

    When no API key is configured, the agent falls back to a rule-based
    heuristic mode so the rest of the pipeline remains fully functional
    without network access.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize agent, optionally with custom settings."""
        self.settings = settings or get_settings()
        self._client: Optional[Any] = None
        self._ai_available = False
        self._init_client()

    def _init_client(self) -> None:
        """Attempt to initialise the Anthropic client."""
        if not self.settings.anthropic_api_key:
            logger.info("No ANTHROPIC_API_KEY set — running in heuristic mode.")
            return
        try:
            import anthropic  # type: ignore[import]

            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            self._ai_available = True
            logger.info(
                "Anthropic client initialized (model=%s)", self.settings.claude_model
            )
        except ImportError:
            logger.warning(
                "anthropic package not installed; running in heuristic mode."
            )

    @property
    def is_ai_enabled(self) -> bool:
        """True when the Claude API is available."""
        return self._ai_available

    def _call_claude(self, prompt: str) -> str:
        """Send a prompt to Claude and return the text response.

        Args:
            prompt: Full user prompt text.

        Returns:
            Raw text content from Claude.

        Raises:
            RuntimeError: If AI is not available.
        """
        if not self._ai_available or self._client is None:
            raise RuntimeError("Claude AI not available; check ANTHROPIC_API_KEY.")

        message = self._client.messages.create(
            model=self.settings.claude_model,
            max_tokens=self.settings.claude_max_tokens,
            system=(
                "You are an expert Data Quality engineer. "
                "When asked to suggest data quality expectations, respond ONLY with "
                "a valid JSON array of expectation objects. Each object must have keys: "
                "'expectation_type', 'column' (or null for table-level), and 'kwargs' dict."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def suggest_expectations(self, profile: DataProfile) -> ExpectationSuite:
        """Generate an ExpectationSuite from a DataProfile.

        Tries Claude AI first; falls back to heuristic rule-based suggestions.

        Args:
            profile: The DataProfile to generate expectations for.

        Returns:
            ExpectationSuite with auto-generated expectations.
        """
        suite = ExpectationSuite(
            suite_name=f"{profile.dataset_name}_auto_suite",
            dataset_name=profile.dataset_name,
        )

        if self._ai_available:
            try:
                ai_expectations = self._suggest_via_claude(profile)
                for exp in ai_expectations:
                    suite.add_expectation(exp)
                logger.info(
                    "Claude AI generated %d expectations", len(suite.expectations)
                )
                return suite
            except Exception as exc:
                logger.warning(
                    "Claude AI suggestion failed (%s); falling back to heuristics.", exc
                )

        heuristic_expectations = self._suggest_heuristic(profile)
        for exp in heuristic_expectations:
            suite.add_expectation(exp)
        logger.info("Heuristic mode generated %d expectations", len(suite.expectations))
        return suite

    def _suggest_via_claude(self, profile: DataProfile) -> List[Expectation]:
        """Ask Claude to generate expectations from a JSON-summarised profile."""
        profile_summary = self._build_profile_summary(profile)
        prompt = (
            "Analyze this dataset profile and suggest data quality expectations.\n\n"
            f"Profile:\n{json.dumps(profile_summary, indent=2)}\n\n"
            "Return a JSON array of expectation objects. Each must have:\n"
            '- "expectation_type": one of the supported types\n'
            '- "column": column name string or null for table-level\n'
            '- "kwargs": dict of parameters for the expectation\n\n'
            "Focus on the most important quality checks based on column types and statistics."
        )

        raw = self._call_claude(prompt)

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        expectations: List[Expectation] = []
        for item in data:
            try:
                exp = Expectation(
                    expectation_type=ExpectationType(item["expectation_type"]),
                    column=item.get("column"),
                    kwargs=item.get("kwargs", {}),
                    meta={"source": "claude_ai"},
                )
                expectations.append(exp)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping malformed AI expectation: %s — %s", item, e)
        return expectations

    def _build_profile_summary(self, profile: DataProfile) -> Dict[str, Any]:
        """Condense a DataProfile to a compact dict suitable for Claude prompt."""
        col_summaries = {}
        for col_name, col in profile.columns.items():
            summary: Dict[str, Any] = {
                "type": col.column_type.value,
                "null_rate": round(col.null_rate, 3),
                "distinct_count": col.distinct_count,
                "row_count": col.row_count,
            }
            if col.numeric_stats:
                ns = col.numeric_stats
                summary["mean"] = ns.mean
                summary["std"] = ns.std
                summary["min"] = ns.min
                summary["max"] = ns.max
                summary["outlier_count"] = ns.outlier_count
            if col.categorical_stats:
                cs = col.categorical_stats
                summary["unique_count"] = cs.unique_count
                summary["top_values"] = list(cs.top_values.keys())[:5]
            col_summaries[col_name] = summary

        return {
            "dataset_name": profile.dataset_name,
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "duplicate_row_rate": profile.duplicate_row_rate,
            "quality_score": profile.overall_quality_score,
            "columns": col_summaries,
            "issue_count": len(profile.quality_issues),
        }

    def _suggest_heuristic(self, profile: DataProfile) -> List[Expectation]:
        """Generate rule-based expectations from column profiles."""
        expectations: List[Expectation] = []

        expectations.append(
            Expectation(
                expectation_type=ExpectationType.ROW_COUNT_BETWEEN,
                column=None,
                kwargs={
                    "min_value": max(1, int(profile.row_count * 0.5)),
                    "max_value": int(profile.row_count * 2.0),
                },
                meta={"source": "heuristic", "note": "row count sanity check"},
            )
        )

        for col_name, col in profile.columns.items():
            expectations.append(
                Expectation(
                    expectation_type=ExpectationType.COLUMN_EXISTS,
                    column=col_name,
                    kwargs={},
                    meta={"source": "heuristic"},
                )
            )

            if col.null_rate == 0.0:
                expectations.append(
                    Expectation(
                        expectation_type=ExpectationType.NOT_NULL,
                        column=col_name,
                        kwargs={},
                        meta={"source": "heuristic"},
                    )
                )
            elif col.null_rate < 0.05:
                expectations.append(
                    Expectation(
                        expectation_type=ExpectationType.NULL_RATE_BELOW,
                        column=col_name,
                        kwargs={"max_null_rate": round(col.null_rate * 3, 3)},
                        meta={"source": "heuristic"},
                    )
                )

            if col.numeric_stats is not None:
                ns = col.numeric_stats
                buffer_lo = abs(ns.min) * 0.1 if ns.min != 0 else 1.0
                buffer_hi = abs(ns.max) * 0.1 if ns.max != 0 else 1.0
                expectations.append(
                    Expectation(
                        expectation_type=ExpectationType.BETWEEN,
                        column=col_name,
                        kwargs={
                            "min_value": ns.min - buffer_lo,
                            "max_value": ns.max + buffer_hi,
                        },
                        meta={"source": "heuristic"},
                    )
                )
                expectations.append(
                    Expectation(
                        expectation_type=ExpectationType.MEAN_BETWEEN,
                        column=col_name,
                        kwargs={
                            "min_value": round(ns.mean - 2 * ns.std, 4),
                            "max_value": round(ns.mean + 2 * ns.std, 4),
                        },
                        meta={"source": "heuristic"},
                    )
                )
                if ns.negative_count == 0 and ns.min >= 0:
                    expectations.append(
                        Expectation(
                            expectation_type=ExpectationType.MIN_GTE,
                            column=col_name,
                            kwargs={"min_value": 0.0},
                            meta={
                                "source": "heuristic",
                                "note": "non-negative constraint",
                            },
                        )
                    )

            if col.categorical_stats is not None:
                cs = col.categorical_stats
                if cs.unique_count <= 20 and cs.unique_count > 1:
                    value_set = list(cs.top_values.keys())
                    expectations.append(
                        Expectation(
                            expectation_type=ExpectationType.IN_SET,
                            column=col_name,
                            kwargs={"value_set": value_set},
                            meta={"source": "heuristic"},
                        )
                    )
                if cs.unique_count > 0:
                    expectations.append(
                        Expectation(
                            expectation_type=ExpectationType.DISTINCT_COUNT_GTE,
                            column=col_name,
                            kwargs={"min_count": 1},
                            meta={"source": "heuristic"},
                        )
                    )

            if col.is_primary_key_candidate:
                expectations.append(
                    Expectation(
                        expectation_type=ExpectationType.UNIQUE,
                        column=col_name,
                        kwargs={},
                        meta={"source": "heuristic", "note": "PK candidate"},
                    )
                )

        return expectations

    def analyze_quality_issues(self, issues: List[QualityIssue]) -> List[str]:
        """Return natural-language analysis of quality issues via Claude or heuristics.

        Args:
            issues: List of detected QualityIssue objects.

        Returns:
            List of actionable recommendation strings.
        """
        if not issues:
            return ["No data quality issues detected. Dataset looks clean."]

        if self._ai_available:
            try:
                return self._analyze_issues_via_claude(issues)
            except Exception as exc:
                logger.warning("Claude issue analysis failed: %s", exc)

        return self._analyze_issues_heuristic(issues)

    def _analyze_issues_via_claude(self, issues: List[QualityIssue]) -> List[str]:
        """Use Claude to produce natural-language recommendations for detected issues."""
        issues_summary = [
            {
                "type": i.issue_type.value,
                "severity": i.severity.value,
                "column": i.column,
                "description": i.description,
            }
            for i in issues[:20]
        ]
        prompt = (
            "These data quality issues were detected in a dataset:\n\n"
            f"{json.dumps(issues_summary, indent=2)}\n\n"
            "Provide a concise, prioritized list of actionable recommendations to fix these issues.\n"
            "Return a JSON array of strings, one per recommendation."
        )

        raw = self._call_claude(prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item) for item in data]
        return [str(data)]

    def _analyze_issues_heuristic(self, issues: List[QualityIssue]) -> List[str]:
        """Generate rule-based recommendations from detected issues."""
        recommendations: List[str] = []
        criticals = [i for i in issues if i.severity.value == "critical"]
        highs = [i for i in issues if i.severity.value == "high"]

        if criticals:
            recommendations.append(
                f"URGENT: {len(criticals)} critical issue(s) require immediate attention. "
                "Address null rates and schema problems before any ML training."
            )
        if highs:
            recommendations.append(
                f"{len(highs)} high-severity issue(s) detected. "
                "Review outliers and high null rates that may bias model outputs."
            )

        for issue in issues:
            recommendations.append(
                f"[{issue.severity.value.upper()}] {issue.recommendation}"
            )

        return recommendations

    def generate_fix_sql(self, profile: DataProfile) -> str:
        """Generate SQL fix suggestions for common data quality issues.

        Args:
            profile: DataProfile with detected quality issues.

        Returns:
            SQL snippet string with fix suggestions.
        """
        if self._ai_available:
            try:
                return self._generate_fix_sql_via_claude(profile)
            except Exception as exc:
                logger.warning("Claude SQL generation failed: %s", exc)

        return self._generate_fix_sql_heuristic(profile)

    def _generate_fix_sql_via_claude(self, profile: DataProfile) -> str:
        """Use Claude to generate SQL fix queries."""
        summary = self._build_profile_summary(profile)
        prompt = (
            "Given this dataset profile with quality issues, generate SQL fix statements.\n\n"
            f"Profile:\n{json.dumps(summary, indent=2)}\n\n"
            f"Issues:\n{json.dumps([i.description for i in profile.quality_issues[:10]], indent=2)}\n\n"
            "Generate SQL statements (using standard SQL) to fix the data quality issues.\n"
            "Return plain SQL, no markdown."
        )
        return self._call_claude(prompt)

    def _generate_fix_sql_heuristic(self, profile: DataProfile) -> str:
        """Generate simple SQL fixes heuristically from the profile."""
        lines = [f"-- Data Quality Fixes for {profile.dataset_name}"]
        table = profile.dataset_name

        for col_name, col in profile.columns.items():
            if col.null_rate > 0 and col.numeric_stats is not None:
                ns = col.numeric_stats
                lines.append(
                    f"-- Fix: impute nulls in {col_name} with mean\n"
                    f"UPDATE {table} SET {col_name} = {ns.mean:.4f} WHERE {col_name} IS NULL;"
                )
            elif col.null_rate > 0 and col.categorical_stats is not None:
                cs = col.categorical_stats
                if cs.most_frequent:
                    lines.append(
                        f"-- Fix: impute nulls in {col_name} with mode\n"
                        f"UPDATE {table} SET {col_name} = '{cs.most_frequent}' WHERE {col_name} IS NULL;"
                    )

        pk_cols = list(profile.columns.keys())[:5]
        group_by = ", ".join(pk_cols)
        lines.append(
            f"\n-- Fix: remove duplicate rows\n"
            f"DELETE FROM {table} WHERE rowid NOT IN "
            f"(SELECT MIN(rowid) FROM {table} GROUP BY {group_by});"
        )
        return "\n".join(lines)
