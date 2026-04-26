"""QualityReporter — formats DataProfile and ValidationResult as rich outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from autonomous_dq_agent.models.profile import DataProfile, IssueSeverity
from autonomous_dq_agent.models.validation import ValidationResult

logger = logging.getLogger(__name__)

_SEVERITY_COLORS = {
    IssueSeverity.CRITICAL: "#c0392b",
    IssueSeverity.HIGH: "#e67e22",
    IssueSeverity.MEDIUM: "#f1c40f",
    IssueSeverity.LOW: "#2ecc71",
    IssueSeverity.INFO: "#3498db",
}


class QualityReporter:
    """Generates text, JSON, and HTML reports from DQ profiles and validation results.

    HTML reports are self-contained (no external dependencies) and suitable
    for embedding in data platform UIs or CI artifacts.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        """Initialize reporter with output directory."""
        self.output_dir = Path(output_dir)

    def _ensure_output_dir(self) -> None:
        """Create the output directory if it does not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_text_report(self, profile: DataProfile) -> str:
        """Return a plain-text summary of the DataProfile.

        Args:
            profile: Computed DataProfile to summarize.

        Returns:
            Multi-line text string with key metrics and issues.
        """
        lines = [
            "=" * 70,
            f"DATA QUALITY REPORT — {profile.dataset_name}",
            f"Generated: {profile.profiled_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "=" * 70,
            "",
            "DATASET OVERVIEW",
            f"  Rows:            {profile.row_count:,}",
            f"  Columns:         {profile.column_count}",
            f"  Memory:          {profile.memory_mb:.2f} MB",
            f"  Duplicate Rows:  {profile.duplicate_row_count:,} ({profile.duplicate_row_rate:.2%})",
            f"  Quality Score:   {profile.overall_quality_score:.1f} / 100",
            "",
            f"QUALITY ISSUES ({len(profile.quality_issues)} total)",
        ]

        sev_counts: Dict[str, int] = {}
        for issue in profile.quality_issues:
            sev_counts[issue.severity.value] = (
                sev_counts.get(issue.severity.value, 0) + 1
            )
        for sev in ["critical", "high", "medium", "low", "info"]:
            count = sev_counts.get(sev, 0)
            if count:
                lines.append(f"  {sev.upper():10s}: {count}")

        if profile.quality_issues:
            lines.append("")
            lines.append("ISSUE DETAILS")
            for i, issue in enumerate(profile.quality_issues, 1):
                col_label = f"[{issue.column}]" if issue.column else "[dataset]"
                lines.append(
                    f"  {i:3d}. [{issue.severity.value.upper()}] {col_label} {issue.description}"
                )
                lines.append(f"       -> {issue.recommendation}")

        lines.append("")
        lines.append("COLUMN SUMMARY")
        for col_name, col in profile.columns.items():
            null_info = f"null={col.null_rate:.1%}"
            distinct_info = f"distinct={col.distinct_count}"
            lines.append(
                f"  {col_name:30s} {col.column_type.value:12s} {null_info:15s} {distinct_info}"
            )

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def generate_json_report(self, profile: DataProfile) -> Dict[str, Any]:
        """Return the profile as a JSON-serializable dictionary.

        Args:
            profile: DataProfile to serialize.

        Returns:
            Dictionary suitable for json.dumps().
        """
        return json.loads(profile.model_dump_json())

    def save_json_report(
        self, profile: DataProfile, filename: Optional[str] = None
    ) -> Path:
        """Write JSON report to disk.

        Args:
            profile: DataProfile to serialize.
            filename: Output filename (auto-generated if None).

        Returns:
            Path to written file.
        """
        self._ensure_output_dir()
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{profile.dataset_name}_{ts}.json"
        out_path = self.output_dir / filename
        out_path.write_text(profile.model_dump_json(indent=2))
        logger.info("JSON report saved to %s", out_path)
        return out_path

    def generate_html_report(self, profile: DataProfile) -> str:
        """Return a self-contained HTML string for the DataProfile.

        Args:
            profile: DataProfile to render.

        Returns:
            Complete HTML document as a string.
        """
        score = profile.overall_quality_score
        score_color = (
            "#2ecc71" if score >= 80 else "#e67e22" if score >= 50 else "#c0392b"
        )

        issues_html = ""
        for issue in profile.quality_issues:
            color = _SEVERITY_COLORS.get(issue.severity, "#999")
            col_label = issue.column or "dataset"
            issues_html += f"""
            <tr>
              <td><span style="color:{color};font-weight:bold">{issue.severity.value.upper()}</span></td>
              <td><code>{col_label}</code></td>
              <td>{issue.issue_type.value}</td>
              <td>{issue.description}</td>
              <td><em>{issue.recommendation}</em></td>
            </tr>"""

        columns_html = ""
        for col_name, col in profile.columns.items():
            null_badge = ""
            if col.null_rate > 0.5:
                null_badge = (
                    f'<span style="color:#c0392b">&#9888; {col.null_rate:.1%}</span>'
                )
            elif col.null_rate > 0.1:
                null_badge = f'<span style="color:#e67e22">{col.null_rate:.1%}</span>'
            else:
                null_badge = f'<span style="color:#2ecc71">{col.null_rate:.1%}</span>'

            extra = ""
            if col.numeric_stats:
                ns = col.numeric_stats
                extra = (
                    f"mean={ns.mean:.2f}, std={ns.std:.2f}, outliers={ns.outlier_count}"
                )
            elif col.categorical_stats:
                cs = col.categorical_stats
                extra = f"unique={cs.unique_count}, top={cs.most_frequent!r}"

            columns_html += f"""
            <tr>
              <td><strong>{col_name}</strong></td>
              <td>{col.dtype}</td>
              <td>{col.column_type.value}</td>
              <td>{null_badge}</td>
              <td>{col.distinct_count:,}</td>
              <td style="font-size:0.85em;color:#666">{extra}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DQ Report — {profile.dataset_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #f9f9f9; color: #333; }}
  h1 {{ color: #2c3e50; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 1.5rem; }}
  .score-box {{ display:inline-block; background:{score_color}; color:#fff; padding: 0.6rem 1.4rem; border-radius:8px; font-size:2rem; font-weight:bold; margin-bottom:1.5rem; }}
  .cards {{ display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:2rem; }}
  .card {{ background:#fff; border-radius:8px; padding:1rem 1.5rem; box-shadow:0 1px 4px rgba(0,0,0,0.1); min-width:150px; }}
  .card .label {{ font-size:0.75em; color:#888; text-transform:uppercase; letter-spacing:0.05em; }}
  .card .value {{ font-size:1.6rem; font-weight:bold; color:#2c3e50; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,0.1); margin-bottom:2rem; }}
  th {{ background:#2c3e50; color:#fff; padding:0.7rem 1rem; text-align:left; font-size:0.85em; text-transform:uppercase; }}
  td {{ padding:0.6rem 1rem; border-bottom:1px solid #eee; vertical-align:top; font-size:0.9em; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover td {{ background:#f0f4f8; }}
  h2 {{ color:#34495e; border-bottom:2px solid #ecf0f1; padding-bottom:0.3rem; }}
</style>
</head>
<body>
<h1>Data Quality Report — {profile.dataset_name}</h1>
<p class="meta">Generated {profile.profiled_at.strftime('%Y-%m-%d %H:%M:%S UTC')} &bull; {profile.row_count:,} rows &bull; {profile.column_count} columns &bull; {profile.memory_mb:.2f} MB</p>

<div class="score-box">Quality Score: {score:.1f}/100</div>

<div class="cards">
  <div class="card"><div class="label">Rows</div><div class="value">{profile.row_count:,}</div></div>
  <div class="card"><div class="label">Columns</div><div class="value">{profile.column_count}</div></div>
  <div class="card"><div class="label">Duplicates</div><div class="value">{profile.duplicate_row_count:,}</div></div>
  <div class="card"><div class="label">Issues</div><div class="value">{len(profile.quality_issues)}</div></div>
  <div class="card"><div class="label">Critical</div><div class="value" style="color:#c0392b">{profile.critical_issue_count()}</div></div>
</div>

<h2>Quality Issues</h2>
<table>
  <thead><tr><th>Severity</th><th>Column</th><th>Type</th><th>Description</th><th>Recommendation</th></tr></thead>
  <tbody>{issues_html or "<tr><td colspan='5' style='text-align:center;color:#2ecc71'>No issues detected</td></tr>"}</tbody>
</table>

<h2>Column Profiles</h2>
<table>
  <thead><tr><th>Column</th><th>Dtype</th><th>Type</th><th>Null Rate</th><th>Distinct</th><th>Stats</th></tr></thead>
  <tbody>{columns_html}</tbody>
</table>

</body>
</html>"""
        return html

    def save_html_report(
        self, profile: DataProfile, filename: Optional[str] = None
    ) -> Path:
        """Write HTML report to disk.

        Args:
            profile: DataProfile to render.
            filename: Output filename (auto-generated if None).

        Returns:
            Path to written file.
        """
        self._ensure_output_dir()
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{profile.dataset_name}_{ts}.html"
        out_path = self.output_dir / filename
        out_path.write_text(self.generate_html_report(profile), encoding="utf-8")
        logger.info("HTML report saved to %s", out_path)
        return out_path

    def generate_validation_text_report(self, result: ValidationResult) -> str:
        """Return plain-text summary of a ValidationResult.

        Args:
            result: ValidationResult from DataValidator.

        Returns:
            Multi-line text string.
        """
        status = "PASS" if result.success else "FAIL"
        lines = [
            "=" * 70,
            f"VALIDATION REPORT — {result.dataset_name} [{status}]",
            f"Suite: {result.suite_name}",
            f"Validated: {result.validated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "=" * 70,
            "",
            f"  Total expectations:      {result.evaluated_expectations}",
            f"  Passed:                  {result.successful_expectations}",
            f"  Failed:                  {result.failed_expectations}",
            f"  Success rate:            {result.success_percent:.1f}%",
            "",
        ]
        failed = result.failed_results()
        if failed:
            lines.append("FAILED EXPECTATIONS")
            for i, r in enumerate(failed, 1):
                lines.append(f"  {i:3d}. {r.expectation.description}")
                if r.error_message:
                    lines.append(f"       Error: {r.error_message}")
                elif r.observed_value is not None:
                    lines.append(f"       Observed: {r.observed_value}")
        lines.append("=" * 70)
        return "\n".join(lines)
