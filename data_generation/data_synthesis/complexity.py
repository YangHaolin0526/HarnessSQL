"""Profile Spider 2.0 SQL complexity without treating predicted SQL as generation templates.

The repository does not ship public gold SQL.  For SQLite we therefore use only
the locally retained, officially result-verified predictions as a *distributional*
reference.  The profiler deliberately emits aggregate features and task IDs; the
question-generation prompt never receives reference SQL text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "benchmarks/spider2_repo/spider2-lite/spider2-lite.jsonl"
DEFAULT_SQL_DIR = (
    ROOT
    / "benchmarks/spider2_lite/output_context_v3_full_20260730_run1"
    / "agent_pred_incumbent_preserving_v6"
)
DEFAULT_CORRECT_IDS = (
    ROOT
    / "benchmarks/spider2_lite/output_context_v3_full_20260730_run1"
    / "agent_pred_incumbent_preserving_v6-ids.csv"
)
DEFAULT_CONTRACT_DIR = (
    ROOT
    / "benchmarks/spider2_lite/output_context_v3_full_20260730_run1"
    / "task_contracts"
)
DEFAULT_OUTPUT = ROOT / "artifacts/data_synthesis/sqlite/spider2_complexity"


FEATURE_PATTERNS: dict[str, str] = {
    "cte": r"\bWITH\b|\b[A-Za-z_][\w$]*\s+AS\s*\(",
    "recursive": r"\bWITH\s+RECURSIVE\b",
    "join": r"\bJOIN\b",
    "left_join": r"\bLEFT(?:\s+OUTER)?\s+JOIN\b",
    "select": r"\bSELECT\b",
    "group_by": r"\bGROUP\s+BY\b",
    "having": r"\bHAVING\b",
    "distinct": r"\bDISTINCT\b",
    "case": r"\bCASE\b",
    "window": r"\bOVER\s*\(",
    "window_navigation": r"\b(?:LAG|LEAD|FIRST_VALUE|LAST_VALUE|NTH_VALUE)\s*\(",
    "window_ranking": r"\b(?:ROW_NUMBER|RANK|DENSE_RANK|NTILE|PERCENT_RANK|CUME_DIST)\s*\(",
    "window_frame": r"\b(?:ROWS|RANGE|GROUPS)\s+BETWEEN\b",
    "set_operation": r"\b(?:UNION(?:\s+ALL)?|INTERSECT|EXCEPT)\b",
    "exists": r"\b(?:NOT\s+)?EXISTS\s*\(",
    "in_subquery": r"\bIN\s*\(\s*SELECT\b",
    "scalar_subquery": r"\(\s*SELECT\b",
    "conditional_aggregate": r"\b(?:SUM|COUNT|AVG|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?CASE\b",
    "date_time": r"\b(?:DATE|DATETIME|JULIANDAY|STRFTIME|TIMESTAMPDIFF|DATE_TRUNC)\s*\(",
    "json": r"\bJSON_(?:EXTRACT|EACH|ARRAY|OBJECT|TYPE)\s*\(",
    "string_transform": r"\b(?:SUBSTR|REPLACE|INSTR|LOWER|UPPER|TRIM|PRINTF)\s*\(",
    "math": r"\b(?:ABS|ROUND|POWER|SQRT|EXP|LOG|LN|SIN|COS|ACOS|RADIANS)\s*\(",
    "null_handling": r"\b(?:COALESCE|NULLIF|IFNULL)\s*\(",
    "limit": r"\bLIMIT\b",
}


def load_correct_ids(path: Path) -> set[str]:
    values: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            value = row[0].strip().removeprefix("sf_")
            if value and value != "instance_id":
                values.add(value)
    return values


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    return {
        item["instance_id"]: item
        for item in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def _max_select_depth(sql: str) -> int:
    """Approximate nesting depth at SELECT tokens, ignoring quoted content."""

    scrubbed = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "", sql)
    select_starts = {match.start() for match in re.finditer(r"\bSELECT\b", scrubbed, re.I)}
    depth = maximum = 0
    for index, char in enumerate(scrubbed):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif index in select_starts:
            maximum = max(maximum, depth)
    return maximum


def profile_sql(sql: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", sql.strip())
    counts = {
        name: len(re.findall(pattern, normalized, flags=re.IGNORECASE))
        for name, pattern in FEATURE_PATTERNS.items()
    }
    # WITH itself and every named CTE both match the broad pattern; retain a
    # conservative named-CTE estimate that is more useful for comparisons.
    cte_names = re.findall(
        r"(?:\bWITH\s+(?:RECURSIVE\s+)?|,)\s*([A-Za-z_][\w$]*)\s+AS\s*\(",
        normalized,
        flags=re.IGNORECASE,
    )
    counts["cte"] = len(cte_names)
    # ``(SELECT`` also occurs at the start of every CTE body.  Remove those
    # before labeling the remainder as nested/scalar subqueries.
    counts["scalar_subquery"] = max(0, counts["scalar_subquery"] - counts["cte"])
    features = {
        "sql_chars": len(sql),
        "sql_tokens": len(re.findall(r"[A-Za-z_][\w$]*|\d+(?:\.\d+)?|<>|!=|<=|>=|\S", sql)),
        "max_select_depth": _max_select_depth(sql),
        **counts,
    }
    # A transparent structural score.  It is used for quota/ranking only; it
    # does not claim to predict model difficulty perfectly.
    features["structural_score"] = round(
        0.012 * features["sql_tokens"]
        + 1.2 * features["cte"]
        + 0.8 * features["join"]
        + 1.4 * max(0, features["select"] - 1)
        + 2.2 * features["window"]
        + 2.0 * features["set_operation"]
        + 1.5 * features["exists"]
        + 1.2 * features["conditional_aggregate"]
        + 0.8 * features["case"]
        + 1.0 * features["date_time"]
        + 1.0 * features["json"]
        + 1.2 * features["max_select_depth"],
        3,
    )
    return features


def advanced_families(features: dict[str, Any]) -> list[str]:
    """Map measurable SQL features to independent hard-task structure families."""

    families = []
    if features["cte"] >= 3:
        families.append("multi_stage_cte")
    if features["window"]:
        families.append("window")
    if features["set_operation"] or features["exists"] or features["in_subquery"]:
        families.append("set_or_anti_join")
    if features["scalar_subquery"] >= 2 or features["max_select_depth"] >= 2:
        families.append("nested_subquery")
    if features["join"] >= 3 and features["cte"] >= 2:
        families.append("grain_safe_multi_fact")
    if features["date_time"] or features["window_navigation"] or features["recursive"]:
        families.append("temporal_cohort_or_sequence")
    if features["case"] >= 2 or features["conditional_aggregate"]:
        families.append("nontrivial_derived_classification")
    return families


def _load_risks(contract_path: Path) -> list[str]:
    if not contract_path.exists():
        return []
    payload = json.loads(contract_path.read_text())
    risks = payload.get("risk_flags", [])
    if not risks and isinstance(payload.get("contract"), dict):
        risks = payload["contract"].get("risk_flags", [])
    return sorted({str(item) for item in risks})


def build_reference_rows(
    tasks_path: Path,
    sql_dir: Path,
    correct_ids_path: Path,
    contract_dir: Path,
) -> list[dict[str, Any]]:
    tasks = load_tasks(tasks_path)
    correct = load_correct_ids(correct_ids_path)
    rows: list[dict[str, Any]] = []
    for task_id in sorted(correct):
        sql_path = sql_dir / f"{task_id}.sql"
        if task_id not in tasks or not sql_path.exists():
            continue
        task = tasks[task_id]
        rows.append(
            {
                "instance_id": task_id,
                "database_id": task["db"],
                "question_chars": len(task["question"]),
                "has_external_knowledge": bool(task.get("external_knowledge")),
                "external_knowledge": task.get("external_knowledge"),
                "risk_flags": _load_risks(contract_dir / f"{task_id}.json"),
                "sql_path": str(sql_path.resolve()),
                "features": profile_sql(sql_path.read_text()),
            }
        )
    return rows


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * probability
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (index - low))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = ["sql_chars", "sql_tokens", "structural_score", "cte", "select", "join", "max_select_depth"]
    distributions: dict[str, Any] = {}
    for key in numeric:
        values = [float(row["features"][key]) for row in rows]
        distributions[key] = {
            "min": min(values, default=0),
            "p25": round(percentile(values, 0.25), 3),
            "median": round(median(values), 3) if values else 0,
            "p75": round(percentile(values, 0.75), 3),
            "p90": round(percentile(values, 0.90), 3),
            "max": max(values, default=0),
            "mean": round(mean(values), 3) if values else 0,
        }
    prevalence = {}
    for feature in FEATURE_PATTERNS:
        count = sum(row["features"][feature] > 0 for row in rows)
        prevalence[feature] = {
            "tasks": count,
            "fraction": round(count / len(rows), 4) if rows else 0,
        }
    risks = Counter(risk for row in rows for risk in row["risk_flags"])
    return {
        "reference_kind": "official-result-verified local predictions; aggregate complexity use only",
        "reference_task_count": len(rows),
        "database_count": len({row["database_id"] for row in rows}),
        "external_knowledge_fraction": round(
            sum(row["has_external_knowledge"] for row in rows) / len(rows), 4
        ) if rows else 0,
        "distributions": distributions,
        "feature_prevalence": prevalence,
        "semantic_risk_prevalence": dict(risks.most_common()),
        "hard_target": {
            "policy": "match Spider 2.0 verified-reference p75 or exceed it on multiple independent axes",
            "minimum_structural_score": distributions["structural_score"]["p75"],
            "minimum_sql_tokens": distributions["sql_tokens"]["p75"],
            "required_advanced_families": 2,
            "advanced_families": [
                "multi_stage_cte",
                "window",
                "set_or_anti_join",
                "nested_subquery",
                "grain_safe_multi_fact",
                "temporal_cohort_or_sequence",
                "external_rule_or_nontrivial_derived_classification",
            ],
        },
        "curriculum_note": {
            "policy": (
                "Generate a bounded mixture across p25/median/p75/p90 bands; "
                "do not treat >=p75 as an unbounded target. Final promotion is empirical."
            ),
            "boundaries": {
                "foundation": "structural_score <= p25",
                "core": "p25 <= structural_score <= median",
                "growth": "median <= structural_score <= p75",
                "stretch": "p75 <= structural_score <= p90",
                "above_p90": "reject by default",
            },
            "default_mix": {
                "foundation": 0.20,
                "core": 0.45,
                "growth": 0.25,
                "stretch": 0.10,
            },
            "promotion": (
                "teacher must solve; use at least two student seeds; keep zero-success tasks only "
                "as limited stretch examples when the student produced a bounded near miss"
            ),
        },
    }


def compare_profiles(reference: dict[str, Any], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = reference["hard_target"]
    output = []
    for row in candidate_rows:
        features = row["features"]
        output.append(
            {
                **row,
                "meets_reference_p75_score": features["structural_score"] >= target["minimum_structural_score"],
                "meets_reference_p75_tokens": features["sql_tokens"] >= target["minimum_sql_tokens"],
            }
        )
    return {
        "candidate_count": len(output),
        "candidate_summary": summarize(output) if output else {},
        "rows": output,
    }


def markdown_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Spider 2.0 SQLite complexity profile",
        "",
        "> Reference SQL is restricted to locally retained predictions that passed official result evaluation. "
        "It is used only for aggregate complexity targets, never as text supplied to task generation.",
        "",
        f"- Verified reference tasks: {summary['reference_task_count']}",
        f"- Databases represented: {summary['database_count']}",
        f"- Tasks requiring external knowledge: {summary['external_knowledge_fraction']:.1%}",
        "",
        "## Structural distributions",
        "",
        "| feature | p25 | median | p75 | p90 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, values in summary["distributions"].items():
        lines.append(
            f"| {key} | {values['p25']} | {values['median']} | {values['p75']} | "
            f"{values['p90']} | {values['max']} |"
        )
    lines += ["", "## Operator prevalence", "", "| operator | tasks | share |", "|---|---:|---:|"]
    for key, values in sorted(
        summary["feature_prevalence"].items(), key=lambda item: (-item[1]["tasks"], item[0])
    ):
        lines.append(f"| {key} | {values['tasks']} | {values['fraction']:.1%} |")
    lines += ["", "## Highest structural references", ""]
    for row in sorted(rows, key=lambda item: -item["features"]["structural_score"])[:15]:
        f = row["features"]
        lines.append(
            f"- `{row['instance_id']}` / `{row['database_id']}`: score {f['structural_score']}, "
            f"tokens {f['sql_tokens']}, CTE {f['cte']}, SELECT {f['select']}, JOIN {f['join']}, "
            f"window {f['window']}, set-op {f['set_operation']}, depth {f['max_select_depth']}"
        )
    lines += [
        "",
        "## Generation target",
        "",
        "The legacy hard-pilot target was >= p75 and is retained for reproducibility only. New generation uses "
        "bounded foundation/core/growth/stretch bands ending at p90, with a 20/45/25/10 target mix. Length alone "
        "is insufficient; live execution, counterfactual mutations, teacher correctness and student rollout "
        "calibration are separate required gates.",
        "",
    ]
    return "\n".join(lines)


def run_profile(
    tasks_path: Path = DEFAULT_TASKS,
    sql_dir: Path = DEFAULT_SQL_DIR,
    correct_ids_path: Path = DEFAULT_CORRECT_IDS,
    contract_dir: Path = DEFAULT_CONTRACT_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    rows = build_reference_rows(tasks_path, sql_dir, correct_ids_path, contract_dir)
    summary = summarize(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reference_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output_dir / "profile.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "REPORT.md").write_text(markdown_report(summary, rows))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--sql-dir", type=Path, default=DEFAULT_SQL_DIR)
    parser.add_argument("--correct-ids", type=Path, default=DEFAULT_CORRECT_IDS)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run_profile(args.tasks, args.sql_dir, args.correct_ids, args.contract_dir, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
