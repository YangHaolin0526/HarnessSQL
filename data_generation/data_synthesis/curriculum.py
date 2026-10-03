"""Difficulty curriculum and empirical promotion rules for synthesized SQL tasks.

Static SQL complexity is useful for requesting a mixture, but it is not a
solvability oracle.  This module therefore keeps two decisions separate:

* a bounded structural band used while a question is being generated;
* a post-rollout promotion decision based on a correct teacher and the current
  student's multi-seed behavior.

The defaults are calibrated from the retained Spider 2.0-Lite SQLite
Codex+KTX run of Qwen3.5-9B.  They are priors, not labels copied from benchmark
questions, and can be replaced in a generated complexity profile.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Iterable

from .complexity import advanced_families, profile_sql


CURRICULUM_VERSION = "spider2-student-frontier-v1"
BANDS = ("foundation", "core", "growth", "stretch")
DEFAULT_MIX: dict[str, float] = OrderedDict(
    foundation=0.20,
    core=0.45,
    growth=0.25,
    stretch=0.10,
)

# Measured on the 106 officially result-verified SQLite reference queries.
# The exact cut points used in that audit (10/18/25) closely track the profile
# p25/median/p75 boundaries (11.4/16.8/22.8).
QWEN35_PRIOR = {
    "harness": "codex+ktx",
    "model": "Qwen/Qwen3.5-9B",
    "reference_tasks": 106,
    "accuracy_by_band": {
        "foundation": 0.591,
        "core": 0.308,
        "growth": 0.217,
        "stretch": 0.136,
    },
    "overall_spider2": {
        "sqlite_full_denominator": 34 / 135,
        "snowflake_full_denominator": 13 / 207,
    },
}

# Deliberately just below the historical SQLite Codex+KTX p90s
# (65 tool events, 32 successful executions, ~2.55M reported input tokens).
# A wrong rollout beyond these caps is a search/convergence failure, not a
# bounded near miss suitable for the stretch bucket.
STUDENT_RESOURCE_CAPS = {
    "tool_events": 60,
    "successful_sql_executions": 25,
    "input_tokens": 2_000_000,
}


def resource_within_cap(result: dict[str, Any]) -> bool:
    """Return whether one rollout stayed inside the calibrated SQLite caps."""

    return (
        int(result.get("input_tokens") or 0) <= STUDENT_RESOURCE_CAPS["input_tokens"]
        and sum((result.get("tool_counts") or {}).values()) <= STUDENT_RESOURCE_CAPS["tool_events"]
        and int(result.get("successful_execute_count") or 0)
        <= STUDENT_RESOURCE_CAPS["successful_sql_executions"]
    )


def parse_mix(value: str | dict[str, float] | None) -> dict[str, float]:
    """Parse and normalize a four-band mixture."""

    if value is None:
        raw = dict(DEFAULT_MIX)
    elif isinstance(value, dict):
        raw = {str(key): float(weight) for key, weight in value.items()}
    else:
        raw = {}
        for item in value.split(","):
            name, separator, weight = item.strip().partition("=")
            if not separator:
                raise ValueError(f"difficulty mix item must be band=weight: {item!r}")
            raw[name.strip()] = float(weight)
    unknown = sorted(set(raw) - set(BANDS))
    missing = sorted(set(BANDS) - set(raw))
    if unknown or missing:
        raise ValueError(f"difficulty mix bands mismatch; missing={missing}, unknown={unknown}")
    if any(raw[name] < 0 for name in BANDS):
        raise ValueError("difficulty weights cannot be negative")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("difficulty weights must have a positive sum")
    return {name: raw[name] / total for name in BANDS}


def allocate_mix(count: int, mix: str | dict[str, float] | None = None) -> dict[str, int]:
    """Allocate an exact count with deterministic largest remainders."""

    if count < 1:
        raise ValueError("count must be positive")
    weights = parse_mix(mix)
    exact = {name: count * weights[name] for name in BANDS}
    allocated = {name: math.floor(exact[name]) for name in BANDS}
    remaining = count - sum(allocated.values())
    # Prefer the learning-frontier bands when fractional remainders tie.
    tie_priority = {"growth": 0, "core": 1, "foundation": 2, "stretch": 3}
    order = sorted(BANDS, key=lambda name: (-(exact[name] - allocated[name]), tie_priority[name]))
    for name in order[:remaining]:
        allocated[name] += 1
    return allocated


def band_schedule(count: int, mix: str | dict[str, float] | None = None) -> list[str]:
    """Interleave exact quotas while keeping every prefix near the target mix."""

    quota = allocate_mix(count, mix)
    emitted = {name: 0 for name in BANDS}
    schedule: list[str] = []
    tie_priority = {"growth": 0, "core": 1, "foundation": 2, "stretch": 3}
    for position in range(1, count + 1):
        available = [name for name in BANDS if emitted[name] < quota[name]]
        selected = max(
            available,
            key=lambda name: (
                position * quota[name] / count - emitted[name],
                -tie_priority[name],
            ),
        )
        schedule.append(selected)
        emitted[selected] += 1
    return schedule


def _distribution(profile: dict[str, Any], feature: str, quantile: str) -> float:
    try:
        return float(profile["distributions"][feature][quantile])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"complexity profile lacks distributions.{feature}.{quantile}") from exc


def difficulty_contract(profile: dict[str, Any], band: str) -> dict[str, Any]:
    """Create a bounded structural request from reference quantiles.

    The upper bound is deliberate.  The previous hard-only pipeline requested
    ``>= p75`` and let candidates drift far beyond p90, which made nearly every
    generated pilot a student-capability outlier.
    """

    if band not in BANDS:
        raise ValueError(f"unknown difficulty band: {band!r}")
    score = {
        "p25": _distribution(profile, "structural_score", "p25"),
        "median": _distribution(profile, "structural_score", "median"),
        "p75": _distribution(profile, "structural_score", "p75"),
        "p90": _distribution(profile, "structural_score", "p90"),
    }
    tokens = {
        "p25": _distribution(profile, "sql_tokens", "p25"),
        "median": _distribution(profile, "sql_tokens", "median"),
        "p75": _distribution(profile, "sql_tokens", "p75"),
        "p90": _distribution(profile, "sql_tokens", "p90"),
    }
    contracts = {
        "foundation": (0.0, score["p25"], 24.0, tokens["p75"], 0),
        "core": (score["p25"], score["median"], tokens["p25"] * 0.65, tokens["p75"], 1),
        "growth": (score["median"], score["p75"], tokens["p25"], tokens["p90"], 1),
        "stretch": (score["p75"], score["p90"], tokens["median"], tokens["p90"] * 1.25, 2),
    }
    minimum_score, maximum_score, minimum_tokens, maximum_tokens, minimum_families = contracts[band]
    return {
        "curriculum_version": CURRICULUM_VERSION,
        "difficulty_band": band,
        "reference": profile.get("reference_kind", "Spider 2.0 verified-reference profile"),
        "minimum_structural_score": round(minimum_score, 3),
        "maximum_structural_score": round(maximum_score, 3),
        "minimum_sql_tokens": round(minimum_tokens, 3),
        "maximum_sql_tokens": round(maximum_tokens, 3),
        "minimum_advanced_families": minimum_families,
        "teacher_must_solve": True,
        "student_calibration_required": True,
        "above_reference_p90_default": "reject",
    }


def gate_sql(sql: str, contract: dict[str, Any]) -> dict[str, Any]:
    """Apply all available lower and upper structural bounds."""

    features = profile_sql(sql)
    families = advanced_families(features)
    checks: dict[str, bool] = {}
    comparisons = (
        ("structural_score", "minimum_structural_score", "at_least", lambda a, b: a >= b),
        ("structural_score", "maximum_structural_score", "at_most", lambda a, b: a <= b),
        ("sql_tokens", "minimum_sql_tokens", "at_least", lambda a, b: a >= b),
        ("sql_tokens", "maximum_sql_tokens", "at_most", lambda a, b: a <= b),
    )
    for feature, threshold, direction, comparator in comparisons:
        if threshold in contract:
            checks[f"{feature}_{direction}_{threshold}"] = comparator(
                float(features[feature]), float(contract[threshold])
            )
    if "minimum_advanced_families" in contract:
        checks["advanced_family_floor"] = len(families) >= int(contract["minimum_advanced_families"])
    return {
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "features": features,
        "families": families,
        "contract": contract,
    }


def classify_rollouts(
    *,
    teacher_solved: bool,
    student_outcomes: Iterable[dict[str, Any]],
    static_band: str,
    above_reference_p90: bool = False,
    teacher_empty_response: bool = False,
    minimum_student_attempts: int = 0,
) -> dict[str, Any]:
    """Make a conservative post-rollout keep/reject decision.

    Each student outcome may contain ``correct``, ``near_miss`` and
    ``resource_within_cap``.  A zero-success candidate is retained only as a
    limited stretch example when at least one seed is a bounded near miss.
    """

    raw_outcomes = list(student_outcomes)
    empty_student_attempts = sum(
        bool(item.get("empty_response_exhausted")) for item in raw_outcomes
    )
    outcomes = [item for item in raw_outcomes if not item.get("empty_response_exhausted")]
    successes = sum(bool(item.get("correct")) for item in outcomes)
    near_misses = sum(
        bool(item.get("near_miss")) and bool(item.get("resource_within_cap", True))
        for item in outcomes
    )
    attempts = len(outcomes)
    success_rate = successes / attempts if attempts else None
    if teacher_empty_response:
        decision, observed, reason = "hold", None, "teacher_empty_response_exhausted"
    elif not teacher_solved:
        decision, observed, reason = "reject", None, "teacher_failed"
    elif above_reference_p90:
        decision, observed, reason = "reject", None, "above_reference_p90"
    elif attempts < minimum_student_attempts:
        decision, observed, reason = "hold", None, "student_valid_rollouts_below_minimum"
    elif not outcomes:
        decision, observed, reason = "hold", None, "student_calibration_missing"
    elif successes == attempts:
        decision, observed, reason = "keep", "foundation", "student_consistently_solved"
    elif successes / attempts >= 0.5:
        decision, observed, reason = "keep", "core", "student_seed_disagreement"
    elif successes > 0:
        decision, observed, reason = "keep", "growth", "student_rarely_solved"
    elif near_misses > 0:
        decision, observed, reason = "keep_limited", "stretch", "teacher_solved_student_near_miss"
    else:
        decision, observed, reason = "reject", None, "student_capacity_gap"
    return {
        "curriculum_version": CURRICULUM_VERSION,
        "decision": decision,
        "reason": reason,
        "static_band": static_band,
        "observed_band": observed,
        "teacher_solved": teacher_solved,
        "student_attempts": attempts,
        "student_empty_response_attempts": empty_student_attempts,
        "minimum_student_attempts": minimum_student_attempts,
        "student_successes": successes,
        "student_success_rate": success_rate,
        "bounded_near_misses": near_misses,
    }
