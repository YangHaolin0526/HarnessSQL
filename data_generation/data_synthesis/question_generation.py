"""Model-assisted, catalog-constrained generation of new Spider 2.0 tasks.

This module is intentionally separate from deterministic validation.  A model
may propose a blueprint, but it cannot promote one: catalog grounding, live
execution, mutation checks, and complexity gates remain programmatic.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .complexity import DEFAULT_OUTPUT as DEFAULT_COMPLEXITY_DIR
from .curriculum import (
    BANDS,
    DEFAULT_MIX,
    band_schedule,
    difficulty_contract,
    gate_sql,
    parse_mix,
)
from .model_backends import (
    EmptyResponseExhausted,
    ModelBackend,
    add_backend_arguments,
    backend_from_args,
)
from .pipeline import build_sample, load_catalog


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_ROOT = ROOT / "benchmarks/spider2_repo/spider2-lite/resource/databases"
DEFAULT_CATALOG_DIR = ROOT / "artifacts/data_synthesis/sqlite/catalogs"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/data_synthesis/sqlite/model_generation"

GENERATION_PROFILES = (
    "standard",
    "student-frontier-hard",
    "student-frontier-extra-hard",
    "student-frontier-target40",
    "student-frontier-27b",
)
STUDENT_FRONTIER_HARD_GUIDANCE = """This batch is intentionally harder than the standard curriculum.
Within the supplied bounded contract, prefer questions whose difficulty comes from two or more interacting
semantic decisions, not from decorative SQL. Useful decision families include: metric grain followed by a
second aggregation; cohort-specific denominators; anti-join or retained-population logic; time direction plus
missing sentinels; raw-versus-rounded downstream computation; per-group argmax/ties; conditional windows; and
JSON/date normalization before aggregation. Use only combinations that the selected database naturally supports.

For growth/stretch slots, a candidate should normally require at least two independently checkable semantic
choices and at least two advanced SQL families when the structural contract permits it. Avoid direct lookup,
single GROUP BY summaries, cosmetic CTE chains, arbitrary long joins, and ambiguous tricks. Spell out every
population, calculation stage, null/sentinel rule, boundary, tie policy, and rounding stage so the challenge is
reasoning rather than guessing. The target remains at or below the Spider 2.0 verified-reference p90 ceiling."""

STUDENT_FRONTIER_EXTRA_HARD_GUIDANCE = """This batch is a calibrated step above student-frontier-hard.
The preceding 200-task batch scored about 50% with Qwen3.5-9B. Offline outcome analysis found that tasks combining
at least four measured advanced SQL families with either temporal/sequence reasoning or set/anti-population logic,
plus explicit grain control or a genuinely nested comparison, were below the desired 40% frontier. Design a
natural task around that interaction while staying at or below the Spider 2.0 verified-reference p90 ceiling.

Every candidate must require at least four measured advanced SQL families. It must include (a) temporal cohort,
ordered-sequence, navigation, recursive, or set/anti-population reasoning and (b) grain-safe multi-fact aggregation
or a nested subquery whose result changes the selected population. Prefer at least three independently checkable
semantic decisions, such as cohort membership followed by retained-population comparison and tie-aware ranking;
event-order direction plus missing-value eligibility and per-entity baselines; or anti-population logic plus a
grain-safe denominator and a downstream threshold. Do not satisfy these constraints with dead CTEs, redundant
windows, arbitrary joins, or decorative nesting.

Keep the instruction self-contained and explicit about entity grain, eligible population, time direction and
boundaries, missing sentinels, denominator, tie policy, raw-versus-rounded downstream use, and output ordering.
Difficulty must come from composing clear rules, never from ambiguity or unstated business knowledge."""

STUDENT_FRONTIER_TARGET40_GUIDANCE = """This batch targets below 40% accuracy for Qwen3.5-9B after an earlier
extra-hard batch unexpectedly scored 55/95 (57.9%). The failure analysis showed that merely combining cohort,
share-of-total and dense rank is too routine. Build the task around a genuinely correlated exclusion or ordered
event decision that cannot be solved as a flat GROUP BY.

Every candidate must use a correlated EXISTS/NOT EXISTS population decision and a real navigation window
(LAG, LEAD, FIRST_VALUE, LAST_VALUE or NTH_VALUE). It must also preserve grain across at least three joins,
contain a nested population-dependent comparison, and use at least five measured advanced SQL families. Make the
EXISTS predicate and navigation direction independently meaningful: for example, exclude entities with a
disqualifying event across all history, then compare eligible entities' first/next/previous events within a
bounded outcome period. A simple anti-list followed by revenue share and DENSE_RANK is not sufficient.

Require at least four independently checkable semantic decisions and at least four executable single-fault
mutations covering different decisions. Explicitly state correlation grain, event ordering and tie-breaker,
time boundaries, missing-value eligibility, denominator, raw-versus-rounded stage, and deterministic output
order. Keep the SQL compactly inside the existing stretch upper bounds: aim below 360 measured SQL tokens so a
repair does not oscillate at the 380.625-token ceiling. Every mutation id must be unique and descriptive (for
example m1_population, m2_direction, m3_boundary, m4_denominator). Never add decorative structure.

Treat the structural-score ceiling as a hard design budget, not merely a repair-time check. Aim for a measured
structural score of 25-31 as well as 220-340 SQL tokens. A reliable compact shape is three or four CTEs: one
correlated eligibility population, one joined event stream containing exactly one navigation window, and one
downstream aggregation/comparison. Normally use no more than six SELECT clauses, five joins, two scalar
subqueries, two window expressions, and four CTEs. Reuse a computed population or metric instead of repeating
the same EXISTS predicate or scalar denominator. Do not add a ranking window unless the requested output truly
needs rank. If a repair reports excessive structural score, remove repeated SELECTs/windows/subqueries first;
shortening aliases or prose does not reduce structural complexity."""

STUDENT_FRONTIER_27B_GUIDANCE = """This continuation targets a substantially stronger Qwen3.6-27B student.
Difficulty must come from composing compact, independently necessary semantic decisions, while staying inside
the same executable Spider reference ceiling. Do not turn the task into merely longer SQL.

Every candidate must contain at least two correlated EXISTS/NOT EXISTS occurrences that express two different
population decisions, at least two navigation-window occurrences whose directions or eligibility rules are
independently meaningful, at least four joins, and at least five measured advanced SQL families; six families are
preferred when they arise naturally. A suitable task
usually combines: an all-history disqualifier; a bounded cohort qualifier; an ordered event transition with a
deterministic tie-breaker; an entity-specific baseline or next/previous-event comparison; a population-dependent
threshold or denominator; and a downstream class/rank/tie decision. Require at least five executable single-fault
mutations, each targeting a different semantic rule.

Keep the implementation compact: target structural score 27-32 and 250-360 measured SQL tokens, normally using
three or four CTEs, at most six SELECT clauses, five joins, three scalar subqueries, and two OVER clauses. Reuse
computed eligibility and baselines. Every stated rule must affect the result, every result must be non-empty, and
every mutation must be distinguished by live execution. Never use decorative joins, windows, or nesting."""


def generation_guidance(generation_profile: str) -> str | None:
    if generation_profile == "student-frontier-hard":
        return STUDENT_FRONTIER_HARD_GUIDANCE
    if generation_profile == "student-frontier-extra-hard":
        return STUDENT_FRONTIER_EXTRA_HARD_GUIDANCE
    if generation_profile == "student-frontier-target40":
        return STUDENT_FRONTIER_TARGET40_GUIDANCE
    if generation_profile == "student-frontier-27b":
        return STUDENT_FRONTIER_27B_GUIDANCE
    return None


def generation_profile_errors(
    generation_profile: str,
    difficulty_band: str,
    gate: dict[str, Any],
) -> list[str]:
    """Apply measured profile-specific floors beyond the shared band contract."""

    families = set(gate.get("families") or [])
    if generation_profile == "student-frontier-hard" and difficulty_band in {"growth", "stretch"}:
        if len(families) < 2:
            return [
                "student-frontier-hard growth/stretch candidates require at least two "
                "measured advanced SQL families"
            ]
    if generation_profile not in {
        "student-frontier-extra-hard",
        "student-frontier-target40",
        "student-frontier-27b",
    }:
        return []
    errors = []
    if difficulty_band != "stretch":
        errors.append("student-frontier-extra-hard candidates must use the stretch structural band")
    if len(families) < 4:
        errors.append(
            "student-frontier-extra-hard candidates require at least four measured advanced SQL families"
        )
    if not families & {"temporal_cohort_or_sequence", "set_or_anti_join"}:
        errors.append(
            "student-frontier-extra-hard candidates require temporal/sequence or set/anti-population structure"
        )
    if not families & {"grain_safe_multi_fact", "nested_subquery"}:
        errors.append(
            "student-frontier-extra-hard candidates require grain-safe multi-fact or nested population structure"
        )
    if generation_profile in {"student-frontier-target40", "student-frontier-27b"}:
        features = gate.get("features") or {}
        if len(families) < 5:
            errors.append(
                "student-frontier-target40 candidates require at least five measured advanced SQL families"
            )
        if int(features.get("exists", 0)) < 1:
            errors.append(
                "student-frontier-target40 candidates require a measured EXISTS/NOT EXISTS population decision"
            )
        if int(features.get("window_navigation", 0)) < 1:
            errors.append(
                "student-frontier-target40 candidates require a measured navigation window"
            )
        if int(features.get("join", 0)) < 3:
            errors.append(
                "student-frontier-target40 candidates require at least three measured joins"
            )
    if generation_profile == "student-frontier-27b":
        features = gate.get("features") or {}
        if int(features.get("exists", 0)) < 2:
            errors.append(
                "student-frontier-27b candidates require two correlated EXISTS/NOT EXISTS occurrences"
            )
        if int(features.get("window_navigation", 0)) < 2:
            errors.append(
                "student-frontier-27b candidates require two measured navigation-window occurrences"
            )
        if int(features.get("join", 0)) < 4:
            errors.append("student-frontier-27b candidates require at least four measured joins")
    return errors


def _compact(value: Any, limit: int = 1000) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return value
    return text[:limit] + "…"


def _stable_order_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def select_connected_catalog_tables(
    catalog: dict[str, Any], *, limit: int, focus_seed: str
) -> list[str]:
    """Choose a deterministic, semantically rich executable-join component.

    The full offline catalog remains authoritative for validation.  This only
    bounds the model-facing search space so a large database does not turn one
    generation request into unconstrained schema browsing.
    """

    tables = {str(table["name"]): table for table in catalog.get("tables") or []}
    if len(tables) <= limit:
        return list(tables)
    adjacency = {name: set() for name in tables}
    for edge in catalog.get("joins") or []:
        left, right = str(edge["left_table"]), str(edge["right_table"])
        if left in tables and right in tables:
            adjacency[left].add(right)
            adjacency[right].add(left)

    components: list[set[str]] = []
    unseen = set(tables)
    while unseen:
        start = min(unseen)
        component = set()
        pending = [start]
        while pending:
            name = pending.pop()
            if name in component:
                continue
            component.add(name)
            pending.extend(adjacency[name] - component)
        unseen -= component
        components.append(component)

    def table_signal(name: str) -> tuple[int, int, int]:
        columns = tables[name].get("columns") or []
        temporal = sum(column.get("type_family") == "date_time" for column in columns)
        numeric = sum(column.get("type_family") == "numeric" for column in columns)
        return temporal, numeric, len(adjacency[name])

    def component_score(component: set[str]) -> tuple[int, int, int, str]:
        signals = [table_signal(name) for name in component]
        return (
            sum(value[0] for value in signals),
            sum(value[2] for value in signals),
            sum(value[1] for value in signals),
            _stable_order_key(focus_seed, "|".join(sorted(component))),
        )

    # Prefer components that can naturally support temporal and multi-table
    # reasoning. The stable hash only breaks equally rich alternatives.
    component = max(components, key=component_score)
    temporal_anchors = [name for name in component if table_signal(name)[0] and adjacency[name]]
    anchors = temporal_anchors or [name for name in component if adjacency[name]] or list(component)
    anchor = max(
        anchors,
        key=lambda name: (
            table_signal(name),
            _stable_order_key(focus_seed, name),
        ),
    )
    selected: list[str] = []
    pending = [anchor]
    seen = set()
    while pending and len(selected) < limit:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        selected.append(current)
        neighbors = sorted(
            adjacency[current] - seen,
            key=lambda name: (
                -table_signal(name)[0],
                -table_signal(name)[2],
                _stable_order_key(focus_seed, f"{current}|{name}"),
            ),
        )
        pending.extend(neighbors)
    return selected


def catalog_digest(
    catalog: dict[str, Any],
    max_tables: int = 60,
    *,
    focus_seed: str = "",
) -> dict[str, Any]:
    """Keep mapping evidence while bounding prompt size.

    The offline catalog remains the complete source of truth.  The model-facing
    digest includes every column plus compact distribution evidence, but does
    not repeat full row samples or large top-value payloads for every numeric
    field.  This cut makes large Spider databases practical for batch API
    generation without falling back to DDL-only prompting.
    """

    selected_names = select_connected_catalog_tables(
        catalog, limit=max_tables, focus_seed=focus_seed or str(catalog["database_id"])
    )
    selected_set = set(selected_names)
    selected_tables = {
        str(table["name"]): table for table in catalog["tables"] if str(table["name"]) in selected_set
    }
    tables = []
    for name in selected_names:
        table = selected_tables[name]
        tables.append(
            {
                "name": table["name"],
                "kind": table.get("kind"),
                "row_count": table.get("row_count"),
                "columns": [
                    {
                        "name": column["name"],
                        "type": column.get("declared_type"),
                        "type_family": column.get("type_family"),
                        "description": (column.get("description") or "")[:160] or None,
                        "tags": column.get("semantic_tags", []),
                        "null_fraction": column.get("distribution", {}).get("null_fraction"),
                        "sample_distinct": column.get("distribution", {}).get("sample_distinct_count"),
                        "sample_values": _compact(
                            column.get("distribution", {}).get("top_values", [])[:2], 240
                        )
                        if column.get("type_family") in {"text", "date_time"}
                        else [],
                        "numeric_range": {
                            key: column.get("distribution", {}).get("numeric", {}).get(key)
                            for key in ("min", "max")
                        }
                        if column.get("type_family") == "numeric"
                        else None,
                    }
                    for column in table["columns"]
                ],
            }
        )
    joins = [
        {
            "left": f"{edge['left_table']}.{edge['left_column']}",
            "right": f"{edge['right_table']}.{edge['right_column']}",
            "source": edge.get("source"),
            "confidence": edge.get("confidence"),
            "relationship": edge.get("relationship_estimate"),
        }
        for edge in catalog.get("joins", [])
        if str(edge["left_table"]) in selected_set and str(edge["right_table"]) in selected_set
    ]
    return {
        "database_id": catalog["database_id"],
        "catalog_version": catalog["catalog_version"],
        "catalog_subset": {
            "policy": "deterministic_semantically_rich_connected_executable_join_subgraph",
            "full_catalog_table_count": len(catalog["tables"]),
            "selected_table_count": len(tables),
            "focus_seed": focus_seed or None,
        },
        "tables": tables,
        "executable_join_graph": joins,
        "important_evidence_note": (
            "Samples and distributions are grounding evidence, not exhaustive data. "
            "Only joins in executable_join_graph may be declared in blueprint.joins."
        ),
    }


def complexity_gate(
    sql: str, profile: dict[str, Any], difficulty_band: str | None = None
) -> dict[str, Any]:
    """Backward-compatible gate plus the bounded curriculum gate.

    Older hand-authored pilots carry a single ``hard_target``.  Newly generated
    tasks must name a band and receive both a floor and a ceiling.
    """

    target = difficulty_contract(profile, difficulty_band) if difficulty_band else profile["hard_target"]
    if "required_advanced_families" in target and "minimum_advanced_families" not in target:
        target = {**target, "minimum_advanced_families": target["required_advanced_families"]}
    return gate_sql(sql, target)


def parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.S | re.I)
    if fenced:
        stripped = fenced.group(1)
    else:
        first, last = stripped.find("{"), stripped.rfind("}")
        if first >= 0 and last > first:
            stripped = stripped[first : last + 1]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("model response must be one JSON object")
    return value


SYSTEM_PROMPT = """You design NEW difficulty-calibrated text-to-SQL tasks over a fixed Spider 2.0 SQLite database.
Work SQL-first. Return exactly one JSON blueprint and no prose.

Hard constraints:
- Never imitate or reconstruct an existing Spider question. You are given no reference SQL.
- Use only the catalog tables, columns, values and executable join edges supplied by the user.
- The SQL must be one read-only SQLite SELECT/WITH statement and must produce a useful nonempty result.
- Match the supplied difficulty_contract, including its UPPER bounds. Do not add CTEs, windows, joins, or
  semantic conditions merely to look difficult. Foundation/core tasks may intentionally use fewer advanced families.
- The English instruction must completely specify population, metric grain, denominator, date boundaries,
  null handling, rounding, ordering/ties and output columns. It must explicitly list the exact output aliases
  from expected_columns so an alias-sensitive verifier never creates a hidden requirement.
- If rounding can affect a downstream average, threshold, rank, or tie, explicitly state whether that operation
  uses the raw or rounded value and where rounding occurs. Prefer computing and ranking raw values and rounding
  only final outputs unless the task intentionally teaches a rounded intermediate.
- Give at least three single-fault mutations. Each old string must occur literally in SQL and each replacement
  should still execute while changing the answer.
- roles are catalog search constraints; every selected table/column must exist. joins must be four-string arrays
  [left_table,left_column,right_table,right_column] found in executable_join_graph.

Required JSON fields: sample_id, database_id, difficulty_band, template_id, instruction, sql, expected_columns,
row_bounds, operators, roles, joins, mutations. Do not add markdown fences."""


def draft_prompt(
    digest: dict[str, Any],
    profile: dict[str, Any],
    sample_id: str,
    difficulty_band: str,
    diversity_memory: list[dict[str, Any]],
    generation_profile: str = "standard",
) -> str:
    target = difficulty_contract(profile, difficulty_band)
    payload = {
        "sample_id": sample_id,
        "database_catalog": digest,
        "difficulty_contract": target,
        "diversity_memory": diversity_memory[-20:],
        "generation_profile": generation_profile,
        "generation_guidance": generation_guidance(generation_profile),
        "blueprint_example_shape": {
            "sample_id": sample_id,
            "database_id": digest["database_id"],
            "difficulty_band": difficulty_band,
            "template_id": "novel_semantic_template_name",
            "instruction": "precise business question",
            "sql": "WITH ... SELECT ...",
            "expected_columns": ["alias_1", "alias_2"],
            "row_bounds": [1, 1000],
            "operators": ["multi_stage_cte", "window"],
            "roles": [
                {"role": "fact", "kind": "table", "query": "semantic search terms", "selected": "table"},
                {
                    "role": "metric",
                    "kind": "column",
                    "table": "table",
                    "type": "numeric",
                    "query": "semantic search terms",
                    "selected": "column",
                },
            ],
            "joins": [["table_a", "key", "table_b", "key"]],
            "mutations": [
                {"id": "single_fault", "category": "denominator", "old": "literal SQL", "new": "replacement"}
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def repair_prompt(
    digest: dict[str, Any],
    profile: dict[str, Any],
    difficulty_band: str,
    candidate: dict[str, Any] | None,
    errors: list[str],
    generation_profile: str = "standard",
) -> str:
    return json.dumps(
        {
            "task": "Repair the candidate and return a complete replacement JSON blueprint only.",
            "catalog_reference": (
                "Use the unchanged database_catalog from the preceding user message; it is not repeated here."
            ),
            "difficulty_contract": difficulty_contract(profile, difficulty_band),
            "candidate": candidate,
            "validator_errors": errors,
            "generation_profile": generation_profile,
            "generation_guidance": generation_guidance(generation_profile),
            "warning": (
                "Stay inside both bounds. Simplify an over-complex candidate and strengthen an under-complex one."
            ),
        },
        ensure_ascii=False,
    )


def _validate_candidate_shape(
    candidate: dict[str, Any], expected_id: str, database_id: str, difficulty_band: str
) -> list[str]:
    required = {
        "sample_id", "database_id", "difficulty_band", "template_id", "instruction", "sql", "expected_columns",
        "row_bounds", "operators", "roles", "joins", "mutations",
    }
    errors = [f"missing required field {key}" for key in sorted(required - set(candidate))]
    if candidate.get("sample_id") != expected_id:
        errors.append(f"sample_id must be {expected_id!r}")
    if candidate.get("database_id") != database_id:
        errors.append(f"database_id must be {database_id!r}")
    if candidate.get("difficulty_band") != difficulty_band:
        errors.append(f"difficulty_band must be {difficulty_band!r}")
    instruction = str(candidate.get("instruction") or "").casefold()
    expected_columns = candidate.get("expected_columns") or []
    if isinstance(expected_columns, list):
        missing_aliases = [
            str(column) for column in expected_columns if str(column).casefold() not in instruction
        ]
        if missing_aliases:
            errors.append(
                "instruction must explicitly state exact output aliases: "
                + ", ".join(missing_aliases)
            )
    return errors


def generate_one(
    backend: ModelBackend,
    *,
    sample_id: str,
    database_id: str,
    difficulty_band: str,
    database_root: Path,
    catalog_dir: Path,
    profile: dict[str, Any],
    diversity_memory: list[dict[str, Any]],
    forbidden_template_ids: set[str],
    generation_profile: str,
    max_attempts: int,
    temperature: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog, _ = load_catalog(catalog_dir, database_id)
    digest = catalog_digest(
        catalog,
        max_tables=(
            12
            if generation_profile in {
                "student-frontier-extra-hard",
                "student-frontier-target40",
                "student-frontier-27b",
            }
            else 60
        ),
        focus_seed=sample_id,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": draft_prompt(
                digest,
                profile,
                sample_id,
                difficulty_band,
                diversity_memory,
                generation_profile,
            ),
        },
    ]
    attempts = []
    candidate: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            reply = backend.complete(messages, temperature=temperature, max_tokens=8192)
        except EmptyResponseExhausted as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "empty_response_exhausted",
                    "empty_response_attempts": exc.attempts,
                    "usage": exc.usage,
                    "errors": [str(exc)],
                }
            )
            exc.generation_attempts = attempts  # type: ignore[attr-defined]
            raise
        errors: list[str] = []
        try:
            candidate = parse_json_object(reply.content)
            errors.extend(
                _validate_candidate_shape(candidate, sample_id, database_id, difficulty_band)
            )
            template_id = str(candidate.get("template_id") or "")
            if template_id and template_id in forbidden_template_ids:
                errors.append(
                    f"template_id {template_id!r} was already used in a prior or current batch; "
                    "design a genuinely new semantic template"
                )
            if isinstance(candidate.get("sql"), str):
                candidate["sql"] = candidate["sql"].rstrip().rstrip(";").rstrip()
            gate = (
                complexity_gate(candidate.get("sql", ""), profile, difficulty_band)
                if candidate.get("sql")
                else None
            )
            if gate and not gate["passed"]:
                errors.append("complexity gate failed: " + json.dumps(gate, ensure_ascii=False))
            if gate:
                errors.extend(
                    generation_profile_errors(generation_profile, difficulty_band, gate)
                )
            required_mutations = {
                "student-frontier-target40": 4,
                "student-frontier-27b": 5,
            }.get(generation_profile, 0)
            if len(candidate.get("mutations") or []) < required_mutations:
                errors.append(
                    f"{generation_profile} candidates require at least {required_mutations} "
                    "single-fault mutations"
                )
            if gate:
                candidate["complexity_target"] = difficulty_contract(profile, difficulty_band)
                candidate["advanced_families"] = gate["families"]
            if not errors:
                try:
                    sample, _ = build_sample(candidate, database_root, catalog_dir)
                    errors.extend(sample["validation"]["errors"])
                except Exception as exc:
                    errors.append(f"live validator raised {type(exc).__name__}: {exc}")
        except Exception as exc:
            gate = None
            errors.append(f"response parsing failed: {type(exc).__name__}: {exc}")
        attempts.append(
            {
                "attempt": attempt,
                "errors": errors,
                "complexity": gate,
                "usage": reply.usage,
                "reasoning_summary": reply.reasoning_summary,
                "response": reply.content,
            }
        )
        if not errors and candidate is not None:
            candidate["generation"] = {
                "mode": "model_api_or_local_then_programmatic_validation",
                "model": backend.model,
                "generation_profile": generation_profile,
                "json_response": getattr(backend, "json_response", False),
                "reasoning_effort": getattr(backend, "reasoning_effort", "") or None,
                "empty_response_retries": getattr(backend, "empty_response_retries", None),
                "attempts": attempt,
                "model_catalog_subset": digest.get("catalog_subset"),
                "reference_sql_exposed_to_model": False,
            }
            return candidate, {"status": "passed", "attempts": attempts}
        messages += [
            {"role": "assistant", "content": reply.content},
            {
                "role": "user",
                "content": repair_prompt(
                    digest,
                    profile,
                    difficulty_band,
                    candidate,
                    errors,
                    generation_profile,
                ),
            },
        ]
    raise RuntimeError(json.dumps({"sample_id": sample_id, "attempts": attempts}, ensure_ascii=False))


def generate_many(
    backend: ModelBackend,
    *,
    database_ids: list[str],
    count: int,
    database_root: Path,
    catalog_dir: Path,
    profile_path: Path,
    output_dir: Path,
    max_attempts: int = 3,
    temperature: float = 0.2,
    difficulty_mix: str | dict[str, float] | None = None,
    workers: int = 1,
    slot_retries: int = 3,
    resume: bool = True,
    sample_prefix: str = "sqlite_model",
    diversity_blueprint_paths: list[Path] | None = None,
    generation_profile: str = "standard",
    minimum_structural_score: float | None = None,
    maximum_structural_score: float | None = None,
    retry_same_database: bool = False,
) -> list[dict[str, Any]]:
    """Generate exactly ``count`` validated slots when the backend permits it.

    A generation slot has a stable id and difficulty band.  If a model exhausts
    the draft/repair loop, the slot is retried against the next catalog rather
    than silently shrinking the requested curriculum.  Completed slots are
    checkpointed after every success, so a long run can be resumed without
    paying for them again.
    """

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if slot_retries < 1:
        raise ValueError("slot_retries must be at least 1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_prefix):
        raise ValueError("sample_prefix may contain only letters, numbers, '_' and '-'")
    if workers > 1 and not getattr(backend, "supports_native_tools", False):
        raise ValueError("parallel generation requires a stateless API backend")
    if generation_profile not in GENERATION_PROFILES:
        raise ValueError(f"unknown generation profile: {generation_profile!r}")

    profile = json.loads(profile_path.read_text())
    if (minimum_structural_score is None) != (maximum_structural_score is None):
        raise ValueError("structural score overrides require both minimum and maximum")
    if minimum_structural_score is not None:
        if minimum_structural_score > maximum_structural_score:
            raise ValueError("minimum structural score cannot exceed maximum")
        # The stretch contract is bounded by p75/p90. Override those reference
        # cut points without changing token or advanced-family safeguards.
        profile["distributions"]["structural_score"]["p75"] = minimum_structural_score
        profile["distributions"]["structural_score"]["p90"] = maximum_structural_score
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "attempts"
    raw_dir.mkdir(exist_ok=True)
    blueprint_path = output_dir / "blueprints.jsonl"
    manifest_path = output_dir / "manifest.json"
    prior_manifest: dict[str, Any] = {}
    if resume and manifest_path.exists():
        try:
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior_manifest = {}
    existing: dict[str, dict[str, Any]] = {}
    if resume and blueprint_path.exists():
        for line in blueprint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            if sample_id:
                existing[sample_id] = row
    # Parallel slots receive diversity snapshots at submission time, so two
    # in-flight requests can rarely choose the same template id.  On resume,
    # keep the earliest deterministic slot and regenerate later collisions.
    resume_duplicate_template_slots: list[dict[str, str]] = []
    seen_template_ids: dict[str, str] = {}
    for sample_id in sorted(existing):
        template_id = str(existing[sample_id].get("template_id") or "")
        if not template_id:
            continue
        kept_sample_id = seen_template_ids.get(template_id)
        if kept_sample_id is None:
            seen_template_ids[template_id] = sample_id
            continue
        resume_duplicate_template_slots.append(
            {
                "sample_id": sample_id,
                "template_id": template_id,
                "kept_sample_id": kept_sample_id,
            }
        )
        del existing[sample_id]

    blueprints: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    diversity: list[dict[str, Any]] = []
    for path in diversity_blueprint_paths or []:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            diversity.append(
                {
                    "database_id": row.get("database_id"),
                    "difficulty_band": row.get("difficulty_band"),
                    "template_id": row.get("template_id"),
                    "operators": row.get("operators"),
                    "source": str(path.resolve()),
                }
            )
    diversity.extend([
        {
            "database_id": row.get("database_id"),
            "difficulty_band": row.get("difficulty_band"),
            "template_id": row.get("template_id"),
            "operators": row.get("operators"),
        }
        for row in existing.values()
    ])
    normalized_mix = parse_mix(difficulty_mix)
    stored_schedule = prior_manifest.get("difficulty_schedule")
    schedule_is_compatible = (
        prior_manifest.get("requested_count") == count
        and prior_manifest.get("sample_prefix") == sample_prefix
        and isinstance(stored_schedule, list)
        and len(stored_schedule) == count
        and all(item in BANDS for item in stored_schedule)
    )
    schedule = list(stored_schedule) if schedule_is_compatible else band_schedule(count, normalized_mix)

    def checkpoint() -> None:
        ordered = sorted(blueprints, key=lambda row: str(row["sample_id"]))
        checkpoint_path = blueprint_path.with_suffix(blueprint_path.suffix + ".tmp")
        checkpoint_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered),
            encoding="utf-8",
        )
        checkpoint_path.replace(blueprint_path)

    pending: list[tuple[int, str, str]] = []
    for index, difficulty_band in enumerate(schedule):
        sample_id = f"{sample_prefix}_{index + 1:06d}"
        prior = existing.get(sample_id)
        if prior is not None:
            if prior.get("difficulty_band") != difficulty_band:
                raise ValueError(
                    f"resumed blueprint {sample_id} has band {prior.get('difficulty_band')!r}; "
                    f"expected {difficulty_band!r}"
                )
            blueprints.append(prior)
        else:
            pending.append((index, sample_id, difficulty_band))
    checkpoint()

    def generate_slot(
        spec: tuple[int, str, str], diversity_snapshot: list[dict[str, Any]]
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        index, sample_id, difficulty_band = spec
        cycles: list[dict[str, Any]] = []
        for cycle in range(slot_retries):
            database_id = database_ids[index % len(database_ids)] if retry_same_database else (
                database_ids[(index + cycle) % len(database_ids)]
            )
            try:
                blueprint, trace = generate_one(
                    backend,
                    sample_id=sample_id,
                    database_id=database_id,
                    difficulty_band=difficulty_band,
                    database_root=database_root,
                    catalog_dir=catalog_dir,
                    profile=profile,
                    diversity_memory=diversity_snapshot,
                    forbidden_template_ids={
                        str(item.get("template_id"))
                        for item in diversity_snapshot
                        if item.get("template_id")
                    },
                    generation_profile=generation_profile,
                    max_attempts=max_attempts,
                    temperature=temperature,
                )
                cycles.append({"cycle": cycle + 1, "database_id": database_id, **trace})
                return blueprint, {
                    "status": "passed",
                    "sample_id": sample_id,
                    "difficulty_band": difficulty_band,
                    "cycles": cycles,
                }
            except EmptyResponseExhausted as exc:
                cycles.append(
                    {
                        "cycle": cycle + 1,
                        "database_id": database_id,
                        "status": "empty_response_exhausted",
                        "empty_response_attempts": exc.attempts,
                        "usage": exc.usage,
                        "attempts": getattr(exc, "generation_attempts", []),
                    }
                )
            except RuntimeError as exc:
                detail: Any
                try:
                    detail = json.loads(str(exc))
                except ValueError:
                    detail = {"error": f"{type(exc).__name__}: {exc}"}
                cycles.append(
                    {
                        "cycle": cycle + 1,
                        "database_id": database_id,
                        "status": "draft_repair_exhausted",
                        "detail": detail,
                    }
                )
        return None, {
            "status": "slot_exhausted",
            "sample_id": sample_id,
            "difficulty_band": difficulty_band,
            "slot_retries": slot_retries,
            "cycles": cycles,
        }

    def record(
        blueprint: dict[str, Any] | None, trace: dict[str, Any]
    ) -> None:
        sample_id = str(trace["sample_id"])
        (raw_dir / f"{sample_id}.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if blueprint is None:
            skipped.append(
                {
                    "sample_id": sample_id,
                    "difficulty_band": trace.get("difficulty_band"),
                    "status": trace.get("status"),
                    "trace": str((raw_dir / f"{sample_id}.json").resolve()),
                }
            )
            print(
                f"[{sample_id}] exhausted ({len(blueprints)}/{count} validated)",
                flush=True,
            )
            return
        blueprints.append(blueprint)
        diversity.append(
            {
                "database_id": blueprint["database_id"],
                "difficulty_band": blueprint["difficulty_band"],
                "template_id": blueprint["template_id"],
                "operators": blueprint["operators"],
            }
        )
        checkpoint()
        print(
            f"[{sample_id}] passed on {blueprint['database_id']} "
            f"({len(blueprints)}/{count} validated)",
            flush=True,
        )

    if workers == 1:
        for spec in pending:
            record(*generate_slot(spec, list(diversity)))
    else:
        spec_iter = iter(pending)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict[
                concurrent.futures.Future[tuple[dict[str, Any] | None, dict[str, Any]]],
                tuple[int, str, str],
            ] = {}
            while True:
                while len(futures) < workers:
                    try:
                        spec = next(spec_iter)
                    except StopIteration:
                        break
                    futures[pool.submit(generate_slot, spec, list(diversity))] = spec
                if not futures:
                    break
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    futures.pop(future)
                    record(*future.result())

    blueprints.sort(key=lambda row: str(row["sample_id"]))
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_count": count,
        "count": len(blueprints),
        "skipped_empty_response": sum(
            any(cycle.get("status") == "empty_response_exhausted" for cycle in trace.get("cycles", []))
            for trace in (
                json.loads(Path(item["trace"]).read_text(encoding="utf-8")) for item in skipped
            )
        ),
        "skipped_slots": len(skipped),
        "skipped": skipped,
        "backend": type(backend).__name__,
        "model": backend.model,
        "empty_response_retries": getattr(backend, "empty_response_retries", None),
        "json_response": getattr(backend, "json_response", False),
        "reasoning_effort": getattr(backend, "reasoning_effort", "") or None,
        "database_ids": database_ids,
        "difficulty_mix": normalized_mix,
        "difficulty_schedule": schedule,
        "workers": workers,
        "slot_retries": slot_retries,
        "generation_runs": list(prior_manifest.get("generation_runs") or [])
        + [
            {
                "started_with_completed_slots": len(existing),
                "pending_slots": len(pending),
                "finished_with_completed_slots": len(blueprints),
                "workers": workers,
                "slot_retries": slot_retries,
                "max_attempts": max_attempts,
                "model": backend.model,
            }
        ],
        "resume": resume,
        "resume_regenerated_duplicate_template_slots": resume_duplicate_template_slots,
        "sample_prefix": sample_prefix,
        "generation_profile": generation_profile,
        "minimum_structural_score_override": minimum_structural_score,
        "maximum_structural_score_override": maximum_structural_score,
        "retry_same_database": retry_same_database,
        "model_catalog_policy": (
            "deterministic connected subgraph of at most 12 tables; full catalog retained by verifier"
            if generation_profile in {
                "student-frontier-extra-hard",
                "student-frontier-target40",
                "student-frontier-27b",
            }
            else "catalog digest of at most 60 tables; full catalog retained by verifier"
        ),
        "diversity_blueprint_paths": [
            str(path.resolve()) for path in diversity_blueprint_paths or []
        ],
        "profile_path": str(profile_path.resolve()),
        "reference_sql_exposed_to_model": False,
        "promotion_policy": (
            "catalog + live execution + mutations + bounded static band; then teacher/student rollout calibration"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return blueprints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_arguments(parser)
    parser.add_argument("--databases", required=True, help="comma-separated Spider 2.0 SQLite database IDs")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--database-root", type=Path, default=DEFAULT_DATABASE_ROOT)
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--profile", type=Path, default=DEFAULT_COMPLEXITY_DIR / "profile.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--slot-retries",
        type=int,
        default=3,
        help="fresh draft/repair cycles for one required curriculum slot",
    )
    parser.add_argument("--sample-prefix", default="sqlite_model")
    parser.add_argument("--minimum-structural-score", type=float)
    parser.add_argument("--maximum-structural-score", type=float)
    parser.add_argument(
        "--retry-same-database",
        action="store_true",
        help="keep each required slot on its assigned database across retries",
    )
    parser.add_argument(
        "--generation-profile",
        choices=GENERATION_PROFILES,
        default="standard",
        help="optional semantic-hardness guidance in addition to the bounded structural band",
    )
    parser.add_argument(
        "--diversity-blueprints",
        type=Path,
        action="append",
        default=[],
        help="prior blueprint JSONL whose template IDs must not be reused (repeatable)",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--difficulty-mix",
        default=",".join(f"{name}={weight}" for name, weight in DEFAULT_MIX.items()),
        help="foundation/core/growth/stretch weights, e.g. foundation=.2,core=.45,growth=.25,stretch=.1",
    )
    args = parser.parse_args()
    database_ids = [value.strip() for value in args.databases.split(",") if value.strip()]
    if not database_ids:
        parser.error("--databases must contain at least one ID")
    backend = backend_from_args(args)
    generated = generate_many(
        backend,
        database_ids=database_ids,
        count=args.count,
        database_root=args.database_root,
        catalog_dir=args.catalog_dir,
        profile_path=args.profile,
        output_dir=args.output_dir,
        max_attempts=args.max_attempts,
        temperature=args.temperature,
        difficulty_mix=args.difficulty_mix,
        workers=args.workers,
        slot_retries=args.slot_retries,
        resume=not args.no_resume,
        sample_prefix=args.sample_prefix,
        diversity_blueprint_paths=args.diversity_blueprints,
        generation_profile=args.generation_profile,
        minimum_structural_score=args.minimum_structural_score,
        maximum_structural_score=args.maximum_structural_score,
        retry_same_database=args.retry_same_database,
    )
    print(f"generated and validated {len(generated)} blueprints in {args.output_dir}")
    return 0 if len(generated) == args.count else 2


if __name__ == "__main__":
    raise SystemExit(main())
