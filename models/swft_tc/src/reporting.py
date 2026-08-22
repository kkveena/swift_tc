"""Executive reporting: KPIs, score distribution, scenario mix, HITL sensitivity.

All reusable reporting logic lives here so the notebook stays an orchestration
and presentation layer.

**Denominators matter and are never mixed.** Three different populations are in
play and each is labelled wherever it is used:

``records``
    input rows.
``address-group instances``
    rows x enabled groups. Split into *empty* (short-circuited before any model
    call) and *non-empty*.
``unique addresses``
    distinct cleaned addresses actually sent for extraction.

The composite-score distribution and the HITL sensitivity table both use
**non-empty address-group instances**, because that is where review workload
actually lands. Empty instances are reported separately and never pad the
histogram.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "BELOW_FIRST_BAND_LABEL",
    "AnalysisResult",
    "ExecutiveReport",
    "OPERATOR_METADATA_KEYS",
    "auto_accept_mask",
    "data_derived_strings",
    "forced_review_mask",
    "band_labels",
    "build_cross_entropy_summary",
    "build_executive_summary",
    "build_hitl_state_distribution",
    "build_kpi_table",
    "build_scenario_distribution",
    "build_score_distribution",
    "build_error_capture_gain",
    "build_error_capture_lift",
    "build_precision_coverage",
    "build_threshold_sensitivity",
    "build_threshold_tradeoff",
    "classify_score",
    "threshold_grid",
    "render_error_capture_gain_chart",
    "render_error_capture_lift_chart",
    "render_precision_coverage_chart",
    "render_score_histogram",
    "render_threshold_tradeoff_chart",
    "write_reports",
]

logger = logging.getLogger(__name__)

#: Label for scores below the first configured band edge.
BELOW_FIRST_BAND_LABEL = "< {edge:.2f}"

# Chart palette: categorical slots 1 and 2 of the validated reference palette.
# Two marks only, so the pair is an adjacent pair and clears every gate
# (CVD ΔE 24.7, normal-vision ΔE 33.6, both ≥ 3:1 on the light surface).
_COLOR_HITL = "#2a78d6"
_COLOR_AUTO = "#eb6834"
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#dcdbd6"


#: Executive-summary fields whose text the *operator* chooses — model IDs, prompt
#: and reference version labels, timestamps, fixed prose. None is derived from
#: input data, so a privacy scan must skip them. Without this, a prompt version
#: named after a test case ("v5-greenwich-json-regression") reads as an address
#: leak when the address it was named for is nowhere near the file.
OPERATOR_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "run_timestamp",
        "mode",
        "model",
        "prompt_version",
        "reference_data_version",
        "town_country_reference_version",
        "threshold_basis",
    }
)


def data_derived_strings(summary: Mapping[str, Any]) -> list[str]:
    """String values in the summary that are *not* operator-chosen metadata.

    This is what a privacy check should scan. The executive summary is built
    from aggregate counts plus run metadata, so this list should be empty — the
    scan is belt and braces against a future field carrying data through.
    """
    return [
        value
        for key, value in summary.items()
        if isinstance(value, str) and key not in OPERATOR_METADATA_KEYS
    ]


def band_labels(edges: Sequence[float]) -> tuple[str, ...]:
    """Ordered band labels for the configured edges.

    ``(0.60, 0.65, …, 0.95)`` yields ``("< 0.60", "0.60 - <0.65", …,
    "0.95 - 1.00")``. Every band is half-open ``[lower, upper)`` except the last,
    which closes at 1.00 so a perfect score is never dropped.
    """
    if not edges:
        raise ValueError("at least one band edge is required")
    labels = [BELOW_FIRST_BAND_LABEL.format(edge=edges[0])]
    for index, lower in enumerate(edges):
        if index + 1 < len(edges):
            labels.append(f"{lower:.2f} - <{edges[index + 1]:.2f}")
        else:
            labels.append(f"{lower:.2f} - 1.00")
    return tuple(labels)


def classify_score(score: float, edges: Sequence[float]) -> str:
    """Place one composite score into its band.

    Boundaries belong to the band they open: ``0.60`` is in ``0.60 - <0.65``,
    ``0.65`` is in ``0.65 - <0.70``, and ``1.00`` closes ``0.95 - 1.00``.
    """
    labels = band_labels(edges)
    if score < edges[0]:
        return labels[0]
    for index, lower in enumerate(edges):
        upper = edges[index + 1] if index + 1 < len(edges) else None
        if upper is None or score < upper:
            return labels[index + 1]
    return labels[-1]  # pragma: no cover - unreachable, the last band is open-topped


def forced_review_mask(instances: pd.DataFrame) -> pd.Series:
    """Which observations stay with a human **whatever the threshold is**.

    The Composite Weighted Score is not the only control. Review is also
    required by a processing error, a manual override, unresolved final country
    ambiguity, or a reference conflict — and none of those care what number the
    cutoff is set to. A hypothetical-threshold sweep that ignores them reports a
    reference-conflicted case as an auto-accept the moment the threshold drops
    below its score, which is not what the pipeline would actually do.

    This is the single definition every reporting path uses — the sensitivity
    table, the KPI table, the executive summary and the threshold curves — so
    they can never drift apart or implement three partial versions of the rule.
    It is *reporting* only: :func:`models.swft_tc.src.scoring.determine_hitl_decision`
    remains the sole authority on the real routing decision, and nothing here
    changes it.

    ``hitl_forced_review`` — written by that engine — is the authority when the
    frame carries it. The individual signals are OR-ed in as well so a frame
    assembled without it (an older artifact, a hand-built test fixture) still
    reports the override rather than silently losing it. Erring toward "forced"
    is the safe direction: it never turns a review case into an auto-accept.
    """
    frame = _non_empty(instances)
    if frame.empty:
        return pd.Series(dtype=bool)

    forced = pd.Series(False, index=frame.index)
    if "hitl_forced_review" in frame.columns:
        forced = forced | frame["hitl_forced_review"].fillna(False).astype(bool)
    for column in ("country_ambiguous", "extraction_error", "manual_override"):
        if column in frame.columns:
            forced = forced | frame[column].fillna(False).astype(bool)
    if "reference_status" in frame.columns:
        from .scoring import REFERENCE_CONFLICT

        forced = forced | frame["reference_status"].eq(REFERENCE_CONFLICT)
    return forced


def auto_accept_mask(instances: pd.DataFrame, threshold: float) -> pd.Series:
    """Observations that would be auto-accept *candidates* at ``threshold``.

    A candidate is an observation that both clears the cutoff and carries no
    forced-review control. Candidate, not accepted — the name is the reminder
    that a human policy still decides what happens to them.
    """
    frame = _non_empty(instances)
    if frame.empty:
        return pd.Series(dtype=bool)
    meets = frame["composite_weighted_score"].astype(float) >= float(threshold)
    return meets & ~forced_review_mask(frame)


def _non_empty(instances: pd.DataFrame) -> pd.DataFrame:
    """`RunResult.instances` already contains only non-empty instances."""
    if instances is None or instances.empty:
        return pd.DataFrame(
            columns=[
                "record_id", "group_id", "composite_weighted_score", "scenario",
                "country_ambiguous", "extraction_error", "reference_status",
                "needs_hitl",
            ]
        )
    return instances


def build_score_distribution(
    instances: pd.DataFrame, edges: Sequence[float]
) -> pd.DataFrame:
    """Composite-score distribution over non-empty address-group instances.

    Every band appears even at zero count, so the shape of the table is stable
    across runs and safe to diff.
    """
    frame = _non_empty(instances)
    labels = band_labels(edges)
    total = len(frame)

    counts = {label: 0 for label in labels}
    for score in frame["composite_weighted_score"]:
        counts[classify_score(float(score), edges)] += 1

    rows: list[dict[str, Any]] = []
    cumulative = 0
    for label in labels:
        count = counts[label]
        cumulative += count
        rows.append(
            {
                "score_band": label,
                "observation_count": count,
                "percent_of_non_empty": _percent(count, total),
                "cumulative_count": cumulative,
                "cumulative_percent": _percent(cumulative, total),
            }
        )
    return pd.DataFrame(rows)


def build_scenario_distribution(instances: pd.DataFrame) -> pd.DataFrame:
    """Deterministic policy-scenario mix over non-empty instances."""
    frame = _non_empty(instances)
    total = len(frame)
    if total == 0:
        return pd.DataFrame(
            columns=["scenario", "observation_count", "percent_of_non_empty"]
        )

    counts = frame["scenario"].value_counts()
    rows = [
        {
            "scenario": scenario,
            "observation_count": int(count),
            "percent_of_non_empty": _percent(int(count), total),
        }
        for scenario, count in counts.items()
    ]
    # Most frequent first, ties broken by name so the table is reproducible.
    rows.sort(key=lambda row: (-row["observation_count"], row["scenario"]))
    return pd.DataFrame(rows)


def build_threshold_sensitivity(
    instances: pd.DataFrame, thresholds: Iterable[float]
) -> pd.DataFrame:
    """How routing volume responds to the HITL threshold.

    A score at or above the threshold is an auto-accept *candidate* — candidate,
    not accepted, because a processing error, a manual override, unresolved
    country ambiguity, and a model/reference conflict force review whatever the
    number says. :func:`forced_review_mask` is the single definition of that
    override, shared with the KPI table, the executive summary and the threshold
    curves. Forced cases are also reported in their own columns so the override
    is visible rather than implied.
    """
    frame = _non_empty(instances)
    total = len(frame)
    forced = forced_review_mask(frame)
    forced_count = int(forced.sum()) if total else 0

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        if total == 0:
            rows.append(
                {
                    "threshold": round(float(threshold), 4),
                    "auto_accept_candidate_count": 0,
                    "auto_accept_candidate_percent": 0.0,
                    "hitl_count": 0,
                    "hitl_percent": 0.0,
                    "forced_review_count": 0,
                    "low_score_hitl_count": 0,
                    "ambiguous_forced_hitl_count": 0,
                    "error_forced_hitl_count": 0,
                }
            )
            continue

        meets = frame["composite_weighted_score"].astype(float) >= float(threshold)
        auto_accept = meets & ~forced

        auto_count = int(auto_accept.sum())
        hitl_count = total - auto_count
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "auto_accept_candidate_count": auto_count,
                "auto_accept_candidate_percent": _percent(auto_count, total),
                "hitl_count": hitl_count,
                "hitl_percent": _percent(hitl_count, total),
                # Forced cases are threshold-invariant; the low-score column is
                # the part of the workload the cutoff actually moves.
                "forced_review_count": forced_count,
                "low_score_hitl_count": int((~meets & ~forced).sum()),
                "ambiguous_forced_hitl_count": int(frame["country_ambiguous"].sum()),
                "error_forced_hitl_count": int(frame["extraction_error"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_cross_entropy_summary(instances: pd.DataFrame) -> pd.DataFrame:
    """Calibration summary over *grounded* observations only.

    Cross-entropy answers a different question from the Composite Weighted
    Score, and in the opposite direction:

    * Composite Weighted Score — operational routing — **higher is better**.
    * Cross-entropy — did confidence match reality — **lower is better**.

    Only observations with the relevant ground truth available enter the
    denominator. An instance the reference could not label is *excluded*, not
    counted as a miss: a coverage gap is not a model error, and letting it
    inflate the loss would make the metric say something untrue.
    """
    frame = _non_empty(instances)
    total = len(frame)

    if total == 0 or "cross_entropy" not in frame.columns:
        return pd.DataFrame(
            [
                {"metric": "Non-empty instances", "value": total,
                 "denominator": "instances"},
                {"metric": "Grounded observations", "value": 0,
                 "denominator": "instances with any ground truth"},
            ]
        )

    losses = pd.to_numeric(frame["cross_entropy"], errors="coerce").dropna()
    town = _correctness(frame, "town_exists_ok", "town_grounded")
    country = _correctness(frame, "country_exists_ok", "country_grounded")

    rows: list[dict[str, Any]] = [
        {"metric": "Non-empty instances", "value": total, "denominator": "instances"},
        {"metric": "Grounded observations", "value": int(len(losses)),
         "denominator": "instances with any ground truth"},
        {"metric": "Ungrounded (excluded from loss)", "value": int(total - len(losses)),
         "denominator": "instances"},
        {"metric": "Mean cross-entropy", "value": _stat(losses.mean()),
         "denominator": "grounded observations — lower is better"},
        {"metric": "Median cross-entropy", "value": _stat(losses.median()),
         "denominator": "grounded observations — lower is better"},
        {"metric": "P95 cross-entropy",
         "value": _stat(losses.quantile(0.95)) if len(losses) else None,
         "denominator": "grounded observations — lower is better"},
        {"metric": "Town correctness rate", "value": town[0],
         "denominator": f"{town[1]} town-grounded instances"},
        {"metric": "Country correctness rate", "value": country[0],
         "denominator": f"{country[1]} country-grounded instances"},
    ]
    return pd.DataFrame(
        {
            "metric": [row["metric"] for row in rows],
            "value": pd.Series([row["value"] for row in rows], dtype=object),
            "denominator": [row["denominator"] for row in rows],
        }
    )


def _correctness(
    frame: pd.DataFrame, column: str, grounded_column: str
) -> tuple[float | None, int]:
    """(percent correct, grounded count) for one correctness-label column.

    ``town_exists_ok`` / ``country_exists_ok`` are plain booleans now (unknown
    collapses to False), so the paired ``*_grounded`` flag — not nullability —
    decides the denominator. A coverage gap is still excluded rather than
    counted as incorrect.
    """
    if column not in frame.columns or grounded_column not in frame.columns:
        return None, 0
    grounded = frame.loc[frame[grounded_column].astype(bool), column]
    if grounded.empty:
        return None, 0
    correct = int(grounded.astype(bool).sum())
    return round(100.0 * correct / len(grounded), 2), int(len(grounded))


def _stat(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    return None if pd.isna(numeric) else round(numeric, 6)


def build_hitl_state_distribution(instances: pd.DataFrame) -> pd.DataFrame:
    """Primary HITL routing state over non-empty address-group instances.

    Every state appears even at zero count, so the table shape is stable across
    runs and safe to diff. Null-skipped groups are excluded from the
    denominator: they were never evaluated, so they are not a routing outcome.

    The state is the *primary* one after precedence — a case that is both
    reference-conflicted and below threshold is counted once, under
    ``HITL_REFERENCE_CONFLICT``. All contributing reasons stay in the detailed
    JSON.
    """
    from .scoring import HITL_STATE_PRECEDENCE

    frame = _non_empty(instances)
    total = len(frame)

    # Report in precedence order rather than by frequency: this table is read as
    # a control hierarchy, not a leaderboard.
    ordered_states = list(HITL_STATE_PRECEDENCE)
    counts = {state: 0 for state in ordered_states}

    if total and "hitl_state" in frame.columns:
        for state, count in frame["hitl_state"].value_counts().items():
            key = str(state)
            counts[key] = counts.get(key, 0) + int(count)
            if key not in ordered_states:
                # An unexpected state would otherwise vanish from the report.
                ordered_states.append(key)

    rows = [
        {
            "HITL_state": state,
            "observation_count": counts.get(state, 0),
            "percent_of_non_empty": _percent(counts.get(state, 0), total),
        }
        for state in ordered_states
    ]
    return pd.DataFrame(
        rows, columns=["HITL_state", "observation_count", "percent_of_non_empty"]
    )


def build_kpi_table(
    metrics: Mapping[str, Any],
    instances: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Compact executive KPI table with explicit denominators."""
    frame = _non_empty(instances)
    total_instances = len(frame)
    auto = int(auto_accept_mask(frame, threshold).sum()) if total_instances else 0
    forced_count = int(forced_review_mask(frame).sum()) if total_instances else 0
    hitl = total_instances - auto

    shape = metrics.get("shape", {})
    pass1 = metrics.get("pass1", {})
    efficiency = metrics.get("efficiency", {})
    outcomes = metrics.get("outcomes", {})

    rows = [
        ("Input records", shape.get("input_rows", 0), "records"),
        ("Enabled address groups", shape.get("groups_enabled", 0), "groups"),
        ("Address-group instances", pass1.get("group_instances", 0), "instances"),
        ("  empty, skipped before any call", pass1.get("empty_instances_skipped", 0), "instances"),
        ("  non-empty", pass1.get("non_empty_instances", 0), "instances"),
        ("Unique addresses eligible", efficiency.get("unique_addresses", 0), "unique addresses"),
        ("Backend model calls", efficiency.get("backend_calls", 0), "calls"),
        ("Cache hits", efficiency.get("cache_hits", 0), "unique addresses"),
        ("Extraction errors", outcomes.get("extraction_errors", 0), "unique addresses"),
        ("Ambiguous-country instances", outcomes.get("ambiguous_country_instances", 0), "instances"),
        ("Reference conflicts", outcomes.get("reference_conflict_instances", 0), "instances"),
        ("HITL instances", hitl, "instances"),
        ("HITL %", _percent(hitl, total_instances), "% of non-empty instances"),
        ("  forced by a control, whatever the threshold", forced_count, "instances"),
        ("  below threshold only", hitl - forced_count, "instances"),
        ("Auto-accept candidates", int(auto), "instances"),
        ("Auto-accept candidate %", _percent(int(auto), total_instances), "% of non-empty instances"),
        ("Configured HITL threshold", round(float(threshold), 4), "composite score"),
    ]
    # `value` mixes counts and percentages; an object column keeps 8 as 8 rather
    # than letting pandas widen every count to 8.00.
    return pd.DataFrame(
        {
            "metric": [row[0] for row in rows],
            "value": pd.Series([row[1] for row in rows], dtype=object),
            "denominator": [row[2] for row in rows],
        }
    )


def build_executive_summary(
    metrics: Mapping[str, Any],
    instances: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, Any]:
    """The JSON executive artifact. Carries no raw address, by construction."""
    frame = _non_empty(instances)
    total_instances = len(frame)
    auto = int(auto_accept_mask(frame, threshold).sum()) if total_instances else 0
    forced_count = int(forced_review_mask(frame).sum()) if total_instances else 0
    hitl = total_instances - auto

    run = metrics.get("run", {})
    reference = metrics.get("reference_data", {})
    town_country = reference.get("town_country", {})

    return {
        "run_timestamp": run.get(
            "started_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")
        ),
        "mode": run.get("mode"),
        "model": run.get("model"),
        "prompt_version": run.get("prompt_version"),
        "reference_data_version": reference.get("context_version"),
        "town_country_reference_version": town_country.get("source_version"),
        "town_country_approved_for_production": town_country.get(
            "approved_for_production", False
        ),
        "hitl_threshold": round(float(threshold), 4),
        "records": metrics.get("shape", {}).get("input_rows", 0),
        "address_group_instances": metrics.get("pass1", {}).get("group_instances", 0),
        "empty_instances_skipped": metrics.get("pass1", {}).get(
            "empty_instances_skipped", 0
        ),
        "non_empty_address_instances": total_instances,
        "unique_addresses": metrics.get("efficiency", {}).get("unique_addresses", 0),
        "backend_calls": metrics.get("efficiency", {}).get("backend_calls", 0),
        "auto_accept_candidates": auto,
        "auto_accept_candidate_percent": _percent(auto, total_instances),
        "hitl_instances": hitl,
        "hitl_percent": _percent(hitl, total_instances),
        "forced_review_instances": forced_count,
        "low_score_only_hitl_instances": hitl - forced_count,
        "ambiguous_country_instances": metrics.get("outcomes", {}).get(
            "ambiguous_country_instances", 0
        ),
        "reference_conflict_instances": metrics.get("outcomes", {}).get(
            "reference_conflict_instances", 0
        ),
        "extraction_errors": metrics.get("outcomes", {}).get("extraction_errors", 0),
        "threshold_basis": (
            "Provisional routing policy, not calibrated accuracy. Recalibrate "
            "against labeled validation data before production use."
        ),
    }


# --------------------------------------------------------------------------
# Threshold analytics
#
# Three different questions, deliberately kept apart because they answer
# different things and only one of them is about model quality:
#
#   threshold trade-off   how much review work does a cutoff create?
#   precision / coverage  how good is what we would auto-accept, and how much
#                         of the labelled population does that cover?
#   error-capture gain    if reviewers work lowest-score-first, how quickly do
#                         they reach the cases that are actually wrong?
#
# All three are decision *support*. None of them sets `scoring.hitl_threshold`;
# that stays configuration, approved separately against business risk appetite.
# --------------------------------------------------------------------------


def threshold_grid(step: float) -> tuple[float, ...]:
    """Evaluation points from 0.00 to 1.00 inclusive at ``step``.

    Values are rounded to the precision the step implies, so a 0.01 grid gives
    exactly 0.00, 0.01, … 1.00 rather than binary-float noise that would make
    the CSV impossible to diff.
    """
    if not 0.0 < step <= 1.0:
        raise ValueError(f"threshold step {step} must lie in (0, 1]")
    digits = max(0, len(f"{step:.10f}".rstrip("0").split(".")[1]))
    count = int(round(1.0 / step))
    grid = [round(min(1.0, index * step), digits) for index in range(count + 1)]
    if grid[-1] != 1.0:
        grid.append(1.0)
    return tuple(dict.fromkeys(grid))


def build_threshold_tradeoff(
    instances: pd.DataFrame, *, step: float = 0.01
) -> pd.DataFrame:
    """Auto-accept vs review workload across a fine threshold grid.

    This is the operational workload curve: at each candidate cutoff, how much
    of the non-empty population would be an auto-accept candidate and how much
    would reach a human. It says nothing about whether those auto-accepts are
    *correct* — that is what :func:`build_precision_coverage` is for.

    Forced-review cases are held out of the auto-accept side at every threshold
    via :func:`forced_review_mask`, so lowering the cutoff can never turn a
    reference-conflicted or ambiguous case into a candidate. ``forced_review_count``
    is therefore flat down the table by construction, and ``low_score_hitl_count``
    is the only part the cutoff actually moves.
    """
    frame = _non_empty(instances)
    total = len(frame)
    grid = threshold_grid(step)

    if total == 0:
        return pd.DataFrame(
            [
                {
                    "threshold": threshold,
                    "auto_accept_candidate_count": 0,
                    "auto_accept_candidate_percent": 0.0,
                    "hitl_count": 0,
                    "hitl_percent": 0.0,
                    "forced_review_count": 0,
                    "low_score_hitl_count": 0,
                }
                for threshold in grid
            ]
        )

    forced = forced_review_mask(frame)
    forced_count = int(forced.sum())
    scores = frame["composite_weighted_score"].astype(float)

    rows: list[dict[str, Any]] = []
    for threshold in grid:
        meets = scores >= threshold
        auto_count = int((meets & ~forced).sum())
        hitl_count = total - auto_count
        rows.append(
            {
                "threshold": threshold,
                "auto_accept_candidate_count": auto_count,
                "auto_accept_candidate_percent": _percent(auto_count, total),
                "hitl_count": hitl_count,
                "hitl_percent": _percent(hitl_count, total),
                "forced_review_count": forced_count,
                "low_score_hitl_count": int((~meets & ~forced).sum()),
            }
        )
    return pd.DataFrame(rows)


def _fully_grounded(instances: pd.DataFrame) -> pd.DataFrame:
    """Observations carrying independent ground truth for **both** Town and Country.

    Precision and error capture are only meaningful where both halves of the
    prediction can be checked. A partially grounded observation is excluded
    rather than assumed correct or incorrect — a coverage gap is not evidence.
    """
    frame = _non_empty(instances)
    required = {"town_grounded", "country_grounded", "town_exists_ok", "country_exists_ok"}
    if frame.empty or not required.issubset(frame.columns):
        return frame.iloc[0:0]
    grounded = frame["town_grounded"].fillna(False).astype(bool) & frame[
        "country_grounded"
    ].fillna(False).astype(bool)
    return frame.loc[grounded]


def _group_correct(frame: pd.DataFrame) -> pd.Series:
    """A fully grounded observation is correct only if both halves are correct."""
    if frame.empty:
        return pd.Series(dtype=bool)
    return frame["town_exists_ok"].fillna(False).astype(bool) & frame[
        "country_exists_ok"
    ].fillna(False).astype(bool)


@dataclass(frozen=True)
class AnalysisResult:
    """One threshold-analytics artifact plus whether it could be produced.

    ``available`` False is a first-class outcome, not a failure: with too few
    labelled observations the honest answer is to say so rather than draw a
    chart the data cannot support. ``reason`` carries that sentence for the
    report and the notebook to print verbatim.
    """

    available: bool
    reason: str
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict[str, Any] = field(default_factory=dict)


def build_precision_coverage(
    instances: pd.DataFrame,
    *,
    step: float = 0.01,
    min_grounded: int = 1,
) -> AnalysisResult:
    """Auto-accept precision against grounded coverage, across the threshold grid.

    Restricted to *fully grounded* observations — both Town and Country
    independently labelled — because precision over a population you cannot
    check is not precision. Ungrounded observations are excluded from both the
    numerator and the denominator.

    ``auto_accept_precision`` is the share of grounded auto-accept candidates
    that are actually correct; ``grounded_coverage`` is the share of the fully
    grounded population those candidates represent. Raising the threshold
    normally trades coverage away for precision, and this is the curve a
    production cutoff should be argued from — not the workload curve.

    Forced-review cases can never be auto-accept candidates here either.
    """
    grounded = _fully_grounded(instances)
    total_grounded = len(grounded)
    if total_grounded < max(1, min_grounded):
        return AnalysisResult(
            available=False,
            reason=(
                "Precision/Coverage analysis not generated: insufficient fully "
                f"grounded observations ({total_grounded} available, "
                f"{max(1, min_grounded)} required). Both Town and Country must "
                "carry independent ground truth for an observation to count."
            ),
            summary={"fully_grounded_observations": total_grounded},
        )

    forced = forced_review_mask(grounded)
    scores = grounded["composite_weighted_score"].astype(float)
    correct = _group_correct(grounded)

    rows: list[dict[str, Any]] = []
    for threshold in threshold_grid(step):
        auto = (scores >= threshold) & ~forced
        auto_count = int(auto.sum())
        correct_count = int((auto & correct).sum())
        rows.append(
            {
                "threshold": threshold,
                "grounded_auto_accept_count": auto_count,
                "grounded_correct_auto_accept_count": correct_count,
                "auto_accept_precision": (
                    round(100.0 * correct_count / auto_count, 2) if auto_count else None
                ),
                "grounded_coverage": _percent(auto_count, total_grounded),
                "grounded_hitl_count": total_grounded - auto_count,
            }
        )

    table = pd.DataFrame(rows)
    return AnalysisResult(
        available=True,
        reason="",
        table=table,
        summary={
            "fully_grounded_observations": total_grounded,
            "fully_grounded_correct": int(correct.sum()),
            "fully_grounded_errors": int((~correct).sum()),
            "forced_review_grounded": int(forced.sum()),
        },
    )


def build_error_capture_gain(
    instances: pd.DataFrame, *, min_errors: int = 1
) -> AnalysisResult:
    """Cumulative share of real errors captured as review depth increases.

    The business question is not "how accurate is the model" but "if a reviewer
    works the lowest-confidence cases first, how quickly do they reach the ones
    that are actually wrong?" Fully grounded observations are sorted by
    Composite Weighted Score ascending — review order — and the curve reports,
    for each depth, what fraction of the known errors has been seen.

    A diagonal is what random review would achieve; anything above it is the
    value the ranking adds. With no labelled errors there is no curve to draw
    and none is invented.
    """
    grounded = _fully_grounded(instances)
    total_grounded = len(grounded)
    if total_grounded == 0:
        return AnalysisResult(
            available=False,
            reason=(
                "Gain/Lift not available because the labelled population "
                "contains insufficient grounded errors: no fully grounded "
                "observations (both Town and Country labelled) exist."
            ),
            summary={"fully_grounded_observations": 0, "grounded_errors": 0},
        )

    is_error = ~_group_correct(grounded)
    total_errors = int(is_error.sum())
    if total_errors < max(1, min_errors):
        return AnalysisResult(
            available=False,
            reason=(
                "Gain/Lift not available because the labelled population "
                f"contains insufficient grounded errors ({total_errors} in "
                f"{total_grounded} fully grounded observations, "
                f"{max(1, min_errors)} required)."
            ),
            summary={
                "fully_grounded_observations": total_grounded,
                "grounded_errors": total_errors,
            },
        )

    # Ascending score = review order. record_id/group_id break ties so the same
    # run always produces the same curve.
    order = grounded.assign(_is_error=is_error).sort_values(
        by=["composite_weighted_score", "record_id", "group_id"],
        kind="mergesort",
    )
    captured = 0
    rows: list[dict[str, Any]] = [
        {
            "reviewed_population_count": 0,
            "reviewed_population_percent": 0.0,
            "errors_captured_count": 0,
            "errors_captured_percent": 0.0,
            "random_baseline_percent": 0.0,
            "composite_weighted_score": None,
        }
    ]
    for position, (_, row) in enumerate(order.iterrows(), start=1):
        captured += int(bool(row["_is_error"]))
        reviewed_percent = _percent(position, total_grounded)
        rows.append(
            {
                "reviewed_population_count": position,
                "reviewed_population_percent": reviewed_percent,
                "errors_captured_count": captured,
                "errors_captured_percent": _percent(captured, total_errors),
                # Random review captures errors at the population rate.
                "random_baseline_percent": reviewed_percent,
                "composite_weighted_score": round(
                    float(row["composite_weighted_score"]), 6
                ),
            }
        )

    return AnalysisResult(
        available=True,
        reason="",
        table=pd.DataFrame(rows),
        summary={
            "fully_grounded_observations": total_grounded,
            "grounded_errors": total_errors,
            "grounded_error_rate_percent": _percent(total_errors, total_grounded),
        },
    )


def build_error_capture_lift(gain: AnalysisResult) -> AnalysisResult:
    """How many times better than random the review ranking is, at each depth.

    ``lift = errors_captured_percent / reviewed_population_percent``. Reviewing
    20% of the population and finding 60% of the errors is a lift of 3.0 — the
    ranking reaches errors three times faster than random review would. The
    0%-reviewed row has no defined lift and is dropped rather than reported as
    zero.
    """
    if not gain.available:
        return AnalysisResult(
            available=False, reason=gain.reason, summary=dict(gain.summary)
        )

    table = gain.table.loc[gain.table["reviewed_population_percent"] > 0].copy()
    if table.empty:  # pragma: no cover - defensive
        return AnalysisResult(
            available=False,
            reason="Gain/Lift not available: no reviewed population to divide by.",
            summary=dict(gain.summary),
        )

    table["lift"] = (
        table["errors_captured_percent"] / table["reviewed_population_percent"]
    ).round(4)
    columns = [
        "reviewed_population_count",
        "reviewed_population_percent",
        "errors_captured_count",
        "errors_captured_percent",
        "lift",
    ]
    table = table[columns].reset_index(drop=True)

    summary = dict(gain.summary)
    summary["max_lift"] = float(table["lift"].max())
    return AnalysisResult(available=True, reason="", table=table, summary=summary)


def render_score_histogram(
    distribution: pd.DataFrame,
    path: str | Path,
    *,
    threshold: float,
    total_non_empty: int,
    title: str = "Composite Weighted Score distribution",
) -> Path:
    """Render the score-band histogram to PNG.

    Bars are split by the routing decision the threshold implies, which is the
    question the chart exists to answer. Two categorical slots, a legend, direct
    count labels on every bar, and a hatch on the auto-accept series so the split
    survives greyscale printing and colour-vision deficiency.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed, and none available
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = distribution["score_band"].tolist()
    counts = distribution["observation_count"].tolist()
    lower_edges = [_band_lower_edge(label) for label in labels]
    is_auto = [edge is not None and edge >= float(threshold) for edge in lower_edges]

    figure, axes = plt.subplots(figsize=(10, 5.2), dpi=150)
    figure.patch.set_facecolor(_SURFACE)
    axes.set_facecolor(_SURFACE)

    bars = axes.bar(
        range(len(labels)),
        counts,
        color=[_COLOR_AUTO if auto else _COLOR_HITL for auto in is_auto],
        hatch=["//" if auto else "" for auto in is_auto],
        edgecolor=_SURFACE,
        linewidth=2,          # 2px surface gap between adjacent fills
        width=0.78,
        zorder=3,
    )

    axes.set_title(title, color=_INK_PRIMARY, fontsize=13, pad=14, loc="left")
    axes.set_xlabel(
        f"Composite Weighted Score band  (n = {total_non_empty} non-empty "
        "address-group instances)",
        color=_INK_SECONDARY,
        fontsize=9.5,
        labelpad=10,
    )
    axes.set_ylabel("Instances", color=_INK_SECONDARY, fontsize=9.5)
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    axes.tick_params(colors=_INK_SECONDARY, length=0)

    # Recessive grid, no chartjunk frame.
    axes.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axes.spines[spine].set_visible(False)
    axes.spines["bottom"].set_color(_GRID)

    # Direct labels: a histogram has few bars, so every count is labelled and
    # the reader never has to trace a bar back to the axis.
    headroom = max(counts) if counts else 0
    for bar, count in zip(bars, counts):
        if count == 0:
            continue
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + headroom * 0.02,
            str(count),
            ha="center", va="bottom", fontsize=9, color=_INK_PRIMARY, zorder=4,
        )
    if headroom:
        axes.set_ylim(0, headroom * 1.16)

    axes.legend(
        handles=[
            Patch(facecolor=_COLOR_AUTO, hatch="//", edgecolor=_SURFACE,
                  label=f"At or above threshold ({threshold:.2f}) — auto-accept candidate"),
            Patch(facecolor=_COLOR_HITL, edgecolor=_SURFACE,
                  label="Below threshold — HITL"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
        frameon=False, fontsize=8.5, labelcolor=_INK_SECONDARY,
    )

    figure.tight_layout()
    figure.savefig(output, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(figure)
    logger.info("wrote score histogram to %s", output)
    return output


def _new_axes(figsize=(10, 5.4)):
    """A configured figure/axes pair sharing the report chart styling."""
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed, and none available
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=figsize, dpi=150)
    figure.patch.set_facecolor(_SURFACE)
    axes.set_facecolor(_SURFACE)
    axes.grid(color=_GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color(_GRID)
    axes.tick_params(colors=_INK_SECONDARY, length=0, labelsize=8.5)
    return figure, axes


def _finish(figure, axes, path: str | Path, *, legend_columns: int = 2) -> Path:
    """Legend below the plot, tight bounds, close the figure, return the path."""
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handles, labels = axes.get_legend_handles_labels()
    if handles:
        axes.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=legend_columns,
            frameon=False, fontsize=8.5, labelcolor=_INK_SECONDARY,
        )
    figure.tight_layout()
    figure.savefig(output, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(figure)
    return output


def _mark_threshold(axes, value: float, label: str, *, color: str, style: str) -> None:
    axes.axvline(
        float(value), color=color, linestyle=style, linewidth=1.4, zorder=2,
        label=f"{label} ({float(value):.2f})",
    )


def render_threshold_tradeoff_chart(
    tradeoff: pd.DataFrame,
    path: str | Path,
    *,
    configured_threshold: float,
    recommended_threshold: float,
    total_non_empty: int,
    title: str = "Threshold Trade-off: Auto-Accept Candidate vs HITL Workload",
) -> Path:
    """Render the cutoff / workload trade-off curve.

    Deliberately *not* labelled with an "optimal" point. The curve shows how
    review workload responds to the cutoff; it says nothing about whether the
    auto-accepted population is correct, so an elbow in it is not evidence for a
    production threshold. Both the configured operational threshold and the
    analytical recommendation are drawn from configuration as reference markers,
    never derived from the shape of the curve.
    """
    figure, axes = _new_axes()

    thresholds = tradeoff["threshold"].astype(float).tolist()
    axes.plot(
        thresholds, tradeoff["auto_accept_candidate_percent"].astype(float),
        color=_COLOR_AUTO, linewidth=2.2, zorder=3,
        label="Auto-accept candidate %",
    )
    axes.plot(
        thresholds, tradeoff["hitl_percent"].astype(float),
        color=_COLOR_HITL, linewidth=2.2, linestyle=(0, (5, 2)), zorder=3,
        label="HITL %",
    )
    _mark_threshold(
        axes, configured_threshold, "Configured operational threshold",
        color=_INK_PRIMARY, style=(0, (2, 2)),
    )
    _mark_threshold(
        axes, recommended_threshold, "Analytical recommendation",
        color=_INK_SECONDARY, style=(0, (1, 2)),
    )

    axes.set_title(title, color=_INK_PRIMARY, fontsize=13, pad=14, loc="left")
    axes.set_xlabel(
        "Composite Weighted Score threshold", color=_INK_SECONDARY,
        fontsize=9.5, labelpad=10,
    )
    axes.set_ylabel(
        f"% of non-empty address-group instances  (n = {total_non_empty})",
        color=_INK_SECONDARY, fontsize=9.5,
    )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(-2, 102)
    output = _finish(figure, axes, path)
    logger.info("wrote threshold trade-off chart to %s", output)
    return output


def render_precision_coverage_chart(
    precision_coverage: pd.DataFrame,
    path: str | Path,
    *,
    configured_threshold: float,
    recommended_threshold: float,
    total_grounded: int,
    title: str = "Auto-Accept Precision vs Grounded Coverage",
) -> Path:
    """Render precision against coverage over fully grounded observations.

    This is the curve a production cutoff should be argued from: it shows what
    is given up in automation to gain quality. The configured and recommended
    thresholds are marked as points on the curve so their cost is visible.
    """
    figure, axes = _new_axes(figsize=(9, 5.6))

    usable = precision_coverage.dropna(subset=["auto_accept_precision"])
    axes.plot(
        usable["grounded_coverage"].astype(float),
        usable["auto_accept_precision"].astype(float),
        color=_COLOR_AUTO, linewidth=2.2, marker="o", markersize=3, zorder=3,
        label="Precision at a given coverage",
    )

    for threshold, label, color in (
        (configured_threshold, "Configured operational threshold", _INK_PRIMARY),
        (recommended_threshold, "Analytical recommendation", _COLOR_HITL),
    ):
        point = _nearest_threshold_row(precision_coverage, threshold)
        if point is None:
            continue
        axes.scatter(
            [float(point["grounded_coverage"])], [float(point["auto_accept_precision"])],
            s=90, color=color, zorder=5, marker="D",
            label=f"{label} ({float(threshold):.2f})",
        )

    axes.set_title(title, color=_INK_PRIMARY, fontsize=13, pad=14, loc="left")
    axes.set_xlabel(
        f"Grounded coverage %  (n = {total_grounded} fully grounded observations)",
        color=_INK_SECONDARY, fontsize=9.5, labelpad=10,
    )
    axes.set_ylabel("Auto-accept precision %", color=_INK_SECONDARY, fontsize=9.5)
    axes.set_xlim(-2, 102)
    axes.set_ylim(-2, 102)
    output = _finish(figure, axes, path, legend_columns=1)
    logger.info("wrote precision/coverage chart to %s", output)
    return output


def _nearest_threshold_row(table: pd.DataFrame, threshold: float):
    """The grid row closest to ``threshold`` that has a defined precision."""
    usable = table.dropna(subset=["auto_accept_precision"])
    if usable.empty:
        return None
    distances = (usable["threshold"].astype(float) - float(threshold)).abs()
    return usable.loc[distances.idxmin()]


def render_error_capture_gain_chart(
    gain: pd.DataFrame,
    path: str | Path,
    *,
    total_grounded: int,
    total_errors: int,
    title: str = "Error-Capture Gain: Low-Score-First HITL Review",
) -> Path:
    """Render the cumulative error-capture curve against the random baseline.

    The diagonal is what reviewing a random sample of the same size would
    achieve. Distance above it is the value the score ranking adds to review
    prioritisation — not a statement about overall model accuracy.
    """
    figure, axes = _new_axes()

    reviewed = gain["reviewed_population_percent"].astype(float)
    axes.plot(
        reviewed, gain["errors_captured_percent"].astype(float),
        color=_COLOR_AUTO, linewidth=2.2, marker="o", markersize=3, zorder=3,
        label="Errors captured by lowest-score-first review",
    )
    axes.plot(
        [0, 100], [0, 100],
        color=_INK_SECONDARY, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2,
        label="Random review baseline",
    )

    axes.set_title(title, color=_INK_PRIMARY, fontsize=13, pad=14, loc="left")
    axes.set_xlabel(
        f"% of fully grounded population sent to review  (n = {total_grounded})",
        color=_INK_SECONDARY, fontsize=9.5, labelpad=10,
    )
    axes.set_ylabel(
        f"% of actual errors captured  (n = {total_errors})",
        color=_INK_SECONDARY, fontsize=9.5,
    )
    axes.set_xlim(-2, 102)
    axes.set_ylim(-2, 102)
    output = _finish(figure, axes, path)
    logger.info("wrote error-capture gain chart to %s", output)
    return output


def render_error_capture_lift_chart(
    lift: pd.DataFrame,
    path: str | Path,
    *,
    total_grounded: int,
    title: str = "Error-Capture Lift vs Random Review",
) -> Path:
    """Render lift against review depth, with the random-review line at 1.0."""
    figure, axes = _new_axes()

    axes.plot(
        lift["reviewed_population_percent"].astype(float),
        lift["lift"].astype(float),
        color=_COLOR_AUTO, linewidth=2.2, marker="o", markersize=3, zorder=3,
        label="Lift over random review",
    )
    axes.axhline(
        1.0, color=_INK_SECONDARY, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2,
        label="Random review (lift = 1.0)",
    )

    axes.set_title(title, color=_INK_PRIMARY, fontsize=13, pad=14, loc="left")
    axes.set_xlabel(
        f"% of fully grounded population sent to review  (n = {total_grounded})",
        color=_INK_SECONDARY, fontsize=9.5, labelpad=10,
    )
    axes.set_ylabel("Lift (x random)", color=_INK_SECONDARY, fontsize=9.5)
    axes.set_xlim(-2, 102)
    axes.set_ylim(bottom=0)
    output = _finish(figure, axes, path)
    logger.info("wrote error-capture lift chart to %s", output)
    return output


@dataclass(frozen=True)
class ExecutiveReport:
    """Every report artifact, in memory and on disk."""

    kpis: pd.DataFrame
    score_distribution: pd.DataFrame
    scenario_distribution: pd.DataFrame
    threshold_sensitivity: pd.DataFrame
    cross_entropy_summary: pd.DataFrame
    hitl_state_distribution: pd.DataFrame
    executive_summary: dict[str, Any]
    paths: dict[str, Path]
    #: Threshold analytics. The trade-off curve is always produced; the three
    #: label-dependent analyses carry their own availability and, when
    #: unavailable, the sentence explaining why.
    threshold_tradeoff: pd.DataFrame = field(default_factory=pd.DataFrame)
    precision_coverage: AnalysisResult = field(
        default_factory=lambda: AnalysisResult(available=False, reason="not computed")
    )
    error_capture_gain: AnalysisResult = field(
        default_factory=lambda: AnalysisResult(available=False, reason="not computed")
    )
    error_capture_lift: AnalysisResult = field(
        default_factory=lambda: AnalysisResult(available=False, reason="not computed")
    )

    @property
    def unavailable_analyses(self) -> dict[str, str]:
        """Analyses the labelled data could not support, and the reason for each."""
        return {
            name: result.reason
            for name, result in (
                ("precision_coverage", self.precision_coverage),
                ("error_capture_gain", self.error_capture_gain),
                ("error_capture_lift", self.error_capture_lift),
            )
            if not result.available
        }


def write_reports(
    metrics: Mapping[str, Any],
    instances: pd.DataFrame,
    config: Any,
    *,
    threshold: float | None = None,
    write_chart: bool = True,
) -> ExecutiveReport:
    """Build every report artifact and write it under the configured directories.

    Parent directories are created on demand, so a deleted ``outputs/`` tree is
    rebuilt rather than crashing the run.
    """
    reporting = config.reporting
    effective_threshold = (
        float(threshold) if threshold is not None else float(config.scoring.hitl_threshold)
    )

    kpis = build_kpi_table(metrics, instances, threshold=effective_threshold)
    distribution = build_score_distribution(instances, reporting.score_band_edges)
    scenarios = build_scenario_distribution(instances)
    sensitivity = build_threshold_sensitivity(
        instances, reporting.sensitivity_thresholds
    )
    cross_entropy = build_cross_entropy_summary(instances)
    hitl_states = build_hitl_state_distribution(instances)
    summary = build_executive_summary(metrics, instances, threshold=effective_threshold)

    # Threshold analytics. Each is decision support only; none of them changes
    # `scoring.hitl_threshold`, which stays configuration.
    step = float(getattr(reporting, "threshold_curve_step", 0.01))
    tradeoff = build_threshold_tradeoff(instances, step=step)
    precision_coverage = build_precision_coverage(
        instances,
        step=step,
        min_grounded=int(getattr(reporting, "min_grounded_for_precision", 1)),
    )
    gain = build_error_capture_gain(
        instances, min_errors=int(getattr(reporting, "min_errors_for_gain", 1))
    )
    lift = build_error_capture_lift(gain)
    for name, result in (
        ("precision/coverage", precision_coverage),
        ("error-capture gain", gain),
        ("error-capture lift", lift),
    ):
        if not result.available:
            logger.info("%s analysis skipped: %s", name, result.reason)

    reports_dir = config.path(reporting.reports_dir)
    charts_dir = config.path(reporting.charts_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["score_distribution"] = _write_csv(
        distribution, reports_dir / reporting.score_distribution_filename
    )
    paths["scenario_distribution"] = _write_csv(
        scenarios, reports_dir / reporting.scenario_distribution_filename
    )
    paths["threshold_sensitivity"] = _write_csv(
        sensitivity, reports_dir / reporting.threshold_sensitivity_filename
    )
    paths["cross_entropy_summary"] = _write_csv(
        cross_entropy, reports_dir / reporting.cross_entropy_summary_filename
    )
    paths["hitl_state_distribution"] = _write_csv(
        hitl_states, reports_dir / reporting.hitl_state_distribution_filename
    )
    paths["threshold_tradeoff"] = _write_csv(
        tradeoff, reports_dir / reporting.threshold_tradeoff_filename
    )
    # A table is written only when the labels support it. A CSV of nothing would
    # read as "the analysis ran and found nothing", which is a different claim.
    if precision_coverage.available:
        paths["precision_coverage"] = _write_csv(
            precision_coverage.table, reports_dir / reporting.precision_coverage_filename
        )
    if gain.available:
        paths["error_capture_gain"] = _write_csv(
            gain.table, reports_dir / reporting.error_capture_gain_filename
        )
    if lift.available:
        paths["error_capture_lift"] = _write_csv(
            lift.table, reports_dir / reporting.error_capture_lift_filename
        )

    summary_path = reports_dir / reporting.executive_summary_filename
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    paths["executive_summary"] = summary_path

    if write_chart:
        charts_dir.mkdir(parents=True, exist_ok=True)
        total_non_empty = len(_non_empty(instances))
        recommended = float(
            getattr(reporting, "recommended_threshold", effective_threshold)
        )
        paths["histogram"] = render_score_histogram(
            distribution,
            charts_dir / reporting.histogram_filename,
            threshold=effective_threshold,
            total_non_empty=total_non_empty,
        )
        paths["threshold_tradeoff_chart"] = render_threshold_tradeoff_chart(
            tradeoff,
            charts_dir / reporting.threshold_tradeoff_chart_filename,
            configured_threshold=effective_threshold,
            recommended_threshold=recommended,
            total_non_empty=total_non_empty,
        )
        if precision_coverage.available:
            paths["precision_coverage_chart"] = render_precision_coverage_chart(
                precision_coverage.table,
                charts_dir / reporting.precision_coverage_chart_filename,
                configured_threshold=effective_threshold,
                recommended_threshold=recommended,
                total_grounded=int(
                    precision_coverage.summary.get("fully_grounded_observations", 0)
                ),
            )
        if gain.available:
            paths["error_capture_gain_chart"] = render_error_capture_gain_chart(
                gain.table,
                charts_dir / reporting.error_capture_gain_chart_filename,
                total_grounded=int(gain.summary.get("fully_grounded_observations", 0)),
                total_errors=int(gain.summary.get("grounded_errors", 0)),
            )
        if lift.available:
            paths["error_capture_lift_chart"] = render_error_capture_lift_chart(
                lift.table,
                charts_dir / reporting.error_capture_lift_chart_filename,
                total_grounded=int(lift.summary.get("fully_grounded_observations", 0)),
            )

    return ExecutiveReport(
        kpis=kpis,
        score_distribution=distribution,
        scenario_distribution=scenarios,
        threshold_sensitivity=sensitivity,
        cross_entropy_summary=cross_entropy,
        hitl_state_distribution=hitl_states,
        executive_summary=summary,
        paths=paths,
        threshold_tradeoff=tradeoff,
        precision_coverage=precision_coverage,
        error_capture_gain=gain,
        error_capture_lift=lift,
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def _percent(count: int, total: int) -> float:
    return round(100.0 * count / total, 2) if total else 0.0


def _band_lower_edge(label: str) -> float | None:
    """The lower edge of a band label, or ``None`` for the "< x" band."""
    if label.startswith("<"):
        return None
    return float(label.split(" ")[0])
