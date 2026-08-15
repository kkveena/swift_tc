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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "BELOW_FIRST_BAND_LABEL",
    "ExecutiveReport",
    "OPERATOR_METADATA_KEYS",
    "data_derived_strings",
    "band_labels",
    "build_cross_entropy_summary",
    "build_executive_summary",
    "build_kpi_table",
    "build_scenario_distribution",
    "build_score_distribution",
    "build_threshold_sensitivity",
    "classify_score",
    "render_score_histogram",
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
    not accepted, because unresolved country ambiguity, an extraction failure,
    and a model/reference conflict force review whatever the number says. Those
    forced cases are reported in their own columns so the override is visible
    rather than implied.
    """
    frame = _non_empty(instances)
    total = len(frame)

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
                    "ambiguous_forced_hitl_count": 0,
                    "error_forced_hitl_count": 0,
                }
            )
            continue

        forced = frame["country_ambiguous"] | frame["extraction_error"]
        meets = frame["composite_weighted_score"] >= float(threshold)
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


def build_kpi_table(
    metrics: Mapping[str, Any],
    instances: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Compact executive KPI table with explicit denominators."""
    frame = _non_empty(instances)
    total_instances = len(frame)
    forced = (
        (frame["country_ambiguous"] | frame["extraction_error"])
        if total_instances
        else pd.Series(dtype=bool)
    )
    auto = (
        ((frame["composite_weighted_score"] >= float(threshold)) & ~forced).sum()
        if total_instances
        else 0
    )
    hitl = total_instances - int(auto)

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
    forced = (
        (frame["country_ambiguous"] | frame["extraction_error"])
        if total_instances
        else pd.Series(dtype=bool)
    )
    auto = (
        int(((frame["composite_weighted_score"] >= float(threshold)) & ~forced).sum())
        if total_instances
        else 0
    )
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


@dataclass(frozen=True)
class ExecutiveReport:
    """Every report artifact, in memory and on disk."""

    kpis: pd.DataFrame
    score_distribution: pd.DataFrame
    scenario_distribution: pd.DataFrame
    threshold_sensitivity: pd.DataFrame
    cross_entropy_summary: pd.DataFrame
    executive_summary: dict[str, Any]
    paths: dict[str, Path]


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
    summary = build_executive_summary(metrics, instances, threshold=effective_threshold)

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

    summary_path = reports_dir / reporting.executive_summary_filename
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    paths["executive_summary"] = summary_path

    if write_chart:
        charts_dir.mkdir(parents=True, exist_ok=True)
        paths["histogram"] = render_score_histogram(
            distribution,
            charts_dir / reporting.histogram_filename,
            threshold=effective_threshold,
            total_non_empty=len(_non_empty(instances)),
        )

    return ExecutiveReport(
        kpis=kpis,
        score_distribution=distribution,
        scenario_distribution=scenarios,
        threshold_sensitivity=sensitivity,
        cross_entropy_summary=cross_entropy,
        executive_summary=summary,
        paths=paths,
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
