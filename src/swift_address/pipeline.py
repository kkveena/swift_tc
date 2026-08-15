"""Phase 1 orchestration: Pass 1 (deterministic) and Pass 2 (unique-address LLM).

The shape of a run:

1. **Validate** the group config against the actual input columns, before any
   model call, so a misconfigured run costs nothing.
2. **Pass 1** builds and cleans one combined address per row per group and
   initializes all 11 output columns. Rows whose combined address is empty are
   finalized here with ``NO_TOWN``/``NO_COUNTRY`` and never enter the work queue.
3. **Dedupe** the non-empty cleaned addresses across every row *and* every
   group into unique cache keys.
4. **Pass 2** extracts each unique cache miss exactly once, concurrently,
   with bounded retry and periodic checkpointing.
5. **Verify and score** once per unique address, then broadcast the result to
   every occurrence.
6. **Emit** the expanded CSV, the error sidecar, and run metrics.

An extraction failure never becomes a business answer. It lands in
``processing_errors.csv``, the affected rows keep safe neutral values and are
routed to HITL, and the failure is visible in run metrics.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from . import io as swift_io
from .cache import AddressCache, CacheEntry, address_hash, make_cache_key
from .cleaning import clean_address
from .gemini_client import (
    AddressExtractionClient,
    ExtractionError,
    ExtractionOutcome,
)
from .grouping import GroupConfig, build_combined_address
from .reference_data import (
    Iso3166Provider,
    ReferenceContext,
    ReferenceDataProvider,
    TownCountryProvider,
    find_iso_provider,
)
from .schemas import (
    MalformedExtractionResponse,
    PromptContract,
    parse_extraction_response,
)
from .serialization import write_detailed_json
from .evaluation import (
    CrossEntropyResult,
    GroundTruth,
    compute_cross_entropy,
    evaluate_ground_truth,
    null_cross_entropy,
    null_ground_truth,
)
from .retraction import RetractionResult, null_retraction, retract_group
from .scoring import ScoreResult, VerifiedExtraction, error_result, evaluate, null_result
from .settings import (
    NULLABLE_BOOLEAN_FIELD_KEYS,
    NULLABLE_FLOAT_FIELD_KEYS,
    OUTPUT_FIELD_KEYS,
    AppConfig,
    raw_logs_allowed,
)

__all__ = [
    "Decision",
    "Pass1Result",
    "Phase1Pipeline",
    "RunResult",
    "WorkItem",
    "run_phase1",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Occurrence:
    """One (row, group) instance that shares a cleaned address."""

    row_index: int
    group_id: str
    record_id: str


@dataclass
class WorkItem:
    """One unique cleaned address plus every place it appears."""

    cache_key: str
    address: str
    occurrences: list[Occurrence] = field(default_factory=list)

    @property
    def address_hash(self) -> str:
        return address_hash(self.address)

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(sorted({occ.group_id for occ in self.occurrences}))

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(occ.record_id for occ in self.occurrences))


@dataclass(frozen=True)
class Decision:
    """Everything Python concluded about one unique address.

    Verification and scoring depend only on the cleaned address and the model
    response, both shared across occurrences, so a decision is computed once per
    unique address and broadcast. Retraction is deliberately *not* here: it
    depends on how the address was split across source columns, which varies per
    row.
    """

    verified: VerifiedExtraction
    score: ScoreResult
    ground_truth: GroundTruth
    cross_entropy: CrossEntropyResult

    @classmethod
    def null(cls) -> "Decision":
        verified, score_result = null_result()
        return cls(verified, score_result, null_ground_truth(), null_cross_entropy())


@dataclass
class Pass1Result:
    """Outcome of the deterministic first pass."""

    frame: pd.DataFrame
    work_items: dict[str, WorkItem]
    empty_instances: int
    non_empty_instances: int

    @property
    def total_instances(self) -> int:
        return self.empty_instances + self.non_empty_instances

    @property
    def unique_addresses(self) -> int:
        return len(self.work_items)

    @property
    def dedupe_saving(self) -> int:
        """Model calls avoided by deduplication alone."""
        return self.non_empty_instances - self.unique_addresses


@dataclass
class RunResult:
    """Everything a caller (notebook, CLI, Phase 2 job) needs after a run."""

    frame: pd.DataFrame
    metrics: dict[str, Any]
    errors: list[swift_io.ProcessingError]
    pass1: Pass1Result
    audit: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Decisions keyed by cleaned address, for the detailed-JSON writer. Bounded
    #: by the unique-address count, not by row count.
    decisions_by_address: dict[str, Decision] = field(default_factory=dict)
    #: One row per **non-empty** address-group instance — the denominator the
    #: executive report uses, since HITL workload occurs at instance level.
    #: Empty instances are counted, not enumerated, so the frame stays bounded
    #: by real work rather than by rows x groups.
    instances: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def output_columns(self) -> int:
        return len(self.frame.columns)


class Phase1Pipeline:
    """Runs Phase 1 end to end.

    The extraction client is injected, so the same pipeline serves a live
    Gemini run, a dry run against the offline stub, and a unit test with a
    scripted double, with no branching inside the pipeline itself.
    """

    def __init__(
        self,
        config: AppConfig,
        group_config: GroupConfig,
        *,
        client: AddressExtractionClient,
        reference_provider: ReferenceDataProvider,
        prompt: PromptContract | None = None,
        cache: AddressCache | None = None,
        mode: str = "live",
        town_country_provider: TownCountryProvider | None = None,
    ) -> None:
        self.config = config
        self.group_config = group_config
        self.client = client
        self.reference_provider = reference_provider
        self.prompt = prompt
        self.mode = mode
        self.iso_provider: Iso3166Provider | None = find_iso_provider(reference_provider)
        self.town_country_provider = town_country_provider

        self.cache = cache if cache is not None else AddressCache(
            config.path(config.processing.cache_path),
            enabled=config.processing.cache_enabled,
        )
        self._checkpoint_lock = threading.Lock()
        self._completed_since_checkpoint = 0

    # -- public API --------------------------------------------------------

    def run(
        self,
        frame: pd.DataFrame,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> RunResult:
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc)

        original_columns = list(frame.columns)
        self.group_config.validate_against_columns(original_columns)

        cache_loaded = self.cache.load()
        pass1 = self.run_pass1(frame)

        outcomes, errors, usage_totals = self.run_pass2(pass1, progress=progress)
        decisions = self._decide(pass1, outcomes)
        result_frame = self._write_results(pass1, decisions)

        swift_io.assert_columns_preserved(original_columns, list(result_frame.columns))

        metrics = self._build_metrics(
            pass1=pass1,
            decisions=decisions,
            errors=errors,
            usage_totals=usage_totals,
            cache_loaded=cache_loaded,
            original_columns=original_columns,
            result_frame=result_frame,
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - started,
        )
        audit = {
            key: {
                "address_hash": address_hash(pass1.work_items[key].address),
                "occurrences": len(pass1.work_items[key].occurrences),
                "group_ids": list(pass1.work_items[key].group_ids),
                "verified": _verified_audit(decision.verified),
                "score": decision.score.to_audit_dict(),
                "ground_truth": decision.ground_truth.to_dict(),
                "cross_entropy": decision.cross_entropy.to_dict(),
            }
            for key, decision in decisions.items()
            if key in pass1.work_items
        }
        return RunResult(
            frame=result_frame,
            metrics=metrics,
            errors=errors,
            pass1=pass1,
            audit=audit,
            decisions_by_address={
                item.address: decisions[key]
                for key, item in pass1.work_items.items()
            },
            instances=_build_instance_frame(pass1, decisions),
        )

    # -- pass 1 ------------------------------------------------------------

    def run_pass1(self, frame: pd.DataFrame) -> Pass1Result:
        """Build, clean, and initialize. Deterministic; no model involvement."""
        zero_is_missing = self.config.input.zero_field_is_missing
        record_column = self.config.project.record_id_column
        record_ids = frame[record_column].astype(str).tolist()

        new_columns: dict[str, list[Any]] = {}
        work_items: dict[str, WorkItem] = {}
        empty_instances = 0
        non_empty_instances = 0

        # Null path: unknown ground truth (never False), no cross-entropy, and
        # an empty retraction. Computed once and reused for every empty instance.
        null_values_by_group = {
            group.group_id: _row_values(
                Decision.null(), null_retraction(group.source_fields)
            )
            for group in self.group_config.enabled_groups
        }

        prompt_version = self.config.project.prompt_version
        reference_version = self.reference_provider.context_version
        model_name = self.client.model

        for group in self.group_config.enabled_groups:
            column_names = self.config.group_column_names(group.group_id)
            columns: dict[str, list[Any]] = {name: [] for name in column_names}
            source_values = [frame[field].tolist() for field in group.source_fields]
            null_values = null_values_by_group[group.group_id]

            for row_index in range(len(frame)):
                combined = build_combined_address(
                    [values[row_index] for values in source_values],
                    zero_is_missing=zero_is_missing,
                )
                cleaned = clean_address(combined)

                columns[column_names[0]].append(combined)
                columns[column_names[1]].append(cleaned)

                if not cleaned:
                    # Null path: finalized here, never enqueued, never a call.
                    empty_instances += 1
                    for name, value in zip(column_names[2:], null_values):
                        columns[name].append(value)
                    continue

                non_empty_instances += 1
                # Placeholders overwritten in _write_results once the unique
                # address has been extracted, verified and scored.
                for name, value in zip(column_names[2:], null_values):
                    columns[name].append(value)

                key = make_cache_key(
                    prompt_version=prompt_version,
                    model=model_name,
                    address=cleaned,
                    reference_context_version=reference_version,
                )
                item = work_items.get(key)
                if item is None:
                    item = WorkItem(cache_key=key, address=cleaned)
                    work_items[key] = item
                item.occurrences.append(
                    Occurrence(
                        row_index=row_index,
                        group_id=group.group_id,
                        record_id=record_ids[row_index],
                    )
                )

            new_columns.update(columns)

        expanded = pd.concat(
            [frame, pd.DataFrame(new_columns, index=frame.index)], axis=1
        )
        logger.info(
            "pass 1: %d group instance(s), %d empty (no model call), %d unique address(es)",
            empty_instances + non_empty_instances,
            empty_instances,
            len(work_items),
        )
        return Pass1Result(
            frame=expanded,
            work_items=work_items,
            empty_instances=empty_instances,
            non_empty_instances=non_empty_instances,
        )

    # -- pass 2 ------------------------------------------------------------

    def run_pass2(
        self,
        pass1: Pass1Result,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[dict[str, ExtractionOutcome], list[swift_io.ProcessingError], dict[str, int]]:
        """Extract every unique cache miss exactly once."""
        outcomes: dict[str, ExtractionOutcome] = {}
        errors: list[swift_io.ProcessingError] = []
        usage_totals: Counter[str] = Counter()

        pending: list[WorkItem] = []
        for key, item in pass1.work_items.items():
            cached = self.cache.get(key)
            if cached is not None:
                try:
                    outcomes[key] = ExtractionOutcome(
                        response=parse_extraction_response(cached.response),
                        model=cached.model or self.client.model,
                        attempts=0,
                    )
                    continue
                except MalformedExtractionResponse:
                    # A cache entry written by an older, incompatible schema.
                    logger.warning("discarding unreadable cache entry %s", key[:12])
            pending.append(item)

        total = len(pending)
        if total == 0:
            logger.info("pass 2: nothing to extract (%d cache hit(s))", len(outcomes))
            return outcomes, errors, dict(usage_totals)

        workers = self.config.model.effective_concurrency
        in_flight = threading.Semaphore(self.config.model.max_in_flight_requests)
        completed = 0

        logger.info(
            "pass 2: extracting %d unique address(es) with %d worker(s) [mode=%s]",
            total,
            workers,
            self.mode,
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._extract_one, item, in_flight): item
                for item in pending
            }
            for future in as_completed(futures):
                item = futures[future]
                completed += 1
                try:
                    outcome = future.result()
                except (ExtractionError, MalformedExtractionResponse) as exc:
                    errors.append(self._to_error(item, exc))
                    logger.error(
                        "extraction failed for address %s (%d occurrence(s)): %s",
                        self._log_ref(item.address),
                        len(item.occurrences),
                        type(exc).__name__,
                    )
                else:
                    outcomes[item.cache_key] = outcome
                    usage_totals.update(
                        {k: int(v) for k, v in outcome.usage.items() if isinstance(v, int)}
                    )
                    self.cache.put(
                        CacheEntry(
                            key=item.cache_key,
                            address_hash=item.address_hash,
                            address=item.address,
                            prompt_version=self.config.project.prompt_version,
                            model=outcome.model,
                            reference_context_version=(
                                self.reference_provider.context_version
                            ),
                            response=outcome.response.to_audit_dict(),
                            metadata={
                                "attempts": outcome.attempts,
                                "usage": dict(outcome.usage),
                                "reference_sources": list(outcome.reference_sources),
                                "mode": self.mode,
                            },
                        )
                    )
                    self._maybe_checkpoint()

                if progress is not None:
                    progress(completed, total)

        written = self.cache.flush()
        if written:
            logger.info("checkpointed %d cache entr(ies)", written)
        return outcomes, errors, dict(usage_totals)

    def _extract_one(
        self, item: WorkItem, in_flight: threading.Semaphore
    ) -> ExtractionOutcome:
        with in_flight:
            context = self._reference_context(item.address)
            return self.client.extract(item.address, context)

    def _reference_context(self, address: str) -> ReferenceContext:
        try:
            return self.reference_provider.get_context(address)
        except Exception as exc:  # noqa: BLE001 - a provider must not fail a run
            logger.warning(
                "reference provider %s failed for address %s: %s",
                getattr(self.reference_provider, "name", "?"),
                self._log_ref(address),
                type(exc).__name__,
            )
            return ReferenceContext()

    def _maybe_checkpoint(self) -> None:
        every = self.config.model.checkpoint_every_unique_addresses
        if every <= 0:
            return
        with self._checkpoint_lock:
            self._completed_since_checkpoint += 1
            due = self._completed_since_checkpoint >= every
            if due:
                self._completed_since_checkpoint = 0
        if due:
            self.cache.flush()

    # -- verification, scoring, write-back ---------------------------------

    def _decide(
        self, pass1: Pass1Result, outcomes: Mapping[str, ExtractionOutcome]
    ) -> dict[str, Decision]:
        """Verify, score, label and evaluate once per unique address.

        Verification depends only on the cleaned address and the model response,
        both of which are shared across occurrences, so this is exact rather
        than an approximation.
        """
        decisions: dict[str, Decision] = {}
        for key, item in pass1.work_items.items():
            outcome = outcomes.get(key)
            if outcome is None:
                verified, score_result = error_result("extraction_unavailable")
                ground_truth = evaluate_ground_truth(verified, extraction_failed=True)
                decisions[key] = Decision(
                    verified=verified,
                    score=score_result,
                    ground_truth=ground_truth,
                    cross_entropy=compute_cross_entropy(verified, ground_truth),
                )
                continue

            verified, score_result = evaluate(
                outcome.response,
                item.address,
                self.config.scoring,
                iso_provider=self.iso_provider,
                town_country_provider=self.town_country_provider,
                separator=self.config.output.country_candidate_separator,
                candidate_sort=self.config.output.country_candidate_sort,
                town_country_ambiguity_policy=(
                    self.config.reference_data.town_country_ambiguity_policy
                ),
                town_country_max_candidates=(
                    self.config.reference_data.town_country_max_candidates
                ),
            )
            ground_truth = evaluate_ground_truth(
                verified, town_country_provider=self.town_country_provider
            )
            decisions[key] = Decision(
                verified=verified,
                score=score_result,
                ground_truth=ground_truth,
                cross_entropy=compute_cross_entropy(verified, ground_truth),
            )
        return decisions

    def _write_results(
        self,
        pass1: Pass1Result,
        decisions: Mapping[str, Decision],
    ) -> pd.DataFrame:
        frame = pass1.frame
        # Column-major staging: gather every update per column, then assign once.
        updates: dict[str, dict[int, Any]] = defaultdict(dict)
        fields_by_group = {
            group.group_id: group.source_fields
            for group in self.group_config.enabled_groups
        }

        for key, item in pass1.work_items.items():
            decision = decisions[key]
            for occurrence in item.occurrences:
                # Retraction is per row/group: the same cleaned address can be
                # split differently across source columns from one row to the next.
                retraction = self.retract_occurrence(
                    frame, occurrence.row_index, occurrence.group_id,
                    fields_by_group[occurrence.group_id], decision.verified,
                )
                values = _row_values(decision, retraction)
                names = self.config.group_column_names(occurrence.group_id)
                for name, value in zip(names[2:], values):
                    updates[name][occurrence.row_index] = value

        for column, row_values in updates.items():
            series = frame[column].copy()
            positions = list(row_values.keys())
            series.iloc[positions] = [row_values[pos] for pos in positions]
            frame[column] = series

        return _coerce_output_dtypes(frame, self.config, self.group_config)

    def retract_occurrence(
        self,
        frame: pd.DataFrame,
        row_index: int,
        group_id: str,
        source_fields: Sequence[str],
        verified: VerifiedExtraction,
    ) -> RetractionResult:
        """Retract one (row, group) instance from its original source columns.

        Reads the source columns; never writes them. The same pure function
        serves the CSV columns and the detailed-JSON writer, so the two can
        never disagree.
        """
        source_values = {
            field_name: frame.iloc[row_index][field_name]
            for field_name in source_fields
            if field_name in frame.columns
        }
        return retract_group(
            source_values,
            source_fields,
            town=verified.town,
            country_value=verified.country_value,
            town_exists=verified.town_exists,
            country_exists=verified.country_exists,
            iso_provider=self.iso_provider,
            zero_is_missing=self.config.input.zero_field_is_missing,
        )

    # -- reporting ---------------------------------------------------------

    def _to_error(self, item: WorkItem, exc: BaseException) -> swift_io.ProcessingError:
        return swift_io.ProcessingError(
            address_hash=item.address_hash,
            occurrences=len(item.occurrences),
            group_ids=item.group_ids,
            record_ids=item.record_ids,
            error_type=type(exc).__name__,
            error_message=str(exc),
            model=self.client.model,
            prompt_version=self.config.project.prompt_version,
            attempts=getattr(exc, "attempts", 0),
        )

    def _town_country_metrics(self) -> dict[str, Any]:
        """Reference-file identity and size, so a run is reproducible from metrics."""
        if self.town_country_provider is None:
            return {
                "enabled": self.config.reference_data.town_country_enabled,
                "loaded": False,
            }
        return {
            "enabled": True,
            "loaded": True,
            "ambiguity_policy": (
                self.config.reference_data.town_country_ambiguity_policy
            ),
            **self.town_country_provider.provenance,
        }

    def _log_ref(self, address: str) -> str:
        """Reference an address in logs without printing it.

        Raw text requires *both* the config opt-out and the
        ``SWIFT_ADDRESS_ALLOW_RAW_LOGS`` environment opt-in, so a stray config
        edit alone cannot start emitting customer addresses into logs.
        """
        if self.config.processing.redact_raw_address_in_logs or not raw_logs_allowed():
            return f"sha256:{address_hash(address)[:16]}"
        return address

    def _build_metrics(
        self,
        *,
        pass1: Pass1Result,
        decisions: Mapping[str, Decision],
        errors: Sequence[swift_io.ProcessingError],
        usage_totals: Mapping[str, int],
        cache_loaded: int,
        original_columns: Sequence[str],
        result_frame: pd.DataFrame,
        started_at: datetime,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        scenarios: Counter[str] = Counter()
        reference_statuses: Counter[str] = Counter()
        cross_entropy_statuses: Counter[str] = Counter()
        hitl_instances = 0
        ambiguous_instances = 0
        ambiguous_addresses = 0
        reference_conflict_instances = 0
        town_grounded = country_grounded = 0

        for key, item in pass1.work_items.items():
            decision = decisions[key]
            verified, score_result = decision.verified, decision.score
            occurrences = len(item.occurrences)
            cross_entropy_statuses[decision.cross_entropy.status] += occurrences
            if decision.ground_truth.town_available:
                town_grounded += occurrences
            if decision.ground_truth.country_available:
                country_grounded += occurrences
            scenarios[score_result.scenario] += occurrences
            reference_statuses[verified.reference_status] += occurrences
            if score_result.needs_hitl:
                hitl_instances += occurrences
            if verified.country_ambiguous:
                ambiguous_instances += occurrences
                ambiguous_addresses += 1
            if verified.reference_conflict:
                reference_conflict_instances += occurrences

        scenarios["null_skip"] += pass1.empty_instances

        enabled = self.group_config.enabled_groups
        appended = len(enabled) * self.config.fields_per_group
        cache_stats = self.cache.stats

        return {
            "run": {
                "started_at_utc": started_at.isoformat(timespec="seconds"),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "mode": self.mode,
                "model": self.client.model,
                "prompt_version": self.config.project.prompt_version,
                "prompt_path": (
                    str(self.prompt.source_path) if self.prompt is not None else None
                ),
                "naming_style": self.config.output.naming_style,
            },
            "reference_data": {
                "provider": getattr(self.reference_provider, "name", "unknown"),
                "context_version": self.reference_provider.context_version,
                "provenance": getattr(self.reference_provider, "provenance", {}),
                "google_search_grounding": (
                    self.config.model.enable_google_search_grounding
                ),
                "town_country": self._town_country_metrics(),
            },
            "shape": {
                "input_rows": len(result_frame),
                "input_columns": len(original_columns),
                "groups_enabled": len(enabled),
                "groups_configured": len(self.group_config.groups),
                "fields_per_group": self.config.fields_per_group,
                "appended_columns": appended,
                "output_columns": len(result_frame.columns),
            },
            "pass1": {
                "group_instances": pass1.total_instances,
                "empty_instances_skipped": pass1.empty_instances,
                "non_empty_instances": pass1.non_empty_instances,
            },
            "efficiency": {
                "unique_addresses": pass1.unique_addresses,
                "calls_avoided_by_null_skip": pass1.empty_instances,
                "calls_avoided_by_dedupe": pass1.dedupe_saving,
                "cache_entries_loaded": cache_loaded,
                "cache_hits": cache_stats["hits"],
                "cache_misses": cache_stats["misses"],
                "backend_calls": self.client.call_count,
                "token_usage": dict(usage_totals),
            },
            "outcomes": {
                "scenario_counts": dict(sorted(scenarios.items())),
                "extraction_errors": len(errors),
                "instances_affected_by_errors": sum(e.occurrences for e in errors),
                "ambiguous_country_addresses": ambiguous_addresses,
                "ambiguous_country_instances": ambiguous_instances,
                "reference_status_counts": dict(sorted(reference_statuses.items())),
                "reference_conflict_instances": reference_conflict_instances,
            },
            "evaluation": {
                "cross_entropy_status_counts": dict(
                    sorted(cross_entropy_statuses.items())
                ),
                "town_ground_truth_instances": town_grounded,
                "country_ground_truth_instances": country_grounded,
                "non_empty_instances": pass1.non_empty_instances,
            },
            "hitl": {
                "threshold": self.config.scoring.hitl_threshold,
                "instances_below_threshold": hitl_instances,
                "instances_total": pass1.total_instances,
                "force_ambiguous_to_hitl": (
                    self.config.scoring.force_ambiguous_country_to_hitl
                ),
            },
        }


def _row_values(decision: Decision, retraction: RetractionResult) -> list[Any]:
    """The post-address output values, in :data:`OUTPUT_FIELD_KEYS` order.

    Length is derived from the field tuple, so adding an output field means
    editing `OUTPUT_FIELD_KEYS` and this list together and nothing else.

    ``town_exists_ok`` / ``country_exists_ok`` are plain booleans — unknown or
    unresolved evidence collapses to ``False`` rather than a blank cell.
    ``cross_entropy`` is likewise ``None`` when ungrounded, which the CSV
    writes as a blank cell rather than a misleading zero.
    """
    verified = decision.verified
    return [
        verified.town,
        verified.country_value,
        verified.country_name_value,
        float(verified.town_probability),
        float(verified.country_probability),
        bool(verified.town_exists),
        bool(verified.country_exists),
        float(decision.score.composite_weighted_score),
        verified.rationale_town,
        verified.rationale_country,
        decision.ground_truth.town_exists_ok,
        decision.ground_truth.country_exists_ok,
        # Rounded to the same 6 places the detailed JSON uses, so the two
        # representations of the same number are byte-comparable.
        _round_or_none(decision.cross_entropy.group_cross_entropy),
        retraction.combined_address_retracted,
        retraction.comment,
    ]


#: Column contract of :attr:`RunResult.instances`, also used to build an empty
#: frame when a run has no non-empty instances at all.
INSTANCE_COLUMNS: tuple[str, ...] = (
    "record_id",
    "group_id",
    "composite_weighted_score",
    "scenario",
    "country_ambiguous",
    "extraction_error",
    "reference_status",
    "needs_hitl",
    # Evaluation columns. town_exists_ok / country_exists_ok are plain
    # booleans (unknown collapses to False); the paired *_grounded flags say
    # whether that came from real evidence, so cross-entropy and the
    # correctness rate can still exclude coverage gaps from the loss rather
    # than counting them as a model failure.
    "town_exists_ok",
    "town_grounded",
    "country_exists_ok",
    "country_grounded",
    "cross_entropy",
    "cross_entropy_status",
    # Raw confidences, kept for calibration diagnostics.
    "town_probability",
    "country_probability",
)


def _build_instance_frame(
    pass1: Pass1Result,
    decisions: Mapping[str, Decision],
) -> pd.DataFrame:
    """Explode unique-address decisions back to one row per non-empty instance."""
    records: list[dict[str, Any]] = []
    for key, item in pass1.work_items.items():
        decision = decisions[key]
        verified, score_result = decision.verified, decision.score
        for occurrence in item.occurrences:
            records.append(
                {
                    "record_id": occurrence.record_id,
                    "group_id": occurrence.group_id,
                    "composite_weighted_score": float(
                        score_result.composite_weighted_score
                    ),
                    "scenario": score_result.scenario,
                    "country_ambiguous": bool(verified.country_ambiguous),
                    "extraction_error": score_result.scenario == "extraction_error",
                    "reference_status": verified.reference_status,
                    "needs_hitl": bool(score_result.needs_hitl),
                    "town_exists_ok": bool(decision.ground_truth.town_exists_ok),
                    "town_grounded": bool(decision.ground_truth.town_available),
                    "country_exists_ok": bool(decision.ground_truth.country_exists_ok),
                    "country_grounded": bool(decision.ground_truth.country_available),
                    "cross_entropy": decision.cross_entropy.group_cross_entropy,
                    "cross_entropy_status": decision.cross_entropy.status,
                    "town_probability": float(verified.town_probability),
                    "country_probability": float(verified.country_probability),
                }
            )
    if not records:
        return pd.DataFrame(columns=list(INSTANCE_COLUMNS))

    frame = pd.DataFrame(records, columns=list(INSTANCE_COLUMNS))
    frame["town_exists_ok"] = frame["town_exists_ok"].astype(bool)
    frame["town_grounded"] = frame["town_grounded"].astype(bool)
    frame["country_exists_ok"] = frame["country_exists_ok"].astype(bool)
    frame["country_grounded"] = frame["country_grounded"].astype(bool)
    frame["cross_entropy"] = pd.to_numeric(frame["cross_entropy"], errors="coerce")
    return frame


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _verified_audit(verified: VerifiedExtraction) -> dict[str, Any]:
    return {
        "town": verified.town,
        "country_candidates": list(verified.country_candidates),
        "country_value": verified.country_value,
        "country_name_value": verified.country_name_value,
        "town_exists": verified.town_exists,
        "country_exists": verified.country_exists,
        "country_ambiguous": verified.country_ambiguous,
        # Town/Country reference findings stay in the audit trail; they are not
        # part of the production CSV contract.
        "reference_status": verified.reference_status,
        "reference_codes": list(verified.reference_codes),
        "notes": list(verified.notes),
    }


def _coerce_output_dtypes(
    frame: pd.DataFrame, config: AppConfig, group_config: GroupConfig
) -> pd.DataFrame:
    """Give the generated columns their intended dtypes.

    Source columns are left exactly as read (strings); only generated columns
    are touched.
    """
    float_keys = {
        "predicted_town_probability",
        "predicted_country_probability",
        "composite_weighted_score",
    }
    bool_keys = {
        "predicted_town_exists",
        "predicted_country_exists",
        "town_exists_ok",
        "country_exists_ok",
    }

    for group in group_config.enabled_groups:
        for key in OUTPUT_FIELD_KEYS:
            column = config.output.column_name(key, group.group_id)
            if key in float_keys:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
            elif key in NULLABLE_FLOAT_FIELD_KEYS:
                # NaN, so an ungrounded observation writes a blank CSV cell
                # rather than a 0.0 that would read as a perfect score.
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            elif key in NULLABLE_BOOLEAN_FIELD_KEYS:
                # pandas BooleanDtype keeps True / False / <NA> distinct;
                # plain bool would silently turn "unknown" into False.
                frame[column] = frame[column].astype("boolean")
            elif key in bool_keys:
                frame[column] = frame[column].astype(bool)
            else:
                frame[column] = frame[column].astype(str)
    return frame


def run_phase1(
    input_path: str,
    config: AppConfig,
    group_config: GroupConfig,
    *,
    client: AddressExtractionClient,
    reference_provider: ReferenceDataProvider,
    prompt: PromptContract | None = None,
    output_path: str | None = None,
    mode: str = "live",
    write_outputs: bool = True,
    town_country_provider: TownCountryProvider | None = None,
) -> RunResult:
    """Convenience entry point: read, run, write CSV + errors + metrics."""
    frame = swift_io.read_input_csv(
        config.path(input_path), record_id_column=config.project.record_id_column
    )
    pipeline = Phase1Pipeline(
        config,
        group_config,
        client=client,
        reference_provider=reference_provider,
        prompt=prompt,
        mode=mode,
        town_country_provider=town_country_provider,
    )
    result = pipeline.run(frame)

    if write_outputs:
        target = output_path or config.processing.output_path
        swift_io.write_output_csv(result.frame, config.path(target))
        swift_io.write_errors_csv(
            result.errors, config.path(config.processing.errors_path)
        )
        swift_io.write_metrics_json(
            result.metrics, config.path(config.processing.metrics_path)
        )
        if config.processing.detailed_json_enabled:
            write_detailed_json(
                result.frame,
                config.path(config.processing.detailed_json_path),
                config=config,
                group_config=group_config,
                decisions_by_address=result.decisions_by_address,
                iso_provider=pipeline.iso_provider,
                output_format=config.processing.detailed_json_format,
                include_empty_groups=(
                    config.processing.detailed_json_include_empty_groups
                ),
            )
    return result
