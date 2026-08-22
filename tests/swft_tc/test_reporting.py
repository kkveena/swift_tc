"""Executive reporting: band boundaries, denominators, and threshold sensitivity."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from models.swft_tc.src.reporting import (
    band_labels,
    build_cross_entropy_summary,
    build_error_capture_gain,
    build_error_capture_lift,
    build_executive_summary,
    build_kpi_table,
    build_precision_coverage,
    build_scenario_distribution,
    build_score_distribution,
    build_threshold_sensitivity,
    build_threshold_tradeoff,
    classify_score,
    data_derived_strings,
    forced_review_mask,
    render_error_capture_gain_chart,
    render_error_capture_lift_chart,
    render_precision_coverage_chart,
    render_score_histogram,
    render_threshold_tradeoff_chart,
    threshold_grid,
    write_reports,
)

EDGES = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def instances(rows) -> pd.DataFrame:
    """Build an instance frame from (score, scenario, ambiguous, error) tuples."""
    frame = pd.DataFrame(
        [
            {
                "record_id": f"R{index}",
                "group_id": "15",
                "composite_weighted_score": score,
                "scenario": scenario,
                "country_ambiguous": ambiguous,
                "extraction_error": error,
                "reference_status": "consistent",
                "needs_hitl": ambiguous or error or score < 0.80,
            }
            for index, (score, scenario, ambiguous, error) in enumerate(rows)
        ],
        columns=[
            "record_id", "group_id", "composite_weighted_score", "scenario",
            "country_ambiguous", "extraction_error", "reference_status", "needs_hitl",
        ],
    )
    return frame


def evaluated(rows) -> pd.DataFrame:
    """Instance frame carrying labels: (town_ok, country_ok, cross_entropy).

    ``None`` in ``town_ok`` / ``country_ok`` means "ungrounded" for building
    fixtures; it is translated into the plain boolean (``False``) the real
    pipeline now writes, plus a separate ``*_grounded`` flag so correctness-rate
    math still excludes it rather than counting it as incorrect.
    """
    base = instances([(0.9, "both_explicit", False, False)] * len(rows))
    town_ok = [row[0] for row in rows]
    country_ok = [row[1] for row in rows]
    base["town_exists_ok"] = [bool(v) if v is not None else False for v in town_ok]
    base["town_grounded"] = [v is not None for v in town_ok]
    base["country_exists_ok"] = [bool(v) if v is not None else False for v in country_ok]
    base["country_grounded"] = [v is not None for v in country_ok]
    base["cross_entropy"] = pd.to_numeric(
        pd.Series([row[2] for row in rows]), errors="coerce"
    )
    return base


class TestBandLabels:
    def test_labels_match_the_specified_bands(self):
        assert band_labels(EDGES) == (
            "< 0.60",
            "0.60 - <0.65", "0.65 - <0.70", "0.70 - <0.75", "0.75 - <0.80",
            "0.80 - <0.85", "0.85 - <0.90", "0.90 - <0.95", "0.95 - 1.00",
        )

    def test_nine_bands_from_eight_edges(self):
        assert len(band_labels(EDGES)) == len(EDGES) + 1

    def test_labels_follow_configured_edges(self):
        assert band_labels((0.5, 0.9)) == ("< 0.50", "0.50 - <0.90", "0.90 - 1.00")

    def test_empty_edges_are_rejected(self):
        with pytest.raises(ValueError):
            band_labels(())


class TestBoundaryRules:
    """Every boundary belongs to the band it opens."""

    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.0000, "< 0.60"),
            (0.5999, "< 0.60"),
            (0.6000, "0.60 - <0.65"),
            (0.6499, "0.60 - <0.65"),
            (0.6500, "0.65 - <0.70"),
            (0.7000, "0.70 - <0.75"),
            (0.7499, "0.70 - <0.75"),
            (0.7500, "0.75 - <0.80"),
            (0.8000, "0.80 - <0.85"),
            (0.8500, "0.85 - <0.90"),
            (0.9000, "0.90 - <0.95"),
            (0.9499, "0.90 - <0.95"),
            (0.9500, "0.95 - 1.00"),
            (0.9801, "0.95 - 1.00"),
            (1.0000, "0.95 - 1.00"),
        ],
    )
    def test_classification(self, score, expected):
        assert classify_score(score, EDGES) == expected


class TestScoreDistribution:
    @pytest.fixture
    def distribution(self):
        return build_score_distribution(
            instances(
                [
                    (0.0, "no_defensible_prediction", False, False),
                    (0.0, "town_explicit_country_ambiguous", True, False),
                    (0.349125, "town_explicit_country_inferred", False, False),
                    (0.62, "neither_explicit_both_inferred", False, False),
                    (0.9702, "both_explicit", False, False),
                    (1.0, "both_explicit", False, False),
                ]
            ),
            EDGES,
        )

    def test_every_band_is_present_even_at_zero(self, distribution):
        assert list(distribution["score_band"]) == list(band_labels(EDGES))
        assert len(distribution) == 9

    def test_counts_land_in_the_right_bands(self, distribution):
        by_band = dict(
            zip(distribution["score_band"], distribution["observation_count"])
        )
        assert by_band["< 0.60"] == 3
        assert by_band["0.60 - <0.65"] == 1
        assert by_band["0.95 - 1.00"] == 2
        assert by_band["0.70 - <0.75"] == 0

    def test_counts_sum_to_the_denominator(self, distribution):
        assert distribution["observation_count"].sum() == 6

    def test_cumulative_columns_are_monotonic_and_complete(self, distribution):
        cumulative = distribution["cumulative_count"].tolist()
        assert cumulative == sorted(cumulative)
        assert cumulative[-1] == 6
        assert distribution["cumulative_percent"].iloc[-1] == pytest.approx(100.0)

    def test_percentages_use_the_non_empty_denominator(self, distribution):
        row = distribution[distribution["score_band"] == "< 0.60"].iloc[0]
        assert row["percent_of_non_empty"] == pytest.approx(50.0)

    def test_required_columns(self, distribution):
        assert list(distribution.columns) == [
            "score_band", "observation_count", "percent_of_non_empty",
            "cumulative_count", "cumulative_percent",
        ]

    def test_empty_input_yields_zeroed_bands(self):
        distribution = build_score_distribution(pd.DataFrame(), EDGES)
        assert len(distribution) == 9
        assert distribution["observation_count"].sum() == 0
        assert distribution["percent_of_non_empty"].max() == 0.0


class TestNullInstancesExcluded:
    """Skipped empty groups must not pad the histogram."""

    def test_denominator_is_non_empty_instances_only(
        self, config, group_config, reference_provider, mock_client, sample_input_path
    ):
        from models.swft_tc.src.io import read_input_csv
        from models.swft_tc.src.pipeline import Phase1Pipeline

        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider, mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        # 8 rows x 16 groups = 128 instances, of which 121 are empty.
        assert result.metrics["pass1"]["group_instances"] == 128
        assert result.metrics["pass1"]["empty_instances_skipped"] == 121
        assert len(result.instances) == 7

        distribution = build_score_distribution(
            result.instances, config.reporting.score_band_edges
        )
        assert distribution["observation_count"].sum() == 7      # not 128
        assert distribution["cumulative_percent"].iloc[-1] == pytest.approx(100.0)

    def test_instance_frame_holds_only_non_empty_rows(
        self, config, group_config, reference_provider, mock_client, sample_input_path
    ):
        from models.swft_tc.src.io import read_input_csv
        from models.swft_tc.src.pipeline import Phase1Pipeline

        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider, mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        assert "null_skip" not in set(result.instances["scenario"])
        assert set(result.instances["group_id"]) == {"15"}


class TestScenarioDistribution:
    def test_counts_and_percentages(self):
        frame = build_scenario_distribution(
            instances(
                [
                    (0.97, "both_explicit", False, False),
                    (0.97, "both_explicit", False, False),
                    (0.35, "town_explicit_country_inferred", False, False),
                    (0.0, "town_explicit_country_ambiguous", True, False),
                ]
            )
        )
        assert list(frame.columns) == [
            "scenario", "observation_count", "percent_of_non_empty"
        ]
        top = frame.iloc[0]
        assert top["scenario"] == "both_explicit"
        assert top["observation_count"] == 2
        assert top["percent_of_non_empty"] == pytest.approx(50.0)
        assert frame["observation_count"].sum() == 4

    def test_ordering_is_deterministic_for_ties(self):
        frame = build_scenario_distribution(
            instances(
                [
                    (0.9, "zeta_scenario", False, False),
                    (0.9, "alpha_scenario", False, False),
                ]
            )
        )
        assert frame["scenario"].tolist() == ["alpha_scenario", "zeta_scenario"]

    def test_empty_input(self):
        assert build_scenario_distribution(pd.DataFrame()).empty


class TestThresholdSensitivity:
    @pytest.fixture
    def sensitivity(self):
        return build_threshold_sensitivity(
            instances(
                [
                    (0.9801, "both_explicit", False, False),
                    (0.9702, "both_explicit", False, False),
                    (0.9000, "both_explicit", False, False),
                    (0.8200, "country_explicit_town_inferred", False, False),
                    (0.3491, "town_explicit_country_inferred", False, False),
                    (0.0000, "town_explicit_country_ambiguous", True, False),
                    (0.0000, "extraction_error", False, True),
                ]
            ),
            (0.80, 0.85, 0.90, 0.95),
        )

    def test_all_thresholds_present(self, sensitivity):
        assert sensitivity["threshold"].tolist() == [0.80, 0.85, 0.90, 0.95]

    def test_required_columns(self, sensitivity):
        assert list(sensitivity.columns) == [
            "threshold", "auto_accept_candidate_count", "auto_accept_candidate_percent",
            "hitl_count", "hitl_percent", "forced_review_count", "low_score_hitl_count",
            "ambiguous_forced_hitl_count", "error_forced_hitl_count",
        ]

    def test_raising_the_threshold_never_increases_auto_accept(self, sensitivity):
        counts = sensitivity["auto_accept_candidate_count"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_counts_partition_the_population(self, sensitivity):
        for _, row in sensitivity.iterrows():
            assert row["auto_accept_candidate_count"] + row["hitl_count"] == 7
            assert row["auto_accept_candidate_percent"] + row["hitl_percent"] == (
                pytest.approx(100.0)
            )

    def test_ambiguity_is_forced_to_hitl_regardless_of_score(self):
        """A perfect score with unresolved ambiguity is still not auto-acceptable."""
        sensitivity = build_threshold_sensitivity(
            instances([(1.0, "town_explicit_country_ambiguous", True, False)]),
            (0.80,),
        )
        row = sensitivity.iloc[0]
        assert row["auto_accept_candidate_count"] == 0
        assert row["hitl_count"] == 1
        assert row["ambiguous_forced_hitl_count"] == 1

    def test_errors_are_forced_to_hitl_regardless_of_score(self):
        sensitivity = build_threshold_sensitivity(
            instances([(1.0, "extraction_error", False, True)]), (0.80,)
        )
        row = sensitivity.iloc[0]
        assert row["auto_accept_candidate_count"] == 0
        assert row["error_forced_hitl_count"] == 1

    def test_forced_counts_are_threshold_invariant(self, sensitivity):
        assert sensitivity["ambiguous_forced_hitl_count"].nunique() == 1
        assert sensitivity["error_forced_hitl_count"].nunique() == 1

    def test_threshold_boundary_is_inclusive(self, sensitivity):
        """A score exactly at the threshold is an auto-accept candidate."""
        at_090 = sensitivity[sensitivity["threshold"] == 0.90].iloc[0]
        assert at_090["auto_accept_candidate_count"] == 3     # 0.9801, 0.9702, 0.9000

    def test_empty_input(self):
        sensitivity = build_threshold_sensitivity(pd.DataFrame(), (0.80, 0.90))
        assert len(sensitivity) == 2
        assert sensitivity["auto_accept_candidate_count"].sum() == 0


class TestKpiTable:
    def test_denominators_are_labelled(self):
        metrics = {
            "shape": {"input_rows": 8, "groups_enabled": 16},
            "pass1": {"group_instances": 128, "empty_instances_skipped": 121,
                      "non_empty_instances": 7},
            "efficiency": {"unique_addresses": 7, "backend_calls": 7, "cache_hits": 0},
            "outcomes": {"extraction_errors": 0, "ambiguous_country_instances": 1,
                         "reference_conflict_instances": 0},
        }
        table = build_kpi_table(
            metrics,
            instances([(0.97, "both_explicit", False, False),
                       (0.0, "town_explicit_country_ambiguous", True, False)]),
            threshold=0.90,
        )
        assert list(table.columns) == ["metric", "value", "denominator"]
        rows = dict(zip(table["metric"], table["value"]))
        assert rows["Input records"] == 8
        assert rows["Address-group instances"] == 128
        assert rows["Auto-accept candidates"] == 1
        assert rows["HITL instances"] == 1

        denominators = dict(zip(table["metric"], table["denominator"]))
        assert denominators["Input records"] == "records"
        assert denominators["HITL %"] == "% of non-empty instances"

    def test_counts_stay_integers_beside_percentages(self):
        """Mixing counts and percentages must not render 8 as 8.00."""
        table = build_kpi_table(
            {"shape": {"input_rows": 8}, "pass1": {}, "efficiency": {}, "outcomes": {}},
            instances([(0.97, "both_explicit", False, False)]),
            threshold=0.90,
        )
        values = dict(zip(table["metric"], table["value"]))
        assert values["Input records"] == 8
        assert isinstance(values["Input records"], int)
        assert isinstance(values["HITL %"], float)


class TestExecutiveSummary:
    def test_shape_and_content(self):
        metrics = {
            "run": {"started_at_utc": "2026-08-15T00:00:00+00:00", "mode": "dry_run",
                    "model": "mock", "prompt_version": "v2"},
            "reference_data": {"context_version": "iso-1",
                               "town_country": {"source_version": "tc-1",
                                                "approved_for_production": False}},
            "shape": {"input_rows": 8},
            "pass1": {"group_instances": 128, "empty_instances_skipped": 121},
            "efficiency": {"unique_addresses": 7, "backend_calls": 7},
            "outcomes": {"ambiguous_country_instances": 1, "extraction_errors": 0,
                         "reference_conflict_instances": 0},
        }
        summary = build_executive_summary(
            metrics,
            instances([(0.97, "both_explicit", False, False),
                       (0.0, "town_explicit_country_ambiguous", True, False)]),
            threshold=0.90,
        )
        assert summary["hitl_threshold"] == 0.90
        assert summary["records"] == 8
        assert summary["non_empty_address_instances"] == 2
        assert summary["auto_accept_candidates"] == 1
        assert summary["hitl_instances"] == 1
        assert summary["hitl_percent"] == pytest.approx(50.0)
        assert summary["town_country_approved_for_production"] is False
        assert "Provisional" in summary["threshold_basis"]

    def test_no_raw_address_appears(
        self, config, group_config, reference_provider, mock_client, sample_input_path
    ):
        """Run the real sample end to end, then look for its addresses in the JSON.

        Only *values* are searched: key names like `address_group_instances`
        legitimately contain the word "address".
        """
        from models.swft_tc.src.io import read_input_csv
        from models.swft_tc.src.pipeline import Phase1Pipeline

        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider, mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        summary = build_executive_summary(
            result.metrics, result.instances, threshold=0.90
        )
        # Operator-chosen metadata (model, prompt version, reference version) is
        # excluded: those labels are named by humans and may legitimately contain
        # a word that also appears in a test address.
        values = json.dumps(data_derived_strings(summary)).upper()

        for fragment in (
            "LINCOLN", "BOSTON", "GREENWICH", "ACCRA", "AUCKLAND", "JIRON",
            "AERONAUTICA", "TAIPEI", "02111", "10013",
        ):
            assert fragment not in values, f"{fragment} leaked into the summary"

    def test_operator_metadata_is_excluded_from_the_privacy_scan(self):
        """A prompt version named after a test case is not an address leak."""
        summary = build_executive_summary(
            {
                "run": {"prompt_version": "v5-greenwich-json-regression",
                        "model": "gemini-3.5-flash", "mode": "live"},
                "reference_data": {"town_country": {}}, "shape": {}, "pass1": {},
                "efficiency": {}, "outcomes": {},
            },
            instances([(0.97, "both_explicit", False, False)]),
            threshold=0.90,
        )
        assert summary["prompt_version"] == "v5-greenwich-json-regression"
        assert "GREENWICH" not in json.dumps(data_derived_strings(summary)).upper()

    def test_record_identifiers_do_not_appear(
        self, config, group_config, reference_provider, mock_client, sample_input_path
    ):
        from models.swft_tc.src.io import read_input_csv
        from models.swft_tc.src.pipeline import Phase1Pipeline

        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider, mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        serialized = json.dumps(
            build_executive_summary(result.metrics, result.instances, threshold=0.90)
        )
        assert "CA0000000318" not in serialized


class TestWriteReports:
    def test_all_artifacts_are_written(self, config, tmp_path):
        metrics = {
            "run": {"mode": "dry_run", "model": "mock"},
            "reference_data": {"context_version": "iso-1", "town_country": {}},
            "shape": {"input_rows": 2}, "pass1": {"group_instances": 2},
            "efficiency": {}, "outcomes": {},
        }
        report = write_reports(
            metrics,
            instances([(0.97, "both_explicit", False, False),
                       (0.35, "town_explicit_country_inferred", False, False)]),
            config,
        )
        for key in (
            "score_distribution", "scenario_distribution",
            "threshold_sensitivity", "executive_summary", "histogram",
        ):
            assert report.paths[key].exists(), key
        assert report.paths["histogram"].stat().st_size > 1000   # a real PNG

    def test_missing_output_directories_are_created(self, config, tmp_path):
        """A deleted outputs/ tree must not break the next run."""
        import shutil

        reports_dir = config.path(config.reporting.reports_dir)
        charts_dir = config.path(config.reporting.charts_dir)
        for directory in (reports_dir, charts_dir):
            if directory.exists():
                shutil.rmtree(directory)
        assert not reports_dir.exists() and not charts_dir.exists()

        report = write_reports(
            {"run": {}, "reference_data": {}, "shape": {}, "pass1": {},
             "efficiency": {}, "outcomes": {}},
            instances([(0.97, "both_explicit", False, False)]),
            config,
        )
        assert reports_dir.exists() and charts_dir.exists()
        assert report.paths["executive_summary"].exists()

    def test_threshold_override_is_honoured(self, config):
        report = write_reports(
            {"run": {}, "reference_data": {}, "shape": {}, "pass1": {},
             "efficiency": {}, "outcomes": {}},
            instances([(0.92, "both_explicit", False, False)]),
            config,
            threshold=0.95,
        )
        assert report.executive_summary["hitl_threshold"] == 0.95
        assert report.executive_summary["auto_accept_candidates"] == 0

    def test_csv_artifacts_reload(self, config):
        report = write_reports(
            {"run": {}, "reference_data": {}, "shape": {}, "pass1": {},
             "efficiency": {}, "outcomes": {}},
            instances([(0.97, "both_explicit", False, False)]),
            config,
        )
        reloaded = pd.read_csv(report.paths["score_distribution"])
        assert len(reloaded) == 9
        summary = json.loads(report.paths["executive_summary"].read_text())
        assert summary["non_empty_address_instances"] == 1


class TestCrossEntropySummary:
    """Calibration reporting: grounded observations only, lower is better."""

    def _value(self, frame, metric):
        return dict(zip(frame["metric"], frame["value"]))[metric]

    def test_ungrounded_rows_are_excluded_from_the_denominator(self):
        summary = build_cross_entropy_summary(
            evaluated([
                (True, True, 0.02),
                (True, False, 2.99),
                (None, None, None),      # reference gap — must not enter the loss
                (None, None, None),
            ])
        )
        assert self._value(summary, "Non-empty instances") == 4
        assert self._value(summary, "Grounded observations") == 2
        assert self._value(summary, "Ungrounded (excluded from loss)") == 2

    def test_mean_median_p95_use_grounded_rows_only(self):
        summary = build_cross_entropy_summary(
            evaluated([
                (True, True, 0.02),
                (True, True, 0.04),
                (None, None, None),
            ])
        )
        assert self._value(summary, "Mean cross-entropy") == pytest.approx(0.03)
        assert self._value(summary, "Median cross-entropy") == pytest.approx(0.03)
        assert self._value(summary, "P95 cross-entropy") == pytest.approx(0.039, abs=1e-3)

    def test_correctness_rates_use_their_own_denominators(self):
        summary = build_cross_entropy_summary(
            evaluated([
                (True, True, 0.02),
                (True, False, 2.99),
                (False, None, 2.99),
                (None, None, None),
            ])
        )
        # Town: 3 labelled, 2 correct. Country: 2 labelled, 1 correct.
        assert self._value(summary, "Town correctness rate") == pytest.approx(66.67)
        assert self._value(summary, "Country correctness rate") == pytest.approx(50.0)

    def test_denominators_are_labelled_distinctly(self):
        summary = build_cross_entropy_summary(
            evaluated([(True, True, 0.02), (False, None, 2.99)])
        )
        denominators = dict(zip(summary["metric"], summary["denominator"]))
        assert "lower is better" in denominators["Mean cross-entropy"]
        assert denominators["Town correctness rate"].endswith("town-grounded instances")

    def test_fully_ungrounded_input_reports_nothing_rather_than_zero_loss(self):
        summary = build_cross_entropy_summary(
            evaluated([(None, None, None), (None, None, None)])
        )
        assert self._value(summary, "Grounded observations") == 0
        assert self._value(summary, "Mean cross-entropy") is None

    def test_empty_input(self):
        summary = build_cross_entropy_summary(pd.DataFrame())
        assert self._value(summary, "Grounded observations") == 0

    def test_written_as_a_report_artifact(self, config):
        report = write_reports(
            {"run": {}, "reference_data": {}, "shape": {}, "pass1": {},
             "efficiency": {}, "outcomes": {}},
            evaluated([(True, True, 0.02)]),
            config,
        )
        assert report.paths["cross_entropy_summary"].exists()
        reloaded = pd.read_csv(report.paths["cross_entropy_summary"])
        assert "Mean cross-entropy" in set(reloaded["metric"])

    def test_real_run_produces_a_grounded_summary(
        self, config, group_config, reference_provider, town_country_provider,
        mock_client, sample_input_path,
    ):
        from models.swft_tc.src.io import read_input_csv
        from models.swft_tc.src.pipeline import Phase1Pipeline

        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider,
            town_country_provider=town_country_provider, mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        summary = build_cross_entropy_summary(result.instances)
        assert self._value(summary, "Non-empty instances") == 7
        assert self._value(summary, "Grounded observations") >= 1
        assert self._value(summary, "Mean cross-entropy") is not None


class TestHistogram:
    def test_renders_a_png(self, tmp_path):
        distribution = build_score_distribution(
            instances([(0.97, "both_explicit", False, False),
                       (0.2, "no_defensible_prediction", False, False)]),
            EDGES,
        )
        path = render_score_histogram(
            distribution, tmp_path / "charts" / "hist.png",
            threshold=0.90, total_non_empty=2,
        )
        assert path.exists()
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_renders_with_an_all_zero_distribution(self, tmp_path):
        distribution = build_score_distribution(pd.DataFrame(), EDGES)
        path = render_score_histogram(
            distribution, tmp_path / "hist.png", threshold=0.90, total_non_empty=0
        )
        assert path.exists()


# ---------------------------------------------------------------------------
# Threshold analytics
# ---------------------------------------------------------------------------


def labelled(rows) -> pd.DataFrame:
    """Instance frame from (score, town_ok, country_ok) with full grounding.

    ``None`` for either label means that half is ungrounded, so the observation
    is not fully grounded and must be excluded from precision and gain.
    """
    base = instances([(row[0], "both_explicit", False, False) for row in rows])
    town_ok = [row[1] for row in rows]
    country_ok = [row[2] for row in rows]
    base["town_exists_ok"] = [bool(v) if v is not None else False for v in town_ok]
    base["town_grounded"] = [v is not None for v in town_ok]
    base["country_exists_ok"] = [bool(v) if v is not None else False for v in country_ok]
    base["country_grounded"] = [v is not None for v in country_ok]
    return base


def conflicted(frame: pd.DataFrame, positions) -> pd.DataFrame:
    """Mark the given row positions as reference-conflicted."""
    frame = frame.copy()
    frame.loc[frame.index[list(positions)], "reference_status"] = "conflict"
    return frame


class TestThresholdGrid:
    def test_default_step_spans_zero_to_one_inclusive(self):
        grid = threshold_grid(0.01)
        assert grid[0] == 0.0 and grid[-1] == 1.0
        assert len(grid) == 101

    def test_coarse_step_is_exact(self):
        assert threshold_grid(0.25) == (0.0, 0.25, 0.5, 0.75, 1.0)

    def test_values_are_not_binary_float_noise(self):
        assert 0.07 in threshold_grid(0.01)

    def test_a_step_that_does_not_divide_one_still_closes_at_one(self):
        grid = threshold_grid(0.3)
        assert grid[-1] == 1.0

    @pytest.mark.parametrize("step", [0.0, -0.1, 1.5])
    def test_an_out_of_range_step_is_rejected(self, step):
        with pytest.raises(ValueError):
            threshold_grid(step)


class TestForcedReviewMask:
    """One definition of "forced", shared by every reporting path."""

    def test_a_reference_conflict_is_forced(self):
        frame = conflicted(instances([(0.91, "both_explicit", False, False)]), [0])
        assert bool(forced_review_mask(frame).iloc[0]) is True

    def test_ambiguity_and_errors_are_forced(self):
        frame = instances(
            [
                (0.91, "both_explicit", True, False),
                (0.0, "extraction_error", False, True),
                (0.91, "both_explicit", False, False),
            ]
        )
        assert forced_review_mask(frame).tolist() == [True, True, False]

    def test_the_engine_flag_is_honoured_when_present(self):
        """`hitl_forced_review` is what the real pipeline writes."""
        frame = instances([(0.95, "both_explicit", False, False)])
        frame["hitl_forced_review"] = [True]
        assert bool(forced_review_mask(frame).iloc[0]) is True

    def test_a_manual_override_column_is_honoured(self):
        frame = instances([(0.99, "both_explicit", False, False)])
        frame["manual_override"] = [True]
        assert bool(forced_review_mask(frame).iloc[0]) is True

    def test_an_empty_frame_yields_an_empty_mask(self):
        assert forced_review_mask(pd.DataFrame()).empty


class TestReferenceConflictNeverAutoAccepts:
    """The regression this consistency fix exists for.

    Composite 0.91, threshold 0.80, reference conflict True. The score clears
    the cutoff comfortably; the control still wins, in every reporting path.
    """

    @pytest.fixture
    def frame(self):
        return conflicted(instances([(0.91, "both_explicit", False, False)]), [0])

    def test_sensitivity_table_keeps_it_in_hitl(self, frame):
        row = build_threshold_sensitivity(frame, [0.80]).iloc[0]
        assert row["auto_accept_candidate_count"] == 0
        assert row["hitl_count"] == 1
        assert row["forced_review_count"] == 1
        assert row["low_score_hitl_count"] == 0

    def test_kpi_table_keeps_it_in_hitl(self, frame):
        kpis = build_kpi_table({}, frame, threshold=0.80).set_index("metric")
        assert kpis.loc["Auto-accept candidates", "value"] == 0
        assert kpis.loc["HITL instances", "value"] == 1
        assert kpis.loc["  forced by a control, whatever the threshold", "value"] == 1

    def test_executive_summary_keeps_it_in_hitl(self, frame):
        summary = build_executive_summary({}, frame, threshold=0.80)
        assert summary["auto_accept_candidates"] == 0
        assert summary["hitl_instances"] == 1
        assert summary["forced_review_instances"] == 1
        assert summary["low_score_only_hitl_instances"] == 0

    def test_threshold_tradeoff_keeps_it_in_hitl_at_every_cutoff(self, frame):
        tradeoff = build_threshold_tradeoff(frame, step=0.1)
        assert tradeoff["auto_accept_candidate_count"].max() == 0
        assert tradeoff["forced_review_count"].unique().tolist() == [1]

    def test_lowering_the_threshold_never_rescues_it(self, frame):
        """Even at 0.00 — the point of a forced-review control."""
        row = build_threshold_tradeoff(frame, step=0.5).iloc[0]
        assert row["threshold"] == 0.0
        assert row["auto_accept_candidate_count"] == 0

    def test_precision_coverage_never_counts_it_as_an_auto_accept(self):
        frame = conflicted(labelled([(0.91, True, True), (0.95, True, True)]), [0])
        result = build_precision_coverage(frame, step=0.5)
        row = result.table.loc[result.table["threshold"] == 0.5].iloc[0]
        # Both score above 0.5; only the unconflicted one is a candidate.
        assert row["grounded_auto_accept_count"] == 1


class TestThresholdTradeoff:
    @pytest.fixture
    def tradeoff(self):
        # Two clear the 0.90 bar, one clears 0.80, one is low, one is forced.
        frame = instances(
            [
                (0.95, "both_explicit", False, False),
                (0.92, "both_explicit", False, False),
                (0.85, "both_explicit", False, False),
                (0.10, "neither_explicit_both_inferred", False, False),
                (0.99, "both_explicit", True, False),
            ]
        )
        return build_threshold_tradeoff(frame, step=0.01)

    def test_required_columns(self, tradeoff):
        assert list(tradeoff.columns) == [
            "threshold", "auto_accept_candidate_count",
            "auto_accept_candidate_percent", "hitl_count", "hitl_percent",
            "forced_review_count", "low_score_hitl_count",
        ]

    def test_grid_is_complete(self, tradeoff):
        assert len(tradeoff) == 101
        assert tradeoff["threshold"].iloc[0] == 0.0
        assert tradeoff["threshold"].iloc[-1] == 1.0

    def test_at_zero_every_unforced_instance_is_a_candidate(self, tradeoff):
        row = tradeoff.iloc[0]
        assert row["auto_accept_candidate_count"] == 4
        assert row["hitl_count"] == 1        # the ambiguous one
        assert row["low_score_hitl_count"] == 0

    def test_at_the_configured_threshold(self, tradeoff):
        row = tradeoff.loc[tradeoff["threshold"] == 0.80].iloc[0]
        assert row["auto_accept_candidate_count"] == 3
        assert row["hitl_count"] == 2

    def test_at_the_recommended_threshold(self, tradeoff):
        row = tradeoff.loc[tradeoff["threshold"] == 0.90].iloc[0]
        assert row["auto_accept_candidate_count"] == 2
        assert row["low_score_hitl_count"] == 2

    def test_at_one_only_a_perfect_score_could_qualify(self, tradeoff):
        row = tradeoff.loc[tradeoff["threshold"] == 1.00].iloc[0]
        assert row["auto_accept_candidate_count"] == 0
        assert row["hitl_count"] == 5

    def test_counts_are_monotonic_in_the_threshold(self, tradeoff):
        counts = tradeoff["auto_accept_candidate_count"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_forced_count_is_flat_across_the_grid(self, tradeoff):
        assert tradeoff["forced_review_count"].unique().tolist() == [1]

    def test_percentages_always_sum_to_one_hundred(self, tradeoff):
        total = (
            tradeoff["auto_accept_candidate_percent"] + tradeoff["hitl_percent"]
        )
        assert total.round(6).eq(100.0).all()

    def test_null_skips_are_excluded(self):
        """`instances` already holds only non-empty rows; an empty frame is zero."""
        tradeoff = build_threshold_tradeoff(pd.DataFrame(), step=0.5)
        assert tradeoff["auto_accept_candidate_count"].sum() == 0
        assert tradeoff["hitl_count"].sum() == 0
        assert len(tradeoff) == 3


class TestPrecisionCoverage:
    def test_precision_and_coverage_are_exact(self):
        # At threshold 0.90: three candidates (0.95, 0.92, 0.90), two correct.
        frame = labelled(
            [
                (0.95, True, True),
                (0.92, True, True),
                (0.90, True, False),      # wrong: country label is False
                (0.50, False, False),
            ]
        )
        result = build_precision_coverage(frame, step=0.01)
        assert result.available
        row = result.table.loc[result.table["threshold"] == 0.90].iloc[0]
        assert row["grounded_auto_accept_count"] == 3
        assert row["grounded_correct_auto_accept_count"] == 2
        assert row["auto_accept_precision"] == pytest.approx(66.67, abs=0.01)
        assert row["grounded_coverage"] == pytest.approx(75.0)
        assert row["grounded_hitl_count"] == 1

    def test_a_correct_group_needs_both_halves_correct(self):
        frame = labelled([(0.95, True, False)])
        result = build_precision_coverage(frame, step=0.5)
        row = result.table.loc[result.table["threshold"] == 0.5].iloc[0]
        assert row["grounded_auto_accept_count"] == 1
        assert row["grounded_correct_auto_accept_count"] == 0
        assert row["auto_accept_precision"] == 0.0

    def test_ungrounded_rows_are_excluded_entirely(self):
        frame = labelled(
            [
                (0.95, True, True),
                (0.95, None, True),       # town ungrounded
                (0.95, True, None),       # country ungrounded
                (0.95, None, None),
            ]
        )
        result = build_precision_coverage(frame, step=0.5)
        assert result.summary["fully_grounded_observations"] == 1
        row = result.table.loc[result.table["threshold"] == 0.5].iloc[0]
        assert row["grounded_auto_accept_count"] == 1
        assert row["grounded_coverage"] == pytest.approx(100.0)

    def test_precision_is_none_when_nothing_qualifies(self):
        frame = labelled([(0.50, True, True)])
        result = build_precision_coverage(frame, step=0.5)
        row = result.table.loc[result.table["threshold"] == 1.0].iloc[0]
        assert row["grounded_auto_accept_count"] == 0
        # Undefined, not zero: no candidates means no precision to report. It
        # arrives as NaN because the column holds floats, and writes as an
        # empty CSV cell.
        assert pd.isna(row["auto_accept_precision"])

    def test_no_grounded_observations_reports_unavailable(self):
        frame = labelled([(0.95, None, None), (0.80, None, None)])
        result = build_precision_coverage(frame, step=0.5)
        assert result.available is False
        assert "insufficient fully grounded observations" in result.reason
        assert result.table.empty

    def test_a_configured_minimum_is_honoured(self):
        frame = labelled([(0.95, True, True)])
        assert build_precision_coverage(frame, step=0.5, min_grounded=5).available is False
        assert build_precision_coverage(frame, step=0.5, min_grounded=1).available is True


class TestErrorCaptureGain:
    @pytest.fixture
    def gain(self):
        """Ten grounded observations; the five lowest scores hold 4 of 5 errors."""
        rows = [
            (0.10, False, True),   # error
            (0.20, True, False),   # error
            (0.30, False, False),  # error
            (0.40, True, True),
            (0.50, False, True),   # error
            (0.60, True, True),
            (0.70, True, True),
            (0.80, True, True),
            (0.90, True, True),
            (0.95, True, False),   # error, but scored high
        ]
        return build_error_capture_gain(labelled(rows))

    def test_available_with_a_labelled_error_population(self, gain):
        assert gain.available
        assert gain.summary["fully_grounded_observations"] == 10
        assert gain.summary["grounded_errors"] == 5

    def test_the_curve_starts_at_the_origin(self, gain):
        first = gain.table.iloc[0]
        assert first["reviewed_population_percent"] == 0.0
        assert first["errors_captured_percent"] == 0.0

    def test_reviewing_twenty_percent_captures_forty_percent(self, gain):
        row = gain.table.loc[gain.table["reviewed_population_count"] == 2].iloc[0]
        assert row["reviewed_population_percent"] == pytest.approx(20.0)
        assert row["errors_captured_percent"] == pytest.approx(40.0)

    def test_reviewing_half_captures_four_of_five_errors(self, gain):
        row = gain.table.loc[gain.table["reviewed_population_count"] == 5].iloc[0]
        assert row["errors_captured_count"] == 4
        assert row["errors_captured_percent"] == pytest.approx(80.0)

    def test_full_review_captures_every_error(self, gain):
        last = gain.table.iloc[-1]
        assert last["reviewed_population_percent"] == pytest.approx(100.0)
        assert last["errors_captured_percent"] == pytest.approx(100.0)

    def test_review_order_is_ascending_score(self, gain):
        scores = gain.table["composite_weighted_score"].dropna().tolist()
        assert scores == sorted(scores)

    def test_early_review_beats_the_random_baseline(self, gain):
        """Where the ranking earns its keep: the shallow end of the review queue.

        The fixture deliberately hides one error at the *top* of the score range,
        so the curve does dip below the diagonal near full review — that is the
        honest shape for data like this, not a defect.
        """
        early = gain.table.loc[gain.table["reviewed_population_percent"].between(10, 50)]
        assert (
            early["errors_captured_percent"] > early["random_baseline_percent"]
        ).all()

    def test_a_high_scoring_error_is_visible_as_a_late_catch(self, gain):
        """The last error is only reached at full review depth."""
        table = gain.table
        at_ninety = table.loc[table["reviewed_population_count"] == 9].iloc[0]
        assert at_ninety["errors_captured_count"] == 4
        assert table.iloc[-1]["errors_captured_count"] == 5

    def test_ungrounded_rows_never_enter_the_population(self):
        result = build_error_capture_gain(
            labelled([(0.10, False, False), (0.20, None, None), (0.90, True, True)])
        )
        assert result.summary["fully_grounded_observations"] == 2

    def test_no_grounded_observations_reports_unavailable(self):
        result = build_error_capture_gain(labelled([(0.95, None, None)]))
        assert result.available is False
        assert "insufficient grounded errors" in result.reason

    def test_no_errors_reports_unavailable_rather_than_a_flat_chart(self):
        result = build_error_capture_gain(
            labelled([(0.95, True, True), (0.80, True, True)])
        )
        assert result.available is False
        assert "insufficient grounded errors" in result.reason
        assert result.table.empty


class TestErrorCaptureLift:
    def test_the_worked_example(self):
        """Review 20% of the population, capture 60% of errors -> lift 3.0."""
        rows = [
            (0.10, False, False),   # error
            (0.20, False, False),   # error
            (0.30, True, True),
            (0.40, True, True),
            (0.50, True, True),
            (0.60, True, True),
            (0.70, True, True),
            (0.80, True, True),
            (0.90, True, True),
            (0.95, False, False),   # error, scored high
        ]
        # 10 observations, 3 errors; the 2 lowest scores hold 2 of them.
        lift = build_error_capture_lift(build_error_capture_gain(labelled(rows)))
        assert lift.available
        row = lift.table.loc[lift.table["reviewed_population_count"] == 2].iloc[0]
        assert row["reviewed_population_percent"] == pytest.approx(20.0)
        assert row["errors_captured_percent"] == pytest.approx(66.67, abs=0.01)
        assert row["lift"] == pytest.approx(3.33, abs=0.01)

    def test_the_specified_twenty_sixty_three_case(self):
        """The exact figures from the specification: 20% reviewed, 60% of errors, lift 3.0.

        20 fully grounded observations with 5 errors; 3 of them sit in the 4
        lowest-scoring cases.
        """
        rows = (
            [(0.01, False, False), (0.02, True, False), (0.03, False, True),
             (0.04, True, True)]                                   # 3 errors in the first 4
            + [(0.10 + 0.04 * i, True, True) for i in range(14)]   # 14 clean
            + [(0.90, False, False), (0.95, True, False)]          # 2 errors scored high
        )
        assert len(rows) == 20
        gain = build_error_capture_gain(labelled(rows))
        assert gain.summary["grounded_errors"] == 5

        lift = build_error_capture_lift(gain)
        row = lift.table.loc[lift.table["reviewed_population_count"] == 4].iloc[0]
        assert row["reviewed_population_percent"] == pytest.approx(20.0)
        assert row["errors_captured_count"] == 3
        assert row["errors_captured_percent"] == pytest.approx(60.0)
        assert row["lift"] == pytest.approx(3.0)

    def test_exact_three_times_lift(self):
        """5 observations, 1 error, sitting at the very bottom: 20% -> 100%."""
        rows = [(0.10, False, False)] + [(0.2 * i, True, True) for i in range(2, 6)]
        lift = build_error_capture_lift(build_error_capture_gain(labelled(rows)))
        row = lift.table.loc[lift.table["reviewed_population_count"] == 1].iloc[0]
        assert row["reviewed_population_percent"] == pytest.approx(20.0)
        assert row["errors_captured_percent"] == pytest.approx(100.0)
        assert row["lift"] == pytest.approx(5.0)

    def test_full_review_always_has_lift_one(self):
        rows = [(0.10, False, False), (0.50, True, True), (0.90, True, True)]
        lift = build_error_capture_lift(build_error_capture_gain(labelled(rows)))
        assert lift.table["lift"].iloc[-1] == pytest.approx(1.0)

    def test_the_undefined_zero_percent_row_is_dropped(self):
        rows = [(0.10, False, False), (0.90, True, True)]
        lift = build_error_capture_lift(build_error_capture_gain(labelled(rows)))
        assert (lift.table["reviewed_population_percent"] > 0).all()

    def test_unavailable_gain_yields_unavailable_lift_with_the_same_reason(self):
        gain = build_error_capture_gain(labelled([(0.95, True, True)]))
        lift = build_error_capture_lift(gain)
        assert lift.available is False
        assert lift.reason == gain.reason
        assert lift.table.empty


class TestThresholdAnalyticsArtifacts:
    """End-to-end: what write_reports puts on disk, and what it refuses to."""

    def test_tradeoff_artifacts_are_always_written(self, config):
        frame = labelled([(0.95, True, True), (0.10, False, False)])
        report = write_reports({}, frame, config)
        assert report.paths["threshold_tradeoff"].exists()
        assert report.paths["threshold_tradeoff_chart"].exists()
        assert not report.threshold_tradeoff.empty

    def test_label_dependent_artifacts_appear_when_labels_allow(self, config):
        rows = [(0.10, False, False), (0.50, True, True), (0.95, True, True)]
        report = write_reports({}, labelled(rows), config)
        assert report.precision_coverage.available
        assert report.error_capture_gain.available
        assert report.error_capture_lift.available
        for key in (
            "precision_coverage", "precision_coverage_chart",
            "error_capture_gain", "error_capture_gain_chart",
            "error_capture_lift", "error_capture_lift_chart",
        ):
            assert report.paths[key].exists(), key

    def test_nothing_misleading_is_written_without_labels(self, config):
        frame = labelled([(0.95, None, None), (0.10, None, None)])
        report = write_reports({}, frame, config)
        assert report.precision_coverage.available is False
        assert report.error_capture_gain.available is False
        assert report.error_capture_lift.available is False
        for key in (
            "precision_coverage", "precision_coverage_chart",
            "error_capture_gain", "error_capture_gain_chart",
            "error_capture_lift", "error_capture_lift_chart",
        ):
            assert key not in report.paths, key
        # The workload curve does not need labels and is still produced.
        assert report.paths["threshold_tradeoff"].exists()

    def test_unavailable_analyses_carry_a_readable_reason(self, config):
        report = write_reports({}, labelled([(0.95, None, None)]), config)
        reasons = report.unavailable_analyses
        assert set(reasons) == {
            "precision_coverage", "error_capture_gain", "error_capture_lift"
        }
        assert all(reason.strip() for reason in reasons.values())

    def test_no_grounded_errors_still_leaves_precision_available(self, config):
        """Perfect labels: precision is computable, gain is not."""
        report = write_reports({}, labelled([(0.95, True, True)]), config)
        assert report.precision_coverage.available is True
        assert report.error_capture_gain.available is False

    def test_the_analytics_never_change_the_operational_threshold(self, config):
        """Charts are decision support; the cutoff stays configuration."""
        before = config.scoring.hitl_threshold
        rows = [(0.10, False, False), (0.50, True, True), (0.95, True, True)]
        report = write_reports({}, labelled(rows), config)
        assert config.scoring.hitl_threshold == before
        assert report.executive_summary["hitl_threshold"] == round(float(before), 4)


class TestAnalyticsChartsRender:
    def test_every_chart_writes_a_png(self, tmp_path):
        rows = [(0.10, False, False), (0.50, True, True), (0.95, True, True)]
        frame = labelled(rows)
        tradeoff = build_threshold_tradeoff(frame, step=0.05)
        precision = build_precision_coverage(frame, step=0.05)
        gain = build_error_capture_gain(frame)
        lift = build_error_capture_lift(gain)

        paths = [
            render_threshold_tradeoff_chart(
                tradeoff, tmp_path / "tradeoff.png",
                configured_threshold=0.80, recommended_threshold=0.90,
                total_non_empty=len(frame),
            ),
            render_precision_coverage_chart(
                precision.table, tmp_path / "precision.png",
                configured_threshold=0.80, recommended_threshold=0.90,
                total_grounded=3,
            ),
            render_error_capture_gain_chart(
                gain.table, tmp_path / "gain.png", total_grounded=3, total_errors=1,
            ),
            render_error_capture_lift_chart(
                lift.table, tmp_path / "lift.png", total_grounded=3,
            ),
        ]
        for path in paths:
            assert path.exists() and path.stat().st_size > 0

