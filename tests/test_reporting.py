"""Executive reporting: band boundaries, denominators, and threshold sensitivity."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from swift_address.reporting import (
    band_labels,
    build_executive_summary,
    build_kpi_table,
    build_scenario_distribution,
    build_score_distribution,
    build_threshold_sensitivity,
    classify_score,
    render_score_histogram,
    write_reports,
)

EDGES = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def instances(rows) -> pd.DataFrame:
    """Build an instance frame from (score, scenario, ambiguous, error) tuples."""
    return pd.DataFrame(
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
        from swift_address.io import read_input_csv
        from swift_address.pipeline import Phase1Pipeline

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
        from swift_address.io import read_input_csv
        from swift_address.pipeline import Phase1Pipeline

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
            "hitl_count", "hitl_percent", "ambiguous_forced_hitl_count",
            "error_forced_hitl_count",
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
        from swift_address.io import read_input_csv
        from swift_address.pipeline import Phase1Pipeline

        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider, mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        summary = build_executive_summary(
            result.metrics, result.instances, threshold=0.90
        )
        values = json.dumps(
            [value for value in summary.values() if isinstance(value, str)]
        ).upper()

        for fragment in (
            "LINCOLN", "BOSTON", "GREENWICH", "ACCRA", "AUCKLAND", "JIRON",
            "AERONAUTICA", "TAIPEI", "02111", "10013",
        ):
            assert fragment not in values, f"{fragment} leaked into the summary"

    def test_record_identifiers_do_not_appear(
        self, config, group_config, reference_provider, mock_client, sample_input_path
    ):
        from swift_address.io import read_input_csv
        from swift_address.pipeline import Phase1Pipeline

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
