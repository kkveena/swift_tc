"""Group configuration drives grouping; nothing about groups is hard-coded."""

from __future__ import annotations

import pytest

from models.swft_tc.src.grouping import (
    AddressGroup,
    GroupConfig,
    GroupConfigError,
    MissingInputColumnsError,
    build_combined_address,
    build_group_addresses,
    load_group_config,
)


class TestSampleGroupConfig:
    def test_loads_sixteen_enabled_groups(self, group_config):
        assert len(group_config.groups) == 16
        assert len(group_config.enabled_groups) == 16

    def test_group_ids_are_unique(self, group_config):
        ids = [group.group_id for group in group_config.groups]
        assert len(ids) == len(set(ids))

    def test_field_order_comes_from_the_config_file(self, group_config):
        group15 = next(g for g in group_config.groups if g.group_id == "15")
        assert group15.source_fields == (
            "PRI_PAY_BNF_ADDR_LINE_1",
            "PRI_PAY_BNF_ADDR_LINE_2",
            "PRI_PAY_BNF_ADDR_LINE_3",
        )

    def test_every_configured_column_exists_in_the_sample_input(
        self, group_config, sample_input_path
    ):
        from models.swft_tc.src.io import read_input_csv

        frame = read_input_csv(sample_input_path)
        group_config.validate_against_columns(frame.columns)  # must not raise

    def test_all_source_fields_are_deduplicated_in_order(self, group_config):
        fields = group_config.all_source_fields
        assert len(fields) == len(set(fields)) == 48  # 16 groups x 3 lines


class TestGroupCountIsConfigDriven:
    """Changing the config changes the run; no code knows "16"."""

    def _write(self, tmp_path, text: str):
        path = tmp_path / "groups.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_three_groups_of_varying_width(self, tmp_path):
        path = self._write(
            tmp_path,
            "group_id,address_line_1,address_line_2,address_line_3,enabled,notes\n"
            "alpha,A1,A2,A3,True,\n"
            "beta,B1,B2,,True,\n"
            "gamma,C1,,,True,\n",
        )
        config = load_group_config(path)
        assert [g.group_id for g in config.groups] == ["alpha", "beta", "gamma"]
        assert [g.line_count for g in config.groups] == [3, 2, 1]

    def test_five_lines_per_group_is_supported(self, tmp_path):
        path = self._write(
            tmp_path,
            "group_id,l1,l2,l3,l4,l5,enabled\n"
            "1,A,B,C,D,E,True\n",
        )
        config = load_group_config(path)
        assert config.groups[0].source_fields == ("A", "B", "C", "D", "E")

    def test_disabled_groups_are_excluded(self, tmp_path):
        path = self._write(
            tmp_path,
            "group_id,address_line_1,enabled\n"
            "1,A1,True\n"
            "2,B1,False\n"
            "3,C1,true\n",
        )
        config = load_group_config(path)
        assert len(config.groups) == 3
        assert [g.group_id for g in config.enabled_groups] == ["1", "3"]

    def test_yaml_group_config_is_supported(self, tmp_path):
        path = tmp_path / "groups.yaml"
        path.write_text(
            "groups:\n"
            "  - group_id: '1'\n"
            "    source_fields: [A1, A2]\n"
            "  - group_id: '2'\n"
            "    source_fields: [B1]\n"
            "    enabled: false\n",
            encoding="utf-8",
        )
        config = load_group_config(path)
        assert len(config.groups) == 2
        assert len(config.enabled_groups) == 1


class TestGroupConfigValidation:
    def test_duplicate_group_ids_are_rejected(self):
        with pytest.raises(GroupConfigError, match="duplicate group_id"):
            GroupConfig(
                groups=(
                    AddressGroup(group_id="1", source_fields=("A",)),
                    AddressGroup(group_id="1", source_fields=("B",)),
                )
            )

    def test_duplicate_source_fields_within_a_group_are_rejected(self):
        with pytest.raises(GroupConfigError, match="duplicate source field"):
            AddressGroup(group_id="1", source_fields=("A", "B", "A"))

    def test_group_with_no_source_fields_is_rejected(self):
        with pytest.raises(GroupConfigError, match="no source fields"):
            AddressGroup(group_id="1", source_fields=())

    def test_missing_input_columns_are_reported_before_any_model_call(self):
        config = GroupConfig(
            groups=(
                AddressGroup(group_id="1", source_fields=("PRESENT", "ABSENT_ONE")),
                AddressGroup(group_id="2", source_fields=("ABSENT_TWO",)),
            )
        )
        with pytest.raises(MissingInputColumnsError) as excinfo:
            config.validate_against_columns(["RECORD_ID", "PRESENT"])

        message = str(excinfo.value)
        assert "ABSENT_ONE" in message and "ABSENT_TWO" in message
        assert "group 1" in message and "group 2" in message
        assert "No model calls were made." in message
        assert excinfo.value.missing_by_group == {
            "1": ("ABSENT_ONE",),
            "2": ("ABSENT_TWO",),
        }

    def test_disabled_groups_do_not_trigger_missing_column_errors(self):
        config = GroupConfig(
            groups=(AddressGroup(group_id="1", source_fields=("GONE",), enabled=False),)
        )
        config.validate_against_columns(["RECORD_ID"])  # must not raise

    def test_unparseable_enabled_flag_is_rejected(self, tmp_path):
        path = tmp_path / "groups.csv"
        path.write_text("group_id,address_line_1,enabled\n1,A1,perhaps\n", encoding="utf-8")
        with pytest.raises(GroupConfigError, match="enabled="):
            load_group_config(path)

    def test_missing_file_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_group_config(tmp_path / "nope.csv")

    def test_unsupported_format_is_rejected(self, tmp_path):
        path = tmp_path / "groups.txt"
        path.write_text("nope", encoding="utf-8")
        with pytest.raises(GroupConfigError, match="unsupported"):
            load_group_config(path)


class TestCombinedAddressConstruction:
    def test_joins_in_configuration_order_with_one_space(self):
        assert build_combined_address(
            ["23 CUSTOMS STREET EAST LEVEL 11", "CITIGROUP CENTRE AUCKLAND AUCKLAND", "1140 NZ"]
        ) == "23 CUSTOMS STREET EAST LEVEL 11 CITIGROUP CENTRE AUCKLAND AUCKLAND 1140 NZ"

    def test_whole_field_zero_is_omitted(self):
        assert build_combined_address(["1 LINCOLN STREET", "BOSTON MA 02111 US", "0"]) == (
            "1 LINCOLN STREET BOSTON MA 02111 US"
        )

    def test_digits_inside_values_survive(self):
        assert build_combined_address(["388 GREENWICH STREET", "NEW YORK NY 10013-2632 US", "0"]) == (
            "388 GREENWICH STREET NEW YORK NY 10013-2632 US"
        )
        assert "10013-2632" in build_combined_address(["10013-2632"])
        assert build_combined_address(["LEVEL 10", "0", "SUITE 0A"]) == "LEVEL 10 SUITE 0A"

    def test_nulls_blanks_and_nan_are_dropped(self):
        assert build_combined_address([None, "  ", float("nan"), "ACCRA", "", "GH"]) == (
            "ACCRA GH"
        )

    def test_all_missing_yields_empty_string(self):
        assert build_combined_address([None, "", "0", "   "]) == ""

    def test_each_field_is_trimmed_before_joining(self):
        assert build_combined_address(["  A  ", "  B  "]) == "A B"

    def test_order_is_preserved_not_sorted(self):
        assert build_combined_address(["ZULU", "ALPHA"]) == "ZULU ALPHA"

    def test_build_group_addresses_returns_combined_and_cleaned(self):
        group = AddressGroup(group_id="1", source_fields=("L1", "L2", "L3"))
        row = {"L1": "  1 LINCOLN   STREET ", "L2": "BOSTON MA 02111 US", "L3": "0"}
        combined, cleaned = build_group_addresses(row, group)
        assert combined == "1 LINCOLN   STREET BOSTON MA 02111 US"
        assert cleaned == "1 LINCOLN STREET BOSTON MA 02111 US"
