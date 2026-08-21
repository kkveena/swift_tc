# Claude Code Enhancement Prompt — `kkveena/swift_tc`

You are modifying the existing repository `kkveena/swift_tc`. Do **not** rebuild the project from scratch. First inspect the current implementation and preserve its existing design strengths: two-pass processing, null short-circuiting, cross-group deduplication, cache/checkpointing, Python-owned deterministic scoring, YAML configuration, structured Gemini output, local ISO validation, no secret leakage, no fake SWIFTRef access, and tests.

## Objective

Enhance Phase 1 in five focused areas:

1. add deterministic expanded Country Name output;
2. make the `outputs/` structure explicit and reusable;
3. add a reusable Town↔Country development reference provider sourced from a public reference dataset;
4. add an executive-quality notebook report with composite-score distribution and HITL threshold sensitivity;
5. strengthen HITL routing guidance without hard-coding an unvalidated business threshold.

Do not add BIC logic yet. Design the reference layer so BIC/SWIFTRef can be integrated later without rewriting the pipeline.

---

## 1. Add `predicted_country_name_group_<id>`

### Requirement

Add one new generated output column per enabled group:

`predicted_country_name_group_<id>`

Place it immediately after:

`predicted_country_group_<id>`

This changes the generated output count from **11 to 12 columns per group**.

### Important design rule

Country Name must **not** be generated independently by Gemini.

The source of truth is the deterministic ISO/reference layer:

- If `predicted_country_group_<id> == "US"` → `predicted_country_name_group_<id> == "United States"`
- If the country code result is ambiguous and contains comma-separated codes, preserve exact code/name alignment:
  - `predicted_country_group_<id> = "CA,US"`
  - `predicted_country_name_group_<id> = "Canada,United States"`
- If no defensible country exists:
  - `predicted_country_group_<id> = "NO_COUNTRY"`
  - `predicted_country_name_group_<id> = "NO_COUNTRY"`

Do not ask Gemini for a second country-name prediction.

### Implementation touch points

Inspect and minimally update:

- `src/swift_address/settings.py`
  - add `predicted_country_name` to `OUTPUT_FIELD_KEYS`
  - verify `fields_per_group` becomes dynamic = 12
- `config/config.yaml`
- `config/config.example.yaml`
  - add canonical and legacy templates
- `src/swift_address/reference_data.py`
  - expose a deterministic code → country-name method from the ISO provider
- `src/swift_address/scoring.py`
  - extend `VerifiedExtraction` with `country_name_value`
  - derive it after ISO candidate validation and deterministic candidate ordering
- `src/swift_address/pipeline.py`
  - update `_row_values`, audit output, dtype handling, null/error defaults
- tests, README, architecture and any schema-count assertions

Do not hard-code 12 anywhere except documentation/tests that intentionally assert the current contract.

---

## 2. Explicit `outputs/` folder

The current code already writes paths under `outputs/` and `io.py` creates parent directories. Preserve that behavior, but make the repository structure explicit.

Desired runtime structure:

```text
outputs/
├── README.md
├── phase1_output.csv
├── processing_errors.csv
├── run_metrics.json
├── address_cache.jsonl
├── reports/
│   ├── executive_summary.json
│   ├── score_distribution.csv
│   ├── threshold_sensitivity.csv
│   └── scenario_distribution.csv
└── charts/
    └── composite_score_histogram.png
```

Generated files must remain ignored by Git.

Change `.gitignore` from ignoring the whole folder to:

```gitignore
outputs/*
!outputs/README.md
```

Commit only `outputs/README.md`, explaining that the directory contains generated artifacts and can contain sensitive address/cache data.

All writer functions must still use `Path(...).parent.mkdir(parents=True, exist_ok=True)` so a deleted folder is safely recreated.

Add configurable report paths to YAML under `processing` or a new `reporting` section. Prefer a typed `ReportingConfig`.

---

## 3. Add Town↔Country reference data

The repository currently has an ISO country provider. Add a **development-only** Town↔Country reference capability.

Use this file path by default:

`data/reference/town_country_reference.csv`

Expected columns:

```text
town_name
town_name_normalized
country_code
country_name
source_dataset
source_version
source_url
approved_for_production
```

A development reference file is supplied as `data/reference/town_country_reference.csv`. It contains GeoNames-derived Town/Country rows for locations with population >= 1,000, and it is explicitly NOT approved for production. Also include/retain a utility script such as:

`scripts/build_geonames_town_country_reference.py`

that can create a much larger file from the GeoNames `cities500.zip` dump plus `countryInfo.txt`.

### Provider behavior

Implement something like `TownCountryProvider` / `TownCountryIndex` with efficient in-memory indexing:

```python
lookup_country_codes("AUCKLAND") -> ("NZ",)
lookup_country_codes("HAMILTON") -> ("BM", "CA", "NZ")
```

Requirements:

- normalize Unicode/case/whitespace deterministically;
- never scan the whole CSV for each address;
- build an index once at startup;
- return distinct sorted country codes;
- retain provenance;
- expose record count and source/version in `run_metrics.json`;
- mark the supplied public development dataset `approved_for_production=False`;
- enterprise replacement must be possible by changing config, not code.

### How it should affect extraction/validation

Do **not** let this provider blindly overwrite Gemini.

Use it primarily as a post-extraction validation/disambiguation signal:

1. Gemini proposes Town and candidate Country code(s).
2. Python validates Town presence and ISO codes as it already does.
3. TownCountry reference lookup is run for the predicted normalized Town.
4. Record the reference candidate countries in audit/metrics.
5. If the Town maps to exactly one country and it agrees with the model, mark reference validation as consistent.
6. If the Town maps to multiple countries and the input has insufficient explicit country evidence to select one:
   - preserve the candidate codes as comma-separated values;
   - force `predicted_country_probability = 0.0`;
   - force Country policy weight = `0.0`;
   - force Composite Weighted Score = `0.0`;
   - mandatory HITL.
7. If model result conflicts with deterministic reference data, do not silently replace it. Route to HITL and record a validation/conflict note.
8. If the Town is absent from the development reference, treat it as `reference_not_found`, not as a model failure.

Do not add Town/Country reference fields as extra production CSV columns unless they are part of the agreed output contract. Keep detailed validation status in audit and run metrics.

---

## 4. Executive report in the notebook

Enhance:

`notebooks/01_phase1_address_extraction.ipynb`

The notebook should remain an orchestration/reporting layer. Put reusable reporting logic in:

`src/swift_address/reporting.py`

### A. Executive KPI section

After the run, display a compact executive summary with at least:

- input records
- enabled address groups
- total group/address instances
- null group instances skipped
- non-empty address instances
- unique addresses sent/eligible for Gemini
- backend Gemini calls
- cache hits
- extraction errors
- ambiguous-country instances
- HITL instances
- HITL %
- auto-accept candidate instances
- auto-accept candidate %
- current configured HITL threshold

Clearly distinguish:
- **record count**
- **address-group instance count**
- **unique-address count**

Do not mix these denominators.

### B. Composite Weighted Score distribution

Primary operational distribution should use **non-empty address-group instances** because HITL workload occurs at the instance level.

Use these exact bins:

```text
< 0.60
0.60 - <0.65
0.65 - <0.70
0.70 - <0.75
0.75 - <0.80
0.80 - <0.85
0.85 - <0.90
0.90 - <0.95
0.95 - 1.00
```

For each bin show:

- score_band
- observation_count
- percent_of_non_empty
- cumulative_count
- cumulative_percent

Boundary rules must be deterministic:
- 0.60 belongs to `0.60 - <0.65`
- 0.65 belongs to `0.65 - <0.70`
- 1.00 belongs to `0.95 - 1.00`

Null/empty group instances should **not** pollute the probability histogram. Report them separately.

Create:
- notebook table
- histogram/bar chart using the same bins
- `outputs/reports/score_distribution.csv`
- `outputs/charts/composite_score_histogram.png`

Add `matplotlib` to requirements if needed. Keep plotting code in `reporting.py`.

### C. Scenario distribution

Create a table by deterministic policy scenario:

- scenario
- observation_count
- percent_of_non_empty

Save to:

`outputs/reports/scenario_distribution.csv`

### D. HITL threshold sensitivity

Do not select a threshold from intuition alone.

Create a sensitivity table for:

```text
0.80
0.85
0.90
0.95
```

For each threshold calculate:

- threshold
- auto_accept_candidate_count
- auto_accept_candidate_percent
- hitl_count
- hitl_percent
- ambiguous_forced_hitl_count
- error_forced_hitl_count

Important: ambiguity/errors remain HITL regardless of numeric score.

Save:

`outputs/reports/threshold_sensitivity.csv`

Display it prominently in the notebook.

### E. Provisional recommendation

Until labeled ground truth is available, label the threshold as **provisional routing policy**, not calibrated accuracy.

For Phase 1, show **0.90 as the recommended conservative starting point**, but do not hard-code it in reporting logic. Explain:

- because the Composite Weighted Score already penalizes inferred values;
- any unresolved country ambiguity has score 0;
- a score of 0.90 in `both_explicit` roughly requires both model confidences to be about 0.95 if they are similar.

However, the final enterprise threshold must be chosen from labeled validation data.

Once labels are available, extend the sensitivity report with:
- precision of auto-accepted population
- error rate of auto-accepted population
- recall/coverage
- HITL volume
- target precision recommendation

A suitable future governance criterion is to select the **lowest threshold that satisfies the business-approved minimum precision**, rather than maximizing throughput.

---

## 5. Executive summary artifact

In addition to notebook display, write:

`outputs/reports/executive_summary.json`

Suggested shape:

```json
{
  "run_timestamp": "...",
  "model": "...",
  "prompt_version": "...",
  "reference_data_version": "...",
  "hitl_threshold": 0.90,
  "records": 0,
  "non_empty_address_instances": 0,
  "unique_addresses": 0,
  "auto_accept_candidates": 0,
  "auto_accept_candidate_percent": 0.0,
  "hitl_instances": 0,
  "hitl_percent": 0.0,
  "ambiguous_country_instances": 0,
  "extraction_errors": 0
}
```

No raw customer address should appear in this executive artifact.

---

## 6. Keep Composite Weighted Score logic unchanged

The scoring formula remains:

```text
adjusted_town_score
    = predicted_town_probability * town_reliability_weight

adjusted_country_score
    = predicted_country_probability * country_reliability_weight

composite_weighted_score
    = adjusted_town_score * adjusted_country_score
```

Existing weight matrix remains:

```yaml
both_explicit:
  town_weight: 1.00
  country_weight: 1.00

country_explicit_town_inferred:
  town_weight: 0.50
  country_weight: 1.00

town_explicit_country_inferred:
  town_weight: 0.75
  country_weight: 0.50

town_explicit_country_ambiguous:
  town_weight: 0.50
  country_weight: 0.00

neither_explicit_both_inferred:
  town_weight: 0.20
  country_weight: 0.20

no_defensible_prediction:
  town_weight: 0.00
  country_weight: 0.00

town_inferred_country_ambiguous:
  town_weight: 0.20
  country_weight: 0.00
```

Country ambiguity continues to override Country probability to `0.0`.

Gemini must not calculate the Composite Weighted Score.

---

## 7. BIC/SWIFTRef — prepare, do not implement

The user has BIC codes and will add them later.

Do not add BIC extraction/lookup now.

Only ensure the architecture can later support another reference provider such as:

```text
BIC / SWIFTRef
      ↓
authoritative institution address
      ↓
Town / Country validation
```

Keep the existing SWIFTRef stub honest: no fake lookup, no scraping, no claim that the model used SWIFTRef unless approved data was actually supplied.

---

## 8. Tests / acceptance criteria

Add or update tests for all of the following:

1. `fields_per_group == 12`.
2. Country code → expanded Country Name is deterministic.
3. Ambiguous code list and Country Name list remain aligned.
4. `NO_COUNTRY` produces `NO_COUNTRY` name.
5. Output keeps original input columns unchanged and in original order.
6. `outputs/` parents are created if absent.
7. Score bin boundaries:
   - 0.5999 → `< 0.60`
   - 0.6000 → `0.60 - <0.65`
   - 0.6500 → `0.65 - <0.70`
   - 0.9500 → `0.95 - 1.00`
   - 1.0000 → `0.95 - 1.00`
8. Null group instances excluded from histogram denominator.
9. Threshold sensitivity counts ambiguity as forced HITL.
10. Reference Town with one country validates correctly.
11. Same Town mapping to several countries is preserved as an ambiguous candidate set when not disambiguated by explicit address evidence.
12. Reference conflict does not silently overwrite model output.
13. No reference match is not treated as an extraction error.
14. Existing `AERONAUTICA` anti-substring test continues to pass.
15. All existing tests continue to pass.

Run:

```bash
pytest -q
```

and execute the notebook in dry-run/mock mode to prove the reporting section works without Gemini credentials.

---

## 9. Documentation

Update:

- `README.md`
- `architecture.md`
- `SCORING_SPEC.md` only if needed to explain threshold reporting
- `data/reference/PROVENANCE.md`
- `outputs/README.md`

Document that:
- Country Name is deterministic reference-derived output;
- Town/Country reference data is development-only until replaced by an enterprise-approved source;
- GeoNames source requires attribution;
- HITL threshold is configurable;
- 0.90 is a provisional conservative starting recommendation, not a calibrated guarantee;
- final threshold should be chosen from labeled validation results.

---

## 10. Deliverables

When finished, report:

1. files changed/added;
2. new output schema and total fields per group;
3. reference-data design;
4. executive-report outputs;
5. threshold sensitivity result from the sample data;
6. all test results;
7. any assumptions or unresolved issues.

Prefer minimal, modular changes over broad refactoring.
