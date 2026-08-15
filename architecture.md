# Architecture — SWIFT Address Town/Country Extraction Phase 1 v2

## 1. Design goals

The solution should be deterministic where possible, use Gemini only where semantic inference is needed, preserve all source data, minimize token usage, remain auditable, and be configurable enough to become a Phase 2 production pipeline without rewriting the core logic.

## 2. Logical architecture

```text
Input CSV (all source columns retained)
        |
        v
Schema + Group Config Validator
        |
        v
Pass 1: Address Group Builder
  - read configured 1..N address fields per group
  - ignore null/blank/field == "0"
  - concatenate in configured order
  - deterministic cleaning
  - initialize 11 output columns/group
        |
        +------------------------------+
        | combined address empty?      |
        | YES                          | NO
        v                              v
Populate NO_TOWN / NO_COUNTRY      Unique-address dedupe
scores=0, exists=False                 |
NO LLM CALL                            v
                                  Reference Enrichment
                                  - ISO 3166 validation context
                                  - optional SWIFTRef provider
                                  - no fabricated access
                                        |
                                        v
                                  Gemini 3.5 Flash
                                  Structured JSON only
                                        |
                                        v
                                  Response Validator
                                  - schema validation
                                  - country alpha-2 validation
                                  - evidence/presence checks
                                  - reject substring hallucinations
                                        |
                                        v
                                  Policy Scoring Engine
                                  - configurable rule matrix
                                  - combined method configurable
                                  - default product
                                        |
                                        v
                                  Result Cache
                                        |
                                        v
Map unique results back to every row/group
        |
        v
HITL Flag Logic (internal/sidecar or future column)
        |
        v
Expanded Output CSV + processing_errors.csv + run metrics
```

## 3. Recommended repository structure

```text
swift-address-extraction/
├── README.md
├── architecture.md
├── CLAUDE.md                      # optional project instructions for Claude Code
├── .env.example
├── requirements.txt
├── config/
│   ├── group_config.csv
│   └── config.yaml
├── prompts/
│   └── gemini_address_extraction.md
├── data/
│   ├── sample_input.csv
│   └── reference/
│       └── iso3166.csv            # approved source in real environment
├── notebooks/
│   └── 01_phase1_address_extraction.ipynb
├── src/
│   └── swift_address/
│       ├── __init__.py
│       ├── settings.py
│       ├── io.py
│       ├── grouping.py
│       ├── cleaning.py
│       ├── schemas.py
│       ├── reference_data.py
│       ├── gemini_client.py
│       ├── scoring.py
│       ├── cache.py
│       └── pipeline.py
├── tests/
│   ├── test_grouping.py
│   ├── test_cleaning.py
│   ├── test_scoring.py
│   ├── test_null_skip.py
│   ├── test_output_schema.py
│   └── test_prompt_contract.py
└── outputs/
```

The notebook should orchestrate and demonstrate the pipeline; it should not contain all business logic.

## 4. Configuration model

### Group configuration
Each row defines a group and ordered source fields. The code must support any number of groups and should not assume exactly 3 address lines, although the current config uses 3.

Recommended schema:

```text
group_id,address_line_1,address_line_2,address_line_3,enabled,notes
```

### Runtime configuration
Keep model, batch, concurrency, scoring, naming templates, HITL threshold, and reference-provider settings in YAML/environment variables.

Secrets must come only from environment variables or enterprise credential mechanisms.

## 5. Dataframe strategy

1. Read CSV with source columns preserved. Read address fields as strings to avoid converting postal codes and identifiers to numbers.
2. Preserve original column order.
3. Validate `RECORD_ID` exists and is not modified.
4. Validate that every enabled config source column exists in the input. Fail fast with a useful report if required columns are missing.
5. Generate output columns group-by-group using naming templates.
6. Never mutate the source address columns.

### Output count
For `G` groups and `K=11` group-output fields:

`final_column_count = input_column_count + G * K`

For 50 input columns and 16 groups: `50 + 16*11 = 226`.

## 6. Pass 1 — deterministic preprocessing

### Missing-value policy
A source address line is missing when it is:

- null/NaN
- empty after trimming
- exactly `"0"` after trimming

Only an entire field equal to `0` is discarded. `10010`, `10 DOWNING STREET`, `LEVEL 10`, etc. remain untouched.

### Combined address
Join non-missing lines with one space. Example:

```text
Line 1 = "23 CUSTOMS STREET EAST LEVEL 11"
Line 2 = "CITIGROUP CENTRE AUCKLAND AUCKLAND"
Line 3 = "1140 NZ"

combined = "23 CUSTOMS STREET EAST LEVEL 11 CITIGROUP CENTRE AUCKLAND AUCKLAND 1140 NZ"
```

### Cleaned address
Use deterministic normalization only. No LLM call.

### Empty combined address
Initialize the required outputs immediately and add no item to the LLM work queue.

## 7. Pass 2 — unique-address LLM processing

Create a work table of unique, non-empty `combined_address_cleaned` values across all rows and all groups.

Recommended unique key:

```text
sha256(prompt_version | model | normalized_address | reference_context_version)
```

This allows repeated institutions/addresses to be processed once and mapped back to every occurrence.

## 8. Gemini contract

Use the Google Gen AI SDK and structured output. Model ID comes from `GEMINI_MODEL`, default `gemini-3.5-flash`.

The model should return an internal schema like:

```json
{
  "town": "AUCKLAND",
  "country_candidates": ["NZ"],
  "town_evidence": "AUCKLAND",
  "country_evidence": "NZ",
  "town_is_explicit": true,
  "country_is_explicit": true,
  "country_ambiguous": false,
  "town_ambiguous": false,
  "town_model_confidence": 0.99,
  "country_model_confidence": 0.99,
  "town_rationale": "...",
  "country_rationale": "...",
  "reference_basis": ["input_text"]
}
```

The final `predicted_*_probability_*` fields carry the validated model-confidence inputs used by the scoring engine. The deterministic business reliability weights are applied separately. For unresolved multiple-country output, final Country probability is overridden to `0.0`. Raw unmodified model responses may be retained in cache/debug data for calibration and audit.

## 9. Presence (`*_exists`) validation

`predicted_town_exists` means the predicted Town is explicitly supported by the supplied address text after normalization/alias handling.

`predicted_country_exists` means the predicted Country is explicitly supported by the supplied address text, including an ISO alpha-2 code or a recognized country-name alias.

Prefer deterministic verification:

- normalize text and evidence span
- for Country, map `NZ` ↔ `New Zealand`, `US` ↔ `United States`, etc. using approved ISO/reference mappings
- require the model to provide evidence text when it claims an explicit match
- if the claimed evidence cannot be located/matched, set `exists=False`

This avoids relying only on the model's assertion.

## 10. Reference-data architecture

Create a provider interface:

```text
ReferenceDataProvider
  lookup_address_context(address, optional identifiers) -> ReferenceContext
```

Implementations:

- `NullReferenceDataProvider` — Phase 1 default when no external reference data is provisioned.
- `Iso3166Provider` — validates canonical country names/codes from an approved local dataset.
- `SwiftRefProvider` — future/optional; uses approved SWIFTRef API or licensed directory file.
- `CompositeReferenceDataProvider` — combines approved providers.

Gemini receives reference context generated by the program. It must not independently claim SWIFTRef lookup.

## 11. Scoring architecture

### Why separate extraction from scoring
Gemini supplies Town/Country extraction, evidence, explicitness/ambiguity flags, and model-confidence estimates. Python verifies evidence, normalizes Country candidates, selects a deterministic business scenario, applies configured reliability weights, and computes the final routing score. This prevents prompt behavior from silently changing the business decision rule.

### Model-confidence inputs

```text
town_probability ∈ [0,1]
country_probability ∈ [0,1]
```

For unresolved multiple-country results, the final production `country_probability` is overridden to `0.0`. Raw model metadata may remain in cache/debug data for audit.

### Reliability weights

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
```

### Multiple-country handling
If more than one valid ISO alpha-2 Country remains defensible and the input/reference evidence cannot uniquely resolve one:

```text
predicted_country = comma-separated unique candidates, e.g. "CA,US"
country_ambiguous = True
predicted_country_probability = 0.0
country_weight = 0.0
composite_weighted_score = 0.0
needs_hitl = True
```

Candidate codes must be canonicalized, deduplicated, and deterministically ordered. Do not make a second unconstrained LLM call just to pick one.

### Composite Weighted Score

```text
adjusted_town_score = town_probability × town_weight
adjusted_country_score = country_probability × country_weight
composite_weighted_score = adjusted_town_score × adjusted_country_score
```

Equivalent:

```text
composite_weighted_score =
(town_probability × town_weight) ×
(country_probability × country_weight)
```

Keep scenario, weights, raw model confidences, candidate list, ambiguity flags, model name, and prompt version in cache/debug/audit structures even though the production CSV remains at 11 appended fields per group.

### HITL

```python
needs_hitl = composite_weighted_score < hitl_threshold
if country_ambiguous:
    needs_hitl = True
```

The threshold is configuration and must eventually be calibrated against labeled production-like data. Cross-entropy/BCE can be used later for offline model evaluation but not as the Phase 1 operational routing score.

## 12. Reliability and quota controls

- Set low temperature / deterministic generation configuration where supported.
- Structured output JSON Schema.
- Validate every response before writing to the dataframe.
- Retry only transient statuses with exponential backoff + jitter.
- Configurable request concurrency.
- Cache unique-address results.
- Periodic checkpoint to disk.
- Track call counts, skipped-null counts, cache hits, failures, token usage when available, and elapsed time.
- Generate `processing_errors.csv` with record/group/address hash/error type; do not silently convert API failures into valid `NO_TOWN` results.

## 13. Security / regulated-data considerations

- Do not log raw addresses at INFO level by default.
- Hash addresses in operational logs; allow raw data only in protected debug mode.
- Do not use public Google Search grounding with customer/payment address data unless the enterprise data-governance process explicitly approves it.
- SWIFTRef is licensed data; use only approved credentials/data copies.
- Keep model/reference provenance and prompt version in run metadata.

## 14. Tests that must exist before Phase 1 is considered complete

1. Input columns are preserved exactly.
2. `RECORD_ID` is unchanged.
3. Group config controls group count and field order.
4. Exact string `0` is removed, but digits inside legitimate address data are retained.
5. Empty combined address causes zero LLM calls.
6. 16 groups add exactly 176 columns.
7. Identical cleaned addresses across rows/groups are deduplicated to one model call.
8. `AERONAUTICA` does not become `RONA` by substring hallucination.
9. Country output is ISO alpha-2 or `NO_COUNTRY`.
10. Null model responses, malformed JSON, 429s, and 5xx do not corrupt the dataframe.
11. Scoring rules are unit tested independently of Gemini.
12. Output CSV can be read back with expected row/column count and `RECORD_ID` values.


### Mandatory ambiguous-country test
A controlled fixture that produces more than one valid Country candidate must write a comma-separated list, force Country probability and Country weight to zero, produce Composite Weighted Score zero, and route to HITL.
