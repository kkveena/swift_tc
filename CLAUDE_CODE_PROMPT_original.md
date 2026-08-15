# Master Prompt for Claude Code — SWIFT Address Extraction Phase 1 v2

You are implementing Phase 1 of a regulated financial-services address-structuring pipeline. Read `README.md`, `architecture.md`, `SCORING_SPEC.md`, `config/group_config.csv`, `config/config.example.yaml`, `.env.example`, `data/sample_input.csv`, `prompts/GEMINI_EXTRACTION_PROMPT.md`, and `swift_project_reference.xlsx` before coding.

Do not start by writing a large monolithic notebook. First inspect the artifacts, summarize the implementation plan, identify any contradictions, and then implement the smallest clean modular solution that satisfies the acceptance criteria.

## Goal

Create a notebook-first Python solution that:

1. reads an input CSV containing `RECORD_ID` plus source columns;
2. preserves all input columns and values unchanged;
3. reads an external group configuration that defines N address groups and the ordered source fields in each group;
4. creates one combined address and one cleaned combined address per group;
5. skips Gemini completely when a combined address is empty;
6. processes only unique, non-empty cleaned addresses with Gemini;
7. extracts/infers Town and ISO alpha-2 Country candidate sets using structured JSON output;
8. verifies explicit Town/Country presence as deterministically as possible;
9. preserves unresolved multiple-country ambiguity and verifies evidence;
10. applies deterministic reliability weights and calculates a Composite Weighted Score in Python;
11. populates exactly 11 output columns per group;
12. writes an expanded output CSV plus an error sidecar and basic run metrics;
13. is structured so Phase 2 can reuse the modules without rewriting notebook logic.

## Mandatory first deliverables

Create:

```text
notebooks/01_phase1_address_extraction.ipynb
src/swift_address/__init__.py
src/swift_address/settings.py
src/swift_address/io.py
src/swift_address/grouping.py
src/swift_address/cleaning.py
src/swift_address/schemas.py
src/swift_address/reference_data.py
src/swift_address/gemini_client.py
src/swift_address/scoring.py
src/swift_address/cache.py
src/swift_address/pipeline.py
tests/test_grouping.py
tests/test_cleaning.py
tests/test_null_skip.py
tests/test_scoring.py
tests/test_output_schema.py
tests/test_prompt_contract.py
requirements.txt
```

Update README only where necessary to document actual run commands and implementation decisions. Do not erase the business requirements.

## Technology choices

- Python 3.11+.
- `pandas` for CSV/dataframe processing.
- `pydantic` (or typed dataclasses plus explicit validation) for config and LLM response schemas.
- Google Gen AI SDK: `google-genai`.
- `python-dotenv` allowed for local development, but code must work with environment variables directly.
- `tenacity` is acceptable for bounded retry/backoff.
- Use standard logging.
- Tests with `pytest`.
- Do not introduce LangChain/ADK/agent frameworks in Phase 1; this is a deterministic batch pipeline with one extraction model call, not an agentic workflow.

## Gemini model and credentials

Read from environment:

```text
GEMINI_MODEL=gemini-3.5-flash
GEMINI_API_KEY=...
```

The Google client should be instantiated without embedding a key in source code. Permit enterprise/Vertex/Enterprise configuration through environment variables without requiring code changes.

Never print secrets.

## Input and config handling

### Input
- Input is CSV.
- Preserve input column order.
- Treat all address columns as strings.
- Preserve `RECORD_ID` exactly; do not coerce it to numeric.
- Do not silently drop unknown input columns.

### Group config
The sample group config has 16 groups. However, the implementation must support any number of groups and any number of address lines per group.

Validate:
- unique group IDs;
- enabled groups only;
- no duplicate source fields within a group;
- every configured source field exists in the input;
- useful exception message listing missing columns before any model call.

## Output column naming

Use naming templates from config. Default canonical names for group `{id}`:

```text
combined_address_group_{id}
combined_address_cleaned_group_{id}
predicted_town_group_{id}
predicted_country_group_{id}
predicted_town_probability_group_{id}
predicted_country_probability_group_{id}
predicted_town_exists_group_{id}
predicted_country_exists_group_{id}
composite_weighted_score_group_{id}
rationale_town_group_{id}
rationale_country_group_{id}
```

Exactly 11 columns per enabled group. Preserve a config option for legacy naming aliases if needed, but do not create both sets simultaneously.

For 50 input columns and 16 groups, assert the final dataframe has 226 columns. Calculate dynamically; never hard-code 226.

## Pass 1 — grouping, cleaning, defaults

Implement a pure function that converts configured fields into `combined_address`:

- trim each field;
- ignore null/NaN;
- ignore empty string;
- ignore a field only when its entire trimmed value equals `"0"`;
- retain zeros/digits inside legitimate strings;
- join remaining lines with one space.

Cleaning is deterministic:
- Unicode NFKC;
- collapse repeated whitespace;
- trim;
- no semantic rewriting.

Initialize all 11 output columns before Gemini processing.

If combined address is empty:

```text
combined_address = ""
combined_address_cleaned = ""
predicted_town = "NO_TOWN"
predicted_country = "NO_COUNTRY"
predicted_town_probability = 0.0
predicted_country_probability = 0.0
predicted_town_exists = False
predicted_country_exists = False
composite_weighted_score = 0.0
rationale_town = ""
rationale_country = ""
```

The null path must never enqueue or call Gemini. Add a unit test with a fake Gemini client proving call count remains zero.

## Token and request efficiency

After Pass 1:

1. collect only non-empty cleaned combined addresses;
2. deduplicate across every row and every group;
3. use a stable cache key including prompt version + model + address + reference-context version;
4. process each unique cache miss once;
5. map the result back to all occurrences.

Add configurable:
- concurrency;
- max in-flight requests;
- timeout;
- retry count;
- exponential backoff/jitter;
- checkpoint frequency.

Retry only transient failures (429 and suitable 5xx/network failures). Do not retry malformed business output forever.

## Gemini prompt implementation

Load the prompt from `prompts/GEMINI_EXTRACTION_PROMPT.md`; do not duplicate a large prompt string in multiple modules.

Use structured JSON output with an explicit schema. Parse/validate every model response.

Internal response fields must include at least:

```text
town
country_candidates
town_evidence
country_evidence
town_is_explicit
country_is_explicit
town_ambiguous
country_ambiguous
town_model_confidence
country_model_confidence
town_rationale
country_rationale
reference_basis
```

Use temperature/configuration appropriate for deterministic extraction.

### Critical anti-hallucination rule
`AERONAUTICA` must not produce Town=`RONA` through substring matching. Add this as a unit/integration test fixture.

## Mandatory multiple-country behavior

When a Town maps to multiple plausible Countries and neither address text nor approved reference context resolves one uniquely:

1. return/write all valid ISO alpha-2 candidates as one comma-separated deterministic value;
2. never choose one arbitrarily;
3. force final Country probability to `0.0`;
4. use Country weight `0.0`;
5. Composite Weighted Score must be `0.0`;
6. route to HITL;
7. add a controlled unit-test fixture for this behavior.

The Gemini structured schema should expose `country_candidates: list[str]` rather than requiring one scalar Country. Python converts one candidate to a scalar and multiple candidates to a comma-separated value.

## Reference data

Create a provider abstraction now, even if Phase 1 uses a no-op provider:

```python
class ReferenceDataProvider(Protocol):
    def get_context(self, address: str) -> ReferenceContext: ...
```

Implement:
- `NullReferenceDataProvider`;
- an ISO 3166 validator/provider backed by a configurable local file or approved package/data source;
- a stub/interface point for `SwiftRefProvider`.

Do **not** scrape or pretend to query SWIFTRef. SWIFTRef is licensed and should be used only when approved API/file access is configured.

Do not enable Google Search grounding by default. This project may contain sensitive payment/customer address data and public search is not a substitute for enterprise reference data.

## `exists` calculation

Do not trust the model boolean blindly.

After Gemini predicts Town/Country:

- normalize the address and evidence spans;
- verify that the claimed explicit town evidence exists in the address;
- for country, allow ISO alpha-2 code or normalized country-name aliases from the approved ISO mapping;
- if the model claims explicit=true but evidence cannot be supported, set the final `*_exists` value to False and record the reason in debug/run logs.

The final CSV stores these verified booleans.

## Probability / policy scoring

Treat Gemini confidence and policy reliability as separate concepts.

Gemini returns model-confidence estimates in `[0,1]`. Python verifies/normalizes them into the final Town/Country probability fields. If more than one Country remains unresolved, **force final Country probability to `0.0`**.

The following are configurable **reliability weights**, not probabilities:

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

No magic numbers in Python; load them from YAML.

Deterministically select the scenario from **verified** explicitness and ambiguity. Do not rely only on Gemini's scenario interpretation.

Calculate outside the LLM:

```python
adjusted_town_score = town_probability * town_weight
adjusted_country_score = country_probability * country_weight
composite_weighted_score = adjusted_town_score * adjusted_country_score
```

For unresolved multiple-country output:

```text
predicted_country = comma-separated unique ISO alpha-2 candidates such as "CA,US"
predicted_country_probability = 0.0
country_weight = 0.0
composite_weighted_score = 0.0
```

Retain scenario, weights, candidate list and ambiguity flags in cache/debug/audit data, not as extra production CSV columns.

Cross-entropy/BCE is reserved for later offline evaluation against labeled ground truth.

## HITL

Read `hitl_threshold` from config. Do not hard-code business routing logic into the prompt.

Phase 1 does not need an extra HITL output column unless already requested; produce a run summary showing how many group instances fall below threshold. Keep the main output schema to exactly the requested 11 fields/group.

## Error behavior

On an LLM/API failure after retries:
- do not claim `NO_TOWN/NO_COUNTRY` is a valid model conclusion;
- write a row to `processing_errors.csv` with address hash, occurrences, group IDs, error type/message, model, prompt version;
- preserve the dataframe row;
- use safe neutral output values and make the failure visible in run metrics/checkpoint metadata.

Raw addresses should not be included in default operational logs.

## Notebook requirements

The notebook must be readable for a data-science review and use the modules above. Include cells for:

1. imports/config paths;
2. load + validate config;
3. load sample CSV;
4. show input shape and expected group count;
5. run Pass 1 and report number of empty vs non-empty group instances;
6. report unique non-empty address count (token-saving benefit);
7. optionally run Gemini only when credentials are available; otherwise run in `dry_run`/mock mode;
8. apply scoring;
9. show a focused output table for sample rows;
10. export CSV;
11. display run metrics.

Do not put secrets in notebook cells or outputs.

## Required tests / acceptance criteria

All must pass:

1. `RECORD_ID` preserved exactly.
2. All input columns preserved in original order.
3. Config controls group creation; no group logic is hard-coded.
4. Whole-field `0` omitted; `10013-2632`, `LEVEL 10`, etc. preserved.
5. Empty group results in `NO_TOWN`, `NO_COUNTRY`, zeros/False/blanks and zero Gemini calls.
6. Sample config with 16 groups appends exactly 176 columns.
7. 50-column input yields 226-column output.
8. Duplicate addresses generate one Gemini call per unique cache miss.
9. Structured-response validation catches malformed output.
10. unique Country output is a 2-letter uppercase ISO code; unresolved multiple-country output is a deterministic comma-separated list of valid 2-letter ISO codes; no candidate becomes `NO_COUNTRY`.
11. `AERONAUTICA` is not incorrectly parsed as `RONA`.
12. Scoring rule unit tests cover every policy scenario.
13. Retry test covers 429 then success without duplicate dataframe rows.
14. Output CSV reload preserves row count and required fields.

## Sample semantic expectations

Use the sample artifacts for tests. Expected high-level outcomes:

```text
1 LINCOLN STREET BOSTON MA 02111 US
  -> BOSTON / US; both explicit

441-445 JIRON SANTA ROSA LIMA METRO MUNIC OF LIMA 15001
  -> LIMA; PE may be inferred; Town explicit, Country not explicit

388 GREENWICH STREET NEW YORK NY 10013-2632 US
  -> NEW YORK / US; both explicit

25A CASTLE ROAD AMBASSADORIAL AREA ACCRA GREATER ACCRA GH
  -> ACCRA / GH; both explicit

23 CUSTOMS STREET EAST LEVEL 11 CITIGROUP CENTRE AUCKLAND AUCKLAND 1140 NZ
  -> AUCKLAND / NZ; both explicit

TAIPEI HEAD OFFICE
  -> TAIPEI; TW inferred only if defensible

AERONAUTICA
  -> NO_TOWN / NO_COUNTRY without approved reference context
```

## Quality bar

Before declaring Phase 1 complete:

- run tests;
- run notebook on `data/sample_input.csv`;
- show input/output shapes;
- show null-skip and dedupe call-count metrics;
- show a sample of the generated group15 fields;
- verify there are no accidental extra columns;
- document how to run with real Gemini credentials and how to run mock/dry mode.

If you discover an ambiguity, prefer implementing it as explicit configuration and document the chosen default rather than burying a business assumption in code.
