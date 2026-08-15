# SWIFT Address Town/Country Extraction — Phase 1 Starter v2

## Objective
Build a configurable, auditable Phase 1 pipeline that reads a CSV containing `RECORD_ID` plus up to 49 additional source columns, groups multi-line address fields using a configuration file, skips empty addresses before any LLM call, extracts/infers Town and Country with Gemini, computes policy-driven confidence scores, and writes an expanded CSV suitable for Human-in-the-Loop (HITL) review.

This starter is intentionally notebook-first, but the implementation should place reusable logic in Python modules so the same code can be used in Phase 2 without rewriting the pipeline.

## Important corrections/clarifications

1. The screenshots and source schema imply **50 input columns**: `RECORD_ID` + 16 address groups × 3 lines + `OTHER`.
2. The screenshot group-config shows 15 groups. The input-schema screenshot includes one additional unmapped group: `PRI_SNDR_CORR_ADDR_LINE_1..3`. The sample configuration therefore keeps the screenshot mappings as groups 1–15 and adds `PRI_SNDR_CORR` as **group16**. This should be confirmed against the authoritative project config before production use.
3. There are **11 requested output columns per group**. Therefore 16 × 11 = **176 appended columns**, and a 50-column input becomes **226 columns**, not 236. The code must calculate this dynamically and never hard-code 226.
4. The user-supplied field names contained likely spelling typos (`comined`, `countrty`, `rational`). The new implementation should use canonical spellings (`combined`, `country`, `rationale`) by default. A naming-template config should make legacy names possible if downstream compatibility requires them.

## Phase 1 scope

- Input: CSV.
- Group configuration: CSV or YAML; the supplied sample uses CSV.
- Output: CSV preserving every input column and appending 11 columns for each configured group.
- Notebook: `notebooks/01_phase1_address_extraction.ipynb`.
- Gemini model: configurable environment variable; default `gemini-3.5-flash`.
- Credentials: environment variables only; never hard-code secrets in notebooks, source code, YAML, or committed files.
- Structured JSON output from Gemini.
- Deterministic preprocessing and scoring around the LLM.
- No model call for a null/empty combined address.
- Caching/deduplication so identical addresses are not sent repeatedly.
- Retry/backoff for transient errors and 429 responses.
- Sidecar processing-error log rather than silently converting failed LLM calls to valid predictions.

## Required output columns per configured group

For `group15`, the canonical names are:

1. `combined_address_group_15`
2. `combined_address_cleaned_group_15`
3. `predicted_town_group_15`
4. `predicted_country_group_15`
5. `predicted_town_probability_group_15`
6. `predicted_country_probability_group_15`
7. `predicted_town_exists_group_15`
8. `predicted_country_exists_group_15`
9. `composite_weighted_score_group_15`
10. `rationale_town_group_15`
11. `rationale_country_group_15`

The same template is generated for every configured group. Do not hard-code group IDs.

## Null-address first pass

Before calling Gemini, build each combined address from the configured source columns.

Treat the following as missing:

- `None` / `NaN`
- empty or whitespace-only strings
- a field whose entire trimmed value is exactly `0`

Do **not** delete the digit `0` when it is part of a legitimate address, postal code, building number, or other text.

Concatenate only the non-missing address lines, in config order, using a single space. Preserve the original source columns unchanged.

If the resulting combined address is empty, populate:

- `predicted_town_* = NO_TOWN`
- `predicted_country_* = NO_COUNTRY`
- numeric score fields = `0.0`
- boolean `*_exists_*` fields = `False`
- string rationale fields = empty string
- combined/cleaned address fields = empty string

and **do not call Gemini**.

## Address cleaning

`combined_address_cleaned_*` must be deterministic and conservative. Phase 1 cleaning should:

- Unicode-normalize (NFKC)
- trim leading/trailing whitespace
- collapse repeated whitespace
- normalize line separators to spaces
- remove only empty fields and fields equal to `0`

Do not semantically rewrite the address, invent missing location data, or remove digits/postal codes.

## Gemini responsibilities

Gemini should determine the most defensible Town and Country candidate set from a non-empty cleaned address. Country candidates must be ISO 3166-1 alpha-2 uppercase codes (`US`, `PE`, `GH`, `NZ`, `TW`, etc.). If exactly one Country is defensible, Python writes that code. If multiple Countries remain unresolved, Python writes a deterministic comma-separated list and forces final Country probability to `0.0`. If no Country is defensible, Python writes `NO_COUNTRY`. Use `NO_TOWN` when Town cannot be defensibly determined.

Gemini should provide evidence-oriented rationales, but **the LLM must not claim it consulted SWIFTRef unless the program actually supplied SWIFTRef lookup data to the request**.

The program should request internal evidence fields (for example the text span supporting Town/Country and ambiguity indicators) even if those fields are not written to the final CSV. Python should use those fields to verify `town_exists` and `country_exists` wherever possible.

## Reference-data policy

### SWIFTRef
SWIFTRef is licensed reference data. Its BIC Directory contains BIC8/BIC11 registered names and addresses. Production code should access SWIFTRef only through an approved/entitled API or approved local directory file. Do not ask Gemini to "browse SWIFTRef" without such access.

Provide a `ReferenceDataProvider` abstraction with a no-op provider in Phase 1 and a `SwiftRefProvider` implementation point for Phase 2.

### ISO country data
Country codes should be validated against an approved ISO 3166-1 reference source. Prefer a locally approved/reference-managed dataset in production. A development fallback may be configurable, but the code should make the provenance visible.

## Scoring design

The two `predicted_*_probability_*` fields are model-confidence inputs in `[0,1]`. The scenario values below are **reliability weights**, not probabilities. Gemini does not calculate the final HITL score; Python applies the weights deterministically.

| Scenario | Town weight | Country weight |
|---|---:|---:|
| `both_explicit` | 1.00 | 1.00 |
| `country_explicit_town_inferred` | 0.50 | 1.00 |
| `town_explicit_country_inferred` | 0.75 | 0.50 |
| `town_explicit_country_ambiguous` | 0.50 | **0.00** |
| `neither_explicit_both_inferred` | 0.20 | 0.20 |
| `no_defensible_prediction` | 0.00 | 0.00 |

The Composite Weighted Score is:

```text
adjusted_town_score    = town_probability × town_weight
adjusted_country_score = country_probability × country_weight

composite_weighted_score = adjusted_town_score × adjusted_country_score
```

Equivalent:

```text
composite_weighted_score =
    (town_probability × town_weight)
    ×
    (country_probability × country_weight)
```

This is an operational routing score, not a statistically calibrated joint probability. Cross-entropy/BCE belongs in later offline evaluation against labeled ground truth, not in Phase 1 HITL routing.

### Unresolved multiple-country rule

If a Town can map to multiple countries and the address plus approved reference context cannot resolve one uniquely:

- keep all defensible ISO 3166-1 alpha-2 candidates as a deterministic comma-separated value such as `CA,US`;
- set `predicted_country_probability_group_<id> = 0.0`;
- use Country reliability weight `0.0`;
- set `composite_weighted_score_group_<id> = 0.0`;
- route the record to HITL;
- never arbitrarily choose one country to make the result scalar.

The raw Gemini candidate-set confidence may be retained in cache/debug metadata, but the production Country probability used by the scoring engine is zero whenever more than one country remains unresolved.

## HITL

`composite_weighted_score_*` is the routing signal. Keep `hitl_threshold` configurable. Do not permanently choose a threshold based only on intuition. After collecting labeled data, measure auto-accepted precision/recall and calibrate the threshold to the required operational risk level.

A suggested development default is `0.80`, but it is a placeholder only. Any unresolved multiple-country case is mandatory HITL regardless of threshold.

## Token/cost controls

The implementation must be economical:

1. First-pass skip of null combined addresses.
2. Deduplicate identical cleaned addresses across all rows and all groups.
3. Cache model results by a stable hash of `(prompt_version, model, cleaned_address, reference_context_version)`.
4. Process only unique, non-empty addresses.
5. Configurable batch size/concurrency.
6. Exponential backoff with jitter for 429/5xx.
7. Checkpoint results periodically so a restart does not repeat completed calls.
8. Keep rationales concise (1–3 sentences) and cap output tokens.

## Minimum acceptance examples

- `1 LINCOLN STREET BOSTON MA 02111 US` → `BOSTON`, `US`; both explicitly supported.
- `441-445 JIRON SANTA ROSA LIMA METRO MUNIC OF LIMA 15001` → `LIMA`, likely `PE`; Town explicit, Country inferred unless reference evidence provides PE.
- `388 GREENWICH STREET NEW YORK NY 10013-2632 US` → `NEW YORK`, `US`; both explicit.
- `25A CASTLE ROAD AMBASSADORIAL AREA ACCRA GREATER ACCRA GH` → `ACCRA`, `GH`; both explicit.
- `23 CUSTOMS STREET EAST LEVEL 11 CITIGROUP CENTRE AUCKLAND AUCKLAND 1140 NZ` → `AUCKLAND`, `NZ`; both explicit.
- `TAIPEI HEAD OFFICE` → `TAIPEI`, `TW` only if the country inference is sufficiently supported; Town is explicit, Country is inferred.
- `AERONAUTICA` → do **not** infer `RONA` from a substring. Return `NO_TOWN` / `NO_COUNTRY` unless approved reference context independently establishes a location.
- controlled ambiguous-country fixture → return comma-separated ISO candidates (for example `CA,US`), final Country probability `0.0`, Composite Weighted Score `0.0`, and HITL.
- empty or `0`-only group → no model call; `NO_TOWN`, `NO_COUNTRY`, zero scores.

## Verified platform notes (August 2026)

- Google lists `gemini-3.5-flash` as a GA model.
- The Google Gen AI SDK supports structured JSON output with JSON Schema.
- The Gemini API client can read `GEMINI_API_KEY` or `GOOGLE_API_KEY` from environment variables.
- Google Search grounding uses publicly available web data. It is not a substitute for licensed SWIFTRef access.

References:
- https://ai.google.dev/gemini-api/docs/changelog
- https://ai.google.dev/gemini-api/docs/structured-output
- https://ai.google.dev/gemini-api/docs/api-key
- https://www.swift.com/products/swiftref
- https://www.swift.com/products/swiftref-bic-directory
- https://www.iso.org/iso-3166-country-codes.html

## Starter artifacts

- `architecture.md` — component and processing design.
- `SCORING_SPEC.md` — Composite Weighted Score and ambiguity rules.
- `CLAUDE_CODE_PROMPT.md` — master prompt to paste into Claude Code.
- `prompts/GEMINI_EXTRACTION_PROMPT.md` — prompt contract for Gemini.
- `config/group_config.csv` — 16-group sample configuration reconstructed from the screenshots/input schema.
- `config/config.example.yaml` — runtime/scoring/naming configuration.
- `.env.example` — environment variable names only.
- `data/sample_input.csv` — small 50-column sample input reconstructed from screenshots.
- `swift_project_reference.xlsx` — workbook version of group config, input schema, sample input, output schema, scoring rules, and expected examples.
