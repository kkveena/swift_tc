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

## Running the pipeline

### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dry run (no credentials, no network)

With no Gemini credentials in the environment, the pipeline uses an offline stub and still
exercises every stage — grouping, cleaning, null skip, dedupe, verification, scoring, output.

```bash
jupyter notebook notebooks/01_phase1_address_extraction.ipynb
```

or non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_phase1_address_extraction.ipynb
```

The stub is not an extraction model. Its country comes only from ISO codes the program can
actually find in the address text, its town from a tiny demo list, and every rationale it writes
says so. Run metrics record `"mode": "dry_run"`.

### Live run against Gemini

```bash
cp .env.example .env          # then fill in .env (git-ignored), or export directly
export GEMINI_API_KEY=...     # never committed, never logged, never printed
export GEMINI_MODEL=gemini-3.5-flash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_phase1_address_extraction.ipynb
```

For enterprise Vertex AI, set `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT` and
`GOOGLE_CLOUD_LOCATION` instead of an API key. The client is constructed with no arguments and
reads its configuration from the environment, so this is a deployment change, not a code change.

Set `SWIFT_ADDRESS_DRY_RUN=true` to force the offline stub even when credentials are present.

### From Python

```python
from swift_address.grouping import load_group_config
from swift_address.pipeline import run_phase1
from swift_address.reference_data import build_provider, find_iso_provider
from swift_address.schemas import load_prompt_contract
from swift_address.settings import load_config, resolve_model_name
from swift_address.gemini_client import build_client

config = load_config("config/config.yaml")
group_config = load_group_config(config.path(config.project.group_config_path))
provider = build_provider(config.reference_data, base_dir=config.base_dir)
prompt = load_prompt_contract(config.path(config.project.prompt_path),
                              config.project.prompt_version)
client = build_client(config.model, prompt, model=resolve_model_name(config),
                      iso_provider=find_iso_provider(provider), dry_run=False)

result = run_phase1("data/sample_input.csv", config, group_config,
                    client=client, reference_provider=provider, prompt=prompt)
```

### Tests

```bash
python -m pytest          # 222 tests
```

### Outputs

| Path | Contents |
|---|---|
| `outputs/phase1_output.csv` | every input column unchanged + 11 columns per enabled group |
| `outputs/processing_errors.csv` | one row per failed unique address; addresses referenced by hash |
| `outputs/run_metrics.json` | shape, null-skip and dedupe savings, cache/call counts, scenario counts, HITL counts, reference-data provenance |
| `outputs/address_cache.jsonl` | cached extractions + audit metadata (**contains raw addresses; git-ignored**) |

## Implementation decisions

Decisions taken where the starter artifacts were silent or inconsistent. Each is configuration
plus documentation rather than an assumption buried in code.

1. **`config/config.yaml`** is the file the pipeline loads; `config.example.yaml` is left untouched
   as the supplied sample. Every key added beyond the example is marked `[EXTENSION]` in place.
   Override the path with `SWIFT_ADDRESS_CONFIG`.
2. **`.env.example` and `data/reference/iso3166.csv` were referenced but absent** from the starter
   artifacts, and have been created. The ISO dataset is a *development fallback*; its status is
   recorded in `data/reference/PROVENANCE.md`, reported by `Iso3166Provider.provenance`, and written
   into `run_metrics.json` on every run, so no run can quietly claim approved reference data.
3. **`*_exists` is decided by the text, not by the model.** The predicted value is matched against
   the address on token boundaries (`AERONAUTICA` does not contain the token `RONA`). A model claim
   of explicit support that the text cannot carry becomes `False`, and the disagreement is recorded
   in the audit notes. The reverse is also recorded: a value the model called "inferred" that is
   literally present is marked `True` with a `town_present_though_model_marked_inferred` note.
4. **Colliding two-letter codes.** A bare alpha-2 token is not accepted as explicit country evidence
   when the code collides with ordinary address vocabulary (`IN`, `IT`, `NO`, `ME`, `MA`, …) unless
   it sits in trailing country position. Without this, "SUITE 5 IN TOWER" would prove India, and the
   US state abbreviation in "BOSTON MA 02111 US" would prove Morocco. The collision list and the
   trailing window are in `reference_data` config.
5. **A seventh scoring scenario.** `SCORING_SPEC.md`'s six scenarios have no row for *country
   ambiguous **and** town not explicitly supported*. Reusing `town_explicit_country_ambiguous` there
   would assert a verified explicitness that does not exist, so `town_inferred_country_ambiguous`
   (0.20 / 0.00) is configured separately. The composite is `0.0` either way — the distinct name
   only keeps the audit trail truthful. Removing it from config falls back to
   `no_defensible_prediction`.
6. **Partial predictions get no partial credit.** A defensible town with no defensible country (or
   vice versa) maps to `no_defensible_prediction`. One factor of the product is missing, so the
   composite is `0.0` regardless of the weights chosen.
7. **Text-resolved ambiguity.** When several candidates are returned but exactly *one* is explicitly
   present in the address, the text has resolved the choice and the result collapses to that code,
   with a note. Two candidates that are *both* named in the text stay ambiguous — that is still an
   unresolved choice, and the pipeline never picks one.
8. **Ambiguity zeroing is policy, not a YAML value.** More than one surviving candidate forces
   country probability *and* country weight to `0.0` in code, so editing the weight matrix cannot
   turn an unresolved multi-country result into a passing score.
9. **Failures are never conclusions.** An exhausted retry writes a row to `processing_errors.csv`,
   leaves the dataframe row intact with safe neutral values, marks scenario `extraction_error`,
   forces HITL, and leaves the rationale fields empty. `NO_TOWN` from the null path and `NO_TOWN`
   from a failed call are distinguishable in the metrics and the audit trail.
10. **Retry asymmetry.** Transient transport failures (429, 5xx, timeouts, connection resets) retry
    with bounded exponential backoff and jitter. Malformed *business* output is retried once, then
    recorded as an error.
11. **Canonical field names by default.** The typo'd names from the source screenshots
    (`comined_`, `countrty`, `rational_`) are available as `output.naming_style: "legacy"`. Exactly
    one naming set is ever emitted; a test asserts the two never appear together.
12. **Group 16 (`PRI_SNDR_CORR`) remains provisional**, as flagged in the starter notes. It is
    enabled in `config/group_config.csv` and must be confirmed against the authoritative project
    config before production use. Disabling it changes the column arithmetic automatically —
    nothing in the code knows the number 16.
13. **Public web-search grounding is refused in code**, not merely defaulted off: constructing the
    Gemini client with grounding enabled raises. This data may contain customer and payment
    addresses.
14. **Logs reference addresses by SHA-256**, never in plaintext, unless
    `SWIFT_ADDRESS_ALLOW_RAW_LOGS=true` is set in an approved debugging environment. The cache and
    the audit payload hold the raw text because they *are* the audit record; `outputs/` is
    git-ignored for that reason.

## Repository layout

```text
config/config.yaml                     runtime config (loaded); config.example.yaml is the sample
config/group_config.csv                16 groups x 3 lines; any N groups / N lines is supported
data/reference/iso3166.csv             ISO 3166-1 development fallback (see PROVENANCE.md)
prompts/GEMINI_EXTRACTION_PROMPT.md    single source of the prompt text
notebooks/01_phase1_address_extraction.ipynb
src/swift_address/                     settings, io, grouping, cleaning, schemas,
                                       reference_data, gemini_client, scoring, cache, pipeline
tests/                                 222 tests covering the acceptance criteria
outputs/                               generated; git-ignored
```

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
