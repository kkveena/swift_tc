# SWIFT Address Town/Country Extraction — Phase 1 Starter v2

## Objective
Build a configurable, auditable Phase 1 pipeline that reads a CSV containing `RECORD_ID` plus up to 49 additional source columns, groups multi-line address fields using a configuration file, skips empty addresses before any LLM call, extracts/infers Town and Country with Gemini, computes policy-driven confidence scores, and writes an expanded CSV suitable for Human-in-the-Loop (HITL) review.

This starter is intentionally notebook-first, but the implementation should place reusable logic in Python modules so the same code can be used in Phase 2 without rewriting the pipeline.

## Important corrections/clarifications

1. The screenshots and source schema imply **50 input columns**: `RECORD_ID` + 16 address groups × 3 lines + `OTHER`.
2. The screenshot group-config shows 15 groups. The input-schema screenshot includes one additional unmapped group: `PRI_SNDR_CORR_ADDR_LINE_1..3`. The sample configuration therefore keeps the screenshot mappings as groups 1–15 and adds `PRI_SNDR_CORR` as **group16**. This should be confirmed against the authoritative project config before production use.
3. There are **11 requested output columns per group**. Therefore 16 × 11 = **176 appended columns**, and a 50-column input becomes **226 columns**, not 236. The code must calculate this dynamically and never hard-code 226.

   > **Superseded — the arithmetic, not the principle.** Later iterations appended
   > `predicted_country_name`, then the ground-truth, cross-entropy and retraction fields,
   > then the explicit HITL decision fields, reaching **20** fields per group:
   > 16 × 20 = **320 appended columns** and **370** total for a 50-column input. The requirement that the count be *calculated* rather than hard-coded still
   > holds and is enforced by `OUTPUT_FIELD_KEYS`.
4. The user-supplied field names contained likely spelling typos (`comined`, `countrty`, `rational`). The new implementation should use canonical spellings (`combined`, `country`, `rationale`) by default. A naming-template config should make legacy names possible if downstream compatibility requires them.

## Phase 1 scope

- Input: CSV.
- Group configuration: CSV or YAML; the supplied sample uses CSV.
- Output: CSV preserving every input column and appending the configured columns for each group
  (20 today), plus a nested detailed JSONL for audit and evaluation depth.
- Notebooks: `notebooks/swft_tc/01_phase1_address_extraction_DRY_RUN.ipynb` and `..._ACTUAL_RUN.ipynb`.
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
5. `predicted_country_name_group_15`
6. `predicted_town_probability_group_15`
7. `predicted_country_probability_group_15`
8. `predicted_town_exists_group_15`
9. `predicted_country_exists_group_15`
10. `composite_weighted_score_group_15`
11. `rationale_town_group_15`
12. `rationale_country_group_15`

The same template is generated for every configured group. Do not hard-code group IDs.

plus five fields added for ground-truth validation, evaluation, and retraction — **appended**, so no
pre-existing column position shifts:

13. `town_exists_ok_group_15`
14. `country_exists_ok_group_15`
15. `cross_entropy_group_15`
16. `combined_address_retracted_group_15`
17. `combined_address_retracted_group_comments_15`
18. `HITL_flag_group_15`
19. `HITL_state_group_15`
20. `HITL_state_reason_group_15`

`predicted_country_name_*` is **deterministic reference-derived output**, not a second model
prediction: Python expands `predicted_country_*` through the ISO 3166-1 layer. The two columns stay
aligned element for element, so `CA,US` pairs with `Canada,United States`, a single code yields a
single name, and `NO_COUNTRY` yields `NO_COUNTRY`. The response schema has no country-name field for
the model to fill in.

**20 fields per group** → 16 groups append 320 columns, and a 50-column input becomes **370**
columns. As before, this is calculated from `OUTPUT_FIELD_KEYS` and the enabled group count; nothing
in the code hard-codes it.

### `predicted_*_exists` versus `*_exists_ok`

These two families look similar and mean different things. Keeping them apart is what lets the
evaluation metric stay honest.

| Field | Question | Type |
|---|---|---|
| `predicted_town_exists_*` | is the predicted Town **explicitly present in the input text**? | boolean |
| `predicted_country_exists_*` | is the predicted Country explicitly present in the input text? | boolean |
| `town_exists_ok_*` | when independent evidence exists, **was the prediction correct**? | boolean |
| `country_exists_ok_*` | same, for Country | boolean |

The meaning of `predicted_*_exists` is unchanged. `*_exists_ok` is a plain boolean — it is never
blank in the CSV or `null` in the JSON audit:

```text
True   evidence is available and supports the prediction
False  evidence contradicts the prediction, OR is insufficient, unavailable, unresolved, or ambiguous
```

**Unknown collapses into `False`.** The distinction between "contradicted" and "no evidence" still
exists internally (see the audit trail's basis fields), and cross-entropy / correctness-rate
reporting still exclude genuine coverage gaps from the loss rather than counting them as a model
error — only the `*_exists_ok` columns themselves no longer surface a third blank state. Stored as
plain `bool`; JSON `true` / `false`.

The two can legitimately disagree. A Country the model inferred is `predicted_country_exists=False`
(it genuinely is not in the text) while `country_exists_ok=True` (a single-country reference town
confirms it): the address did not state it, but the model was right.

Ground-truth rules, conservative by design:

* **Town `True`** needs the town explicitly present on token boundaries **and** known to the
  Town/Country reference. Neither alone suffices — looking the model's own answer up in a gazetteer
  would be circular ground truth.
* **Town `False`** on positive contradiction (the model asserted an explicit town that
  token-boundary verification proves is absent — the `AERONAUTICA` → `RONA` case), or when there is
  simply no independent evidence to judge the prediction against.
* **Country `True`** when explicitly supported in the address, or when a reference-known town
  resolves to exactly one country and the prediction matches it.
* **Country `False`** when deterministic reference truth contradicts the prediction, or when there is
  no independent evidence — including an unresolved multi-country town.

### Two metrics, opposite directions

| Metric | Question | Direction |
|---|---|---|
| `composite_weighted_score_*` | should this be auto-accepted? | **higher is better** |
| `cross_entropy_*` | did the model's confidence match reality? | **lower is better** |

Cross-entropy is binary log loss of each confidence against the correctness label:

```text
BCE(y, p) = -( y*log(p) + (1-y)*log(1-p) )     p clipped to [1e-6, 1-1e-6]

correct   at p = 0.95  ->  0.051293   (cheap)
incorrect at p = 0.95  ->  2.995732   (expensive)
```

Both labels available → the group value is the **mean** of the two component losses. One label →
that component's loss, status `town_only` / `country_only`. Neither → **blank**, status
`not_available` / `reference_not_found` / `ambiguous_ground_truth`. Missing reference coverage is
never scored as an artificially high loss; the observation is excluded from the metric instead.
Component detail lives in the detailed JSON, not in more CSV columns. Gemini never computes it.

The Composite Weighted Score formula and the reliability-weight matrix are **unchanged**.

### Address retraction

`combined_address_retracted_*` removes from the original address only the Town/Country information
that was actually present and deterministically verified in the source text.

* Town is removed only when `predicted_town_exists` is True.
* Country is removed only when `predicted_country_exists` is True.
* A Country that was *inferred* was never in the text, so nothing is removed — it stays a prediction.

Removal is **token-span based, never substring replacement**, and it happens at the **original
source-column level**: each configured field is processed independently, then the retracted combined
address is rebuilt from the after-values using the same Pass 1 conventions. The original input
columns are never modified.

Safety properties, all tested: `AERONAUTICA` cannot lose `RONA`; `CUSTOMS` cannot lose `US`; an
ambiguous code such as `IN` is removed only where country verification concluded it really meant
India — and that trailing-position judgement is made against the **combined** address, because a
token sitting mid-address can be in the trailing window of its own short field. Repeated standalone
occurrences are all removed (`CITIGROUP CENTRE AUCKLAND AUCKLAND` → `CITIGROUP CENTRE`); postal codes
survive (`1140 NZ` → `1140`); surviving text keeps its original casing.

Example:

```text
before  LINE_1  23 CUSTOMS STREET EAST LEVEL 11
        LINE_2  CITIGROUP CENTRE AUCKLAND AUCKLAND
        LINE_3  1140 NZ

after   LINE_1  23 CUSTOMS STREET EAST LEVEL 11
        LINE_2  CITIGROUP CENTRE
        LINE_3  1140

combined_address_retracted = "23 CUSTOMS STREET EAST LEVEL 11 CITIGROUP CENTRE 1140"
comment = "Retracted Town=AUCKLAND and Country=NZ from verified explicit address evidence."
```

The comment is generated deterministically in Python (never by the model) and is at most three
lines, usually one. Per-column before/after detail lives in the detailed JSON.

### Detailed nested output

The CSV remains the flat compatibility artifact for spreadsheets and downstream flat-file consumers.
Audit and evaluation depth lives in `models/swft_tc/outputs/phase1_detailed_output.jsonl` — one JSON object per
input record, every enabled group nested inside — rather than growing dozens more columns.

```yaml
processing:
  detailed_json_enabled: true
  detailed_json_path: "outputs/phase1_detailed_output.jsonl"
  detailed_json_format: "jsonl"     # "jsonl" (default) | "json" (small dev runs)
```

Per group: `source_fields`, `address`, `prediction`, `text_evidence`,
`ground_truth_validation`, `scoring`, `cross_entropy`, `rationale`, and `retraction` (with
`actual_column_before_retraction` / `actual_column_after_retraction`). Null-skipped groups appear
with `"status": "null_skip"` and a lean body. Run-wide metadata is *not* repeated per group — it
belongs in `run_metrics.json` and `executive_summary.json`.

The writer **streams**: one record is built, serialized, and written before the next is touched, so
peak memory is a single record. Unavailable numerics serialize as `null`; `allow_nan=False` makes
emitting the invalid token `NaN` impossible. The file contains raw address data and is therefore
sensitive — it stays under git-ignored `models/swft_tc/outputs/`.

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

### The explicit HITL decision fields

Three fields per group make the routing decision visible in the CSV:

| Field | Meaning | Type |
|---|---|---|
| `HITL_flag_group_<id>` | final decision — is human review required? | boolean |
| `HITL_state_group_<id>` | primary routing outcome after precedence | closed enum |
| `HITL_state_reason_group_<id>` | one deterministic sentence explaining it | string |

All three are computed in Python from configured policy. **Gemini never chooses the state and never
writes the reason** — the response schema has no HITL field for it to fill in.

Vocabulary, kept distinct because these are four different things:

```text
Composite Weighted Score        the numeric routing score
scoring.hitl_threshold          the configured operational cutoff (0.80)
HITL_flag                       the final human-review decision
HITL_state                      the primary routing outcome / reason category
HITL_state_reason               the deterministic human-readable explanation
forced_review                   did a non-score control override or independently require HITL?
reporting.recommended_threshold an analytical recommendation only (0.90) — never the cutoff
```

#### The threshold is not the only rule

Review is required when the score falls below the threshold **or** when a forced-review control
applies:

```text
HITL required  =  score < threshold  OR  a forced-review condition exists
```

A case with a score comfortably above the threshold still goes to a human if deterministic reference
data disagrees with the prediction. The number never outranks the control, and the prediction is
never silently replaced — a human decides.

#### State precedence

Several conditions can hold at once. An ambiguous Country always scores `0.00`, so it is *also*
below any threshold; the primary state names the root cause, not the symptom.

| # | State | `forced_review` | Meaning |
|---|---|---|---|
| 1 | `HITL_PROCESSING_ERROR` | `True` | extraction failed after configured retries |
| 2 | `HITL_MANUAL_OVERRIDE` | `True` | manual / business override (Phase 2 seam; never set in Phase 1) |
| 3 | `HITL_AMBIGUOUS_COUNTRY` | `True` | several Country candidates remain unresolved |
| 4 | `HITL_REFERENCE_CONFLICT` | `True` | prediction contradicts deterministic reference data |
| 5 | `HITL_LOW_SCORE` | `False` | score below the configured threshold, nothing else wrong |
| 6 | `AUTO_ACCEPT_CANDIDATE` | `False` | score meets the threshold and no control applies |

Only the primary state reaches the CSV; every applicable reason is retained in the detailed JSON
under `contributing_reasons`. Comparison against the threshold uses full precision — rounding happens
only when composing the reason text, and never in a way that makes the sentence contradict itself.

`forced_review` is what tells an auditor *why* a case is with a human: because the score fell short,
or because a stronger control took over regardless of the score.

> **`AUTO_ACCEPT_CANDIDATE` means eligible under the current Phase 1 routing policy.** It is not
> regulatory approval or final downstream authorization.

A **null-skipped group is blank** — `HITL_flag=False`, empty state, empty reason. It is deliberately
not `AUTO_ACCEPT_CANDIDATE`: the model never saw it, so no judgement was made either way. The
detailed JSON keeps `"status": "null_skip"` as the authoritative marker.

`ScoreResult.needs_hitl` is unchanged and still present. `HitlDecision.required` agrees with it on
every Phase 1 path; the documented exception is a manual override, which forces review by design.

#### JSON block

```json
"hitl": {
  "required": true,
  "state": "HITL_REFERENCE_CONFLICT",
  "reason": "Predicted Country conflicts with deterministic reference data; human review is required despite Composite Weighted Score 0.91 meeting configured threshold 0.80.",
  "configured_threshold": 0.80,
  "composite_weighted_score": 0.91,
  "forced_review": true,
  "contributing_reasons": ["reference_conflict"],
  "manual_override": false
}
```

`models/swft_tc/outputs/reports/hitl_state_distribution.csv` reports the state mix over **non-empty** address-group
instances; null skips are excluded from the denominator.

### Provisional recommendation: 0.90

`reporting.recommended_threshold` is **0.90** — a conservative Phase 1 starting point, and
configuration rather than anything hard-coded in reporting logic. It is a **provisional routing
policy, not calibrated accuracy.**

The reasoning is structural, not tuned. Because the composite is a *product* of two weighted terms,
each scenario has a hard ceiling reached only at perfect model confidence:

| Scenario | Ceiling | Can clear 0.90? |
|---|---:|---|
| `both_explicit` | 1.0000 | yes |
| `country_explicit_town_inferred` | 0.5000 | no |
| `town_explicit_country_inferred` | 0.3750 | no |
| `neither_explicit_both_inferred` | 0.0400 | no |
| any ambiguous / no-defensible scenario | 0.0000 | no |

So 0.90 is really a *shape* decision — "auto-accept only when Town and Country are both explicitly
present in the address text and the model is highly confident" — and within `both_explicit` it
requires roughly 0.95 on both confidences if the two are similar (`0.95 × 0.95 = 0.9025`). Note that
between 0.3750 and 0.5000 the threshold makes no practical difference at all: no scenario produces
scores there.

**Choose the production threshold from labeled validation data.** Once labels exist, extend
`threshold_sensitivity.csv` with precision of the auto-accepted population, its error rate, and
recall/coverage, then apply the governance criterion: **select the lowest threshold that still meets
the business-approved minimum precision** — optimize for precision at an accepted review cost, not
for throughput.

Ambiguity, extraction errors, and model/reference conflicts force HITL at every threshold, so the
sensitivity table reports those forced counts separately from the score-driven ones.

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

Run from the repository root. There is one import convention: modules are
imported from `models.swft_tc.src`, the same way the tests, the notebooks and
`scripts/swft_tc/run_batch.py` import them.

```python
from models.swft_tc.src.grouping import load_group_config
from models.swft_tc.src.pipeline import run_phase1
from models.swft_tc.src.reference_data import build_provider, find_iso_provider
from models.swft_tc.src.schemas import load_prompt_contract
from models.swft_tc.src.settings import load_config, resolve_model_name
from models.swft_tc.src.gemini_client import build_client

# Relative config and data paths resolve against the model root
# (models/swft_tc/), never against the working directory.
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
python -m pytest          # 526 tests, run from the repository root
```

Tests never read the real Town/Country reference file. `tests/swft_tc/conftest.py` repoints
`reference_data.town_country_path` at `tests/swft_tc/fixtures/town_country_reference_test.csv` and redirects
every output path into `tmp_path`, so the suite is hermetic and runs with no credentials, no network,
and no large local data.

### Outputs

Everything under `models/swft_tc/outputs/` is generated. Artifacts holding raw addresses are
git-ignored; the aggregate-only reports, the run metrics and `outputs/README.md` are tracked so a
fresh clone gets the directory, its sensitivity warning and shareable enterprise reference figures.
Every writer creates its parent directories, so deleting the tree is safe. Paths below are relative
to `models/swft_tc/`.

| Path | Contents | Raw addresses |
|---|---|---|
| `outputs/phase1_output.csv` | every input column unchanged + 20 columns per enabled group | **Yes** |
| `outputs/phase1_detailed_output.jsonl` | nested per-record detail (streamed JSON Lines) | **Yes** |
| `outputs/address_cache.jsonl` | cached extractions + audit metadata | **Yes** |
| `outputs/errors/processing_errors.csv` | one row per failed unique address, referenced by SHA-256 | No |
| `outputs/run_metrics.json` | shape, savings, cache/call counts, scenario counts, HITL counts, reference-data provenance | No |
| `outputs/reports/executive_summary.json` | headline KPIs for circulation | No |
| `outputs/reports/score_distribution.csv` | composite-score bands over non-empty instances | No |
| `outputs/reports/scenario_distribution.csv` | policy-scenario mix | No |
| `outputs/reports/threshold_sensitivity.csv` | routing volume at each candidate threshold | No |
| `outputs/reports/cross_entropy_summary.csv` | calibration over grounded observations only | No |
| `outputs/reports/hitl_state_distribution.csv` | HITL routing-state mix over non-empty instances | No |
| `outputs/charts/composite_score_histogram.png` | the score distribution as a chart | No |

### Town/Country reference data (external runtime dependency)

The Town/Country reference is **not part of this repository or any package built from it**. It is a
large (~38 MB), environment-specific file read directly from its configured path:

```yaml
reference_data:
  town_country_enabled: true
  town_country_path: "data/reference/town_country_reference.csv"   # relative to repo root
```

Build a development copy — the script downloads from GeoNames and writes the expected schema:

```bash
python scripts/swft_tc/build_geonames_town_country_reference.py \
  --output models/swft_tc/data/reference/town_country_reference.csv
```

GeoNames Gazetteer data is licensed **CC BY 4.0 and requires attribution**; preserve it wherever the
derived file or its outputs are distributed. `models/swft_tc/data/reference/*.csv` is git-ignored
(except the small tracked ISO 3166-1 table), so the file never
enters version control. Unit tests use the tiny committed fixture
`tests/swft_tc/fixtures/town_country_reference_test.csv` instead, never the real file.

If `town_country_enabled` is true and the file is missing, the pipeline **fails fast** with a message
naming the builder script and the config keys. It does not fall back to web search or to the model's
geographic knowledge.

### Executive report

`models/swft_tc/src/reporting.py` builds the report; the notebook presents it. Three denominators are
tracked separately and never mixed: **records**, **address-group instances** (split empty/non-empty),
and **unique addresses**. The score distribution and threshold sensitivity both use *non-empty
address-group instances*, since that is where review workload lands — empty instances are reported
separately and never pad the histogram.

## Implementation decisions

Decisions taken where the starter artifacts were silent or inconsistent. Each is configuration
plus documentation rather than an assumption buried in code.

1. **`models/swft_tc/config/config.yaml`** is the file the pipeline loads; `config.example.yaml` is left untouched
   as the supplied sample. Every key added beyond the example is marked `[EXTENSION]` in place.
   Override the path with `SWIFT_ADDRESS_CONFIG`.
2. **`.env.example` and `models/swft_tc/data/reference/iso3166.csv` were referenced but absent** from the starter
   artifacts, and have been created. The ISO dataset is a *development fallback*; its status is
   recorded in `models/swft_tc/data/reference/PROVENANCE.md`, reported by `Iso3166Provider.provenance`, and written
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
    enabled in `models/swft_tc/config/group_config.csv` and must be confirmed against the authoritative project
    config before production use. Disabling it changes the column arithmetic automatically —
    nothing in the code knows the number 16.
13. **Public web-search grounding is refused in code**, not merely defaulted off: constructing the
    Gemini client with grounding enabled raises. This data may contain customer and payment
    addresses.
15. **Country Name is deterministic, never predicted.** It is expanded from `predicted_country_*`
    through the ISO layer and stays aligned element for element. Two consequences worth knowing:
    an unknown code expands to itself rather than to a blank cell, and any separator character
    occurring *inside* a country name is folded to a space — several ISO short names are inverted
    forms (`Taiwan, Province of China`), and one of those would otherwise add a phantom element and
    silently break the alignment contract. `models/swft_tc/data/reference/iso3166.csv` now stores comma-free
    display names, keeping the official inverted form as a matching alias.
16. **The Town/Country reference validates; it does not overwrite.** Precedence is fixed and
    deterministic:
    1. explicit country evidence in the address always wins — the reference can only agree
       (`consistent`) or disagree (`conflict`);
    2. a `conflict` never replaces the model's value; it forces HITL and records the finding;
    3. if the model returned no country at all, a single-country town fills the gap
       (`supplied_by_reference`) with the model's own confidence, so it surfaces for review rather
       than manufacturing certainty;
    4. only when the address states **no** country explicitly *and* the town genuinely spans several
       countries does `town_country_ambiguity_policy: escalate` (the default) preserve every
       candidate and force the composite to `0.0`;
    5. a town absent from the reference is `reference_not_found` — a reference miss, never an
       extraction error.

   **The shipped runtime configuration uses `annotate`.** The development Town/Country reference is
   corroborative only, so `LIMA` with a defensible model inference of `PE` stays `PE`; the presence
   of Lima, Ohio is recorded in the audit trail and metrics. Use
   `town_country_ambiguity_policy: escalate` only when multi-country reference findings must force
   `PE,US`, a `0.0` composite score, and mandatory HITL. Explicitly stated countries always retain
   precedence in either mode.
17. **Reference findings are audit data, not CSV columns.** `reference_status` and the candidate set
    live in the audit payload and `run_metrics.json`; the production schema stays at 12 fields.
18. **Logs reference addresses by SHA-256**, never in plaintext, unless
    `SWIFT_ADDRESS_ALLOW_RAW_LOGS=true` is set in an approved debugging environment. The cache and
    the audit payload hold the raw text because they *are* the audit record;
    `models/swft_tc/outputs/` is
    git-ignored for that reason.

## Repository layout

The model is a self-contained unit under `models/swft_tc/`. Everything it owns —
source, configuration, prompt contract, reference data and run artifacts — lives
inside that directory; notebooks, scripts, tests and documentation sit in the
repository's shared top-level folders under a matching `swft_tc/` subfolder.

```text
models/swft_tc/
├── __init__.py                        exports MODEL_ROOT and REPO_ROOT: the single path anchor
├── config/
│   ├── config.yaml                    runtime config (loaded); config.example.yaml is the sample
│   ├── config.example.yaml            sample runtime/scoring/naming configuration
│   └── group_config.csv               16 groups x 3 lines; any N groups / N lines is supported
├── data/
│   ├── sample_input.csv               small 50-column sample input
│   ├── sample_expected_group15.csv    expected Town/Country for the sample rows
│   └── reference/
│       ├── iso3166.csv                ISO 3166-1 development fallback (see PROVENANCE.md)
│       ├── town_country_reference.csv external runtime dependency; git-ignored, not bundled
│       └── PROVENANCE.md              source, licence and refresh policy for both files
├── outputs/                           generated run artifacts; raw-address files git-ignored
│   ├── errors/                        processing_errors.csv
│   └── reports/                       aggregate-only report artifacts (tracked)
├── prompts/
│   └── GEMINI_EXTRACTION_PROMPT.md    single source of the prompt text
└── src/                               settings, io, grouping, cleaning, schemas, reference_data,
                                       gemini_client, scoring, evaluation, retraction, cache,
                                       pipeline, reporting, serialization

notebooks/swft_tc/                     DRY_RUN and ACTUAL_RUN walkthroughs
scripts/swft_tc/
├── run_batch.py                       batch entry point; argument parsing only, no business logic
└── build_geonames_town_country_reference.py   builds the reference from GeoNames (CC BY 4.0)
tests/swft_tc/                         526 tests covering the acceptance criteria
└── fixtures/                          tiny Town/Country fixture used instead of the real file
docs/swft_tc/                          architecture, scoring spec, provenance, prompt history
```

### Running it

Every entry point runs from the repository root and resolves its own paths from
the package location, so no step depends on the working directory:

```bash
python -m pytest                                  # 526 tests
python scripts/swft_tc/run_batch.py --dry-run     # offline batch run, no credentials
jupyter lab notebooks/swft_tc/                    # the notebook walkthroughs
```

## Starter artifacts

- `docs/swft_tc/architecture.md` — component and processing design.
- `docs/swft_tc/SCORING_SPEC.md` — Composite Weighted Score and ambiguity rules.
- `docs/swft_tc/REFERENCE_PROVENANCE.md` — where each reference table comes from and how it is refreshed.
- `docs/swft_tc/prompt-history/` — the original and successive build prompts, kept for audit.
- `docs/swft_tc/swift_project_reference.xlsx` — workbook version of group config, input schema, sample input, output schema, scoring rules, and expected examples.
- `models/swft_tc/prompts/GEMINI_EXTRACTION_PROMPT.md` — prompt contract for Gemini.
- `models/swft_tc/config/group_config.csv` — 16-group sample configuration reconstructed from the screenshots/input schema.
- `models/swft_tc/config/config.example.yaml` — runtime/scoring/naming configuration.
- `models/swft_tc/data/sample_input.csv` — small 50-column sample input reconstructed from screenshots.
- `.env.example` — environment variable names only; no value is ever committed.
