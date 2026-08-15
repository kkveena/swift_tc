# Reference data provenance

## `iso3166.csv`

| Field | Value |
|---|---|
| Standard | ISO 3166-1 alpha-2 |
| Provenance | **Development fallback.** Hand-transcribed country list plus commonly encountered name aliases. |
| Approved for production | **No.** |
| Entries | 249 alpha-2 codes |
| Referenced by | `config/config.yaml` → `reference_data.iso3166_path` |
| Context version | `reference_data.reference_context_version` (participates in the cache key) |

### Why this file exists as a fallback

`config.example.yaml` points at `data/reference/iso3166.csv`, but no such file
was supplied with the starter artifacts. Rather than silently degrading to a
no-op provider — which would disable all deterministic country verification —
this development dataset is committed so the pipeline is runnable and testable
end to end.

### Display names

`name` is the value the pipeline writes into `predicted_country_name_group_<id>`. Eight entries whose
ISO short name is an inverted form containing a comma (`Taiwan, Province of China`,
`Korea, Republic of`, …) are stored here in comma-free display form, with the official inverted form
kept in `aliases` so presence verification still matches it. This is not cosmetic: the country-name
column is comma-joined and must stay aligned element-for-element with the country-code column, and an
embedded comma would add a phantom element. `scoring._expand_country_names` also folds any remaining
separator character defensively, so a replacement dataset cannot reintroduce the problem.

`US` is stored as `United States` (with `United States of America` as an alias) to match the agreed
output contract.

### Before production use

Replace this file with the reference-managed ISO 3166-1 extract from your
organization's approved data source, keep the same three columns
(`alpha2,name,aliases`, aliases pipe-separated), and bump
`reference_data.reference_context_version` so cached extractions built against
the development dataset are invalidated.

`Iso3166Provider.provenance` surfaces this status at runtime and it is written
into `outputs/run_metrics.json` for every run, so no run can quietly claim
approved reference data.

## `town_country_reference.csv`

| Field | Value |
|---|---|
| Status | **External runtime dependency — not in version control, not bundled.** |
| Approved for production | **No.** |
| Upstream source | GeoNames (`cities500` / `world-cities` derivative) |
| License | Creative Commons Attribution 4.0 — **attribution required** |
| Referenced by | `config/config.yaml` → `reference_data.town_country_path` |
| Git status | ignored via `data/reference/*.csv` |

The file is large (tens of MB) and environment-specific. It is read directly from the configured
path, loaded once per run into an in-memory index, and never copied, renamed, embedded, or packaged.
See `REFERENCE_PROVENANCE.md` in the repository root for the supplied file's row counts and source
commit.

Build a development copy:

```bash
python scripts/build_geonames_town_country_reference.py \
  --output data/reference/town_country_reference.csv
```

If `reference_data.town_country_enabled` is true and the file is absent, the pipeline fails fast with
a message naming the builder script and the relevant config keys. It never falls back to web search
or to the model's own geographic knowledge.

Unit tests use `tests/fixtures/town_country_reference_test.csv` — a 17-row schema sample — and never
the real file. A reference file that claims `approved_for_production=true` cannot override the
operator's configuration: both must agree before a run reports the reference as approved.

Replacing this with an enterprise-managed source requires configuration only
(`town_country_path`, plus `town_country_column_map` if the headers differ). Bump
`reference_data.reference_context_version` so cached extractions built against the development
dataset are invalidated.

## SWIFTRef

**Not included, and not fetched.** SWIFTRef (including the BIC Directory) is
licensed data. `SwiftRefProvider` in `src/swift_address/reference_data.py` is a
deliberate interface stub that raises unless an approved API endpoint or
licensed local directory file is configured. Nothing in this repository
scrapes, mirrors, or simulates SWIFTRef content, and the Gemini prompt forbids
the model from claiming a SWIFTRef consultation that the program did not
actually supply.
