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

### Before production use

Replace this file with the reference-managed ISO 3166-1 extract from your
organization's approved data source, keep the same three columns
(`alpha2,name,aliases`, aliases pipe-separated), and bump
`reference_data.reference_context_version` so cached extractions built against
the development dataset are invalidated.

`Iso3166Provider.provenance` surfaces this status at runtime and it is written
into `outputs/run_metrics.json` for every run, so no run can quietly claim
approved reference data.

## SWIFTRef

**Not included, and not fetched.** SWIFTRef (including the BIC Directory) is
licensed data. `SwiftRefProvider` in `src/swift_address/reference_data.py` is a
deliberate interface stub that raises unless an approved API endpoint or
licensed local directory file is configured. Nothing in this repository
scrapes, mirrors, or simulates SWIFTRef content, and the Gemini prompt forbids
the model from claiming a SWIFTRef consultation that the program did not
actually supply.
