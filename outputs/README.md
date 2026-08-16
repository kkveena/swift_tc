# `outputs/` — generated run artifacts

**Everything in this directory except this README is generated and git-ignored.**

The `.gitignore` rule is deliberately `outputs/*` plus a `!outputs/README.md`
negation, so the directory exists in a fresh clone (with this explanation in it)
while nothing it produces is ever committed.

Every writer calls `Path(...).parent.mkdir(parents=True, exist_ok=True)` before
writing, so deleting this whole tree is safe — the next run rebuilds it.

## Sensitivity

> **This directory can contain customer and payment address data.**

`phase1_output.csv` carries the source address columns verbatim,
`phase1_detailed_output.jsonl` carries them again in nested form (including
per-column before/after retraction detail), and `address_cache.jsonl` stores raw
cleaned addresses because the cache *is* the audit record for what was sent to
the model. Treat all three as regulated data: do not commit them, do not attach
them to tickets, and do not copy them outside approved storage.

The reporting artifacts are deliberately free of raw addresses —
`executive_summary.json` and the three report CSVs are aggregate-only, and
`processing_errors.csv` references addresses by SHA-256 hash rather than text —
so those are the ones safe to circulate.

## Contents

| Path | Written by | Contains raw addresses |
|---|---|---|
| `phase1_output.csv` | `io.write_output_csv` | **Yes** |
| `phase1_detailed_output.jsonl` | `serialization.write_detailed_json` | **Yes** |
| `address_cache.jsonl` | `cache.AddressCache` | **Yes** |
| `processing_errors.csv` | `io.write_errors_csv` | No — SHA-256 hashes only |
| `run_metrics.json` | `io.write_metrics_json` | No |
| `reports/executive_summary.json` | `reporting.write_reports` | No |
| `reports/score_distribution.csv` | `reporting.write_reports` | No |
| `reports/scenario_distribution.csv` | `reporting.write_reports` | No |
| `reports/threshold_sensitivity.csv` | `reporting.write_reports` | No |
| `reports/cross_entropy_summary.csv` | `reporting.write_reports` | No |
| `reports/hitl_state_distribution.csv` | `reporting.write_reports` | No |
| `charts/composite_score_histogram.png` | `reporting.render_score_histogram` | No |

## Paths are configuration

All of the above are configurable — `processing.*_path` and the `reporting`
section of `config/config.yaml`. Point them at approved storage for a production
run; nothing in the code assumes this directory.
