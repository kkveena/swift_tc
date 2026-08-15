# Composite Weighted Score Specification

## Purpose
Define the deterministic operational score used to route Town/Country extraction results to Human-in-the-Loop (HITL).

## Inputs
- `town_probability` — final Town model confidence in `[0,1]`.
- `country_probability` — final Country model confidence in `[0,1]`.
- verified `town_exists` / `country_exists`.
- `country_ambiguous`.
- configured reliability weights.

## Multiple-country override
When more than one Country remains defensible and cannot be uniquely resolved:

1. preserve all candidates as comma-separated ISO alpha-2 codes (for example `CA,US`);
2. set final `country_probability = 0.0`;
3. set `country_weight = 0.0`;
4. set `composite_weighted_score = 0.0`;
5. force HITL.

## Reliability weights

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

## Formula

```text
adjusted_town_score = town_probability × town_weight
adjusted_country_score = country_probability × country_weight
composite_weighted_score = adjusted_town_score × adjusted_country_score
```

The result is an operational routing score, not a calibrated joint probability unless later validation demonstrates calibration.

## Examples

| Case | Town p | Country p | Town w | Country w | Composite |
|---|---:|---:|---:|---:|---:|
| Both explicit | 0.99 | 0.98 | 1.00 | 1.00 | 0.9702 |
| Country explicit, Town inferred | 0.92 | 0.99 | 0.50 | 1.00 | 0.4554 |
| Town explicit, Country inferred | 0.98 | 0.95 | 0.75 | 0.50 | 0.349125 |
| Town explicit, Country ambiguous | 0.98 | 0.00 | 0.50 | 0.00 | 0.0000 |
| Neither explicit | 0.80 | 0.75 | 0.20 | 0.20 | 0.0240 |

## Scenario selection
Select scenarios deterministically from verified explicitness and ambiguity. Do not let Gemini calculate or choose the business score. Keep scenario and weights in audit/cache data even if they are not output CSV columns.

## Threshold
Keep `hitl_threshold` in config. Calibrate against labeled production-like data and report precision/coverage by scenario. Any unresolved multiple-country result is always HITL.

### Threshold reporting

`reporting.build_threshold_sensitivity` evaluates the configured candidate thresholds (default
`0.80, 0.85, 0.90, 0.95`) over **non-empty address-group instances** and reports, per threshold:
auto-accept candidate count and percent, HITL count and percent, and — separately — the counts forced
to HITL by country ambiguity and by extraction errors. Those forced cases are invariant across
thresholds by design: a numeric score never releases them.

Because the composite is a product, each scenario has a ceiling reached only at perfect confidence:
`both_explicit` 1.0000, `country_explicit_town_inferred` 0.5000, `town_explicit_country_inferred`
0.3750, `neither_explicit_both_inferred` 0.0400, ambiguous/no-defensible 0.0000. A threshold above
0.5000 therefore admits `both_explicit` only, and any threshold between 0.3750 and 0.5000 is
operationally identical.

`reporting.recommended_threshold` (default **0.90**) is a **provisional routing policy, not
calibrated accuracy**. Once labeled validation data exists, extend the sensitivity report with
auto-accepted precision, auto-accepted error rate, and recall/coverage, then adopt the lowest
threshold that satisfies the business-approved minimum precision.
