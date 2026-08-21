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

**This formula and the weight matrix above are unchanged** by the addition of the ground-truth and
cross-entropy fields. Those are evaluation outputs; they never enter routing.

| Metric | Question | Direction |
|---|---|---|
| Composite Weighted Score | should this be auto-accepted? | **higher is better** |
| Cross-entropy | did the model's confidence match reality? | **lower is better** |

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

An intuitive reading of 0.90: in `both_explicit` with both weights at 1.00 and roughly equal
confidences, `p × p ≥ 0.90` means `p ≥ √0.90 ≈ 0.949`. So a 0.90 composite threshold approximately
requires **~95% confidence on both Town and Country**, in the one scenario where nothing was
inferred.

The notebooks render this reasoning in a dedicated "Human-in-the-Loop Threshold Recommendation"
section, showing the configured operational threshold and the recommended analytical threshold
*separately* — the recommendation never silently overwrites `scoring.hitl_threshold`.

## HITL decision fields

The Composite Weighted Score is the numeric routing *score*; it is not by itself the routing
*decision*. Three group-level fields make the decision explicit:

```text
HITL_flag          final human-review decision (boolean)
HITL_state         primary routing outcome after precedence (closed enum)
HITL_state_reason  one deterministic sentence explaining it
```

Review is required when `composite < scoring.hitl_threshold` **or** when a forced-review control
applies. Precedence, strongest first:

| # | State | `forced_review` |
|---|---|---|
| 1 | `HITL_PROCESSING_ERROR` | `True` |
| 2 | `HITL_MANUAL_OVERRIDE` | `True` |
| 3 | `HITL_AMBIGUOUS_COUNTRY` | `True` |
| 4 | `HITL_REFERENCE_CONFLICT` | `True` |
| 5 | `HITL_LOW_SCORE` | `False` |
| 6 | `AUTO_ACCEPT_CANDIDATE` | `False` |

A reference conflict at score `0.91` against threshold `0.80` is `HITL_REFERENCE_CONFLICT` with
`HITL_flag=True`: the control overrides the number. A null-skipped group is blank, never
`AUTO_ACCEPT_CANDIDATE`. `ScoreResult.needs_hitl` is unchanged and agrees with `HitlDecision.required`
on every Phase 1 path.

`scoring.hitl_threshold` (0.80) is the operational cutoff used for routing.
`reporting.recommended_threshold` (0.90) is an analytical recommendation shown in reporting and the
notebooks — it never becomes the cutoff.

## Cross-entropy (evaluation, not routing)

`cross_entropy_group_<id>` is binary log loss of model confidence against the nullable ground-truth
labels described in README:

```text
BCE(y, p) = -( y*log(p) + (1-y)*log(1-p) )     p clipped to [1e-6, 1-1e-6]
```

Both labels available → mean of the two component losses. One → that component (status `town_only` /
`country_only`). Neither → blank. Observations without ground truth are **excluded** from the metric,
never assigned an artificially high loss: a reference-coverage gap is not a model error. Component
detail lives in the detailed JSON. Gemini never computes it.
