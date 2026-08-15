# Town/Country Reference — Development Provenance

## Supplied file

`data/reference/town_country_reference.csv`

This is a **development/reference-validation dataset**, not an enterprise-approved golden source.

| Item | Value |
|---|---|
| Rows | 170,511 |
| Unique normalized town names | 150,119 |
| Distinct Town/Country pairs | 156,535 |
| Town names mapping to >1 country | 4,505 |
| Country codes represented | 245 |
| Public source | `joelacus/world-cities` |
| Upstream geographic source | GeoNames |
| Population threshold | >= 1,000 |
| Source commit | `55bcdd6387eb17e3b12ef56860e42f34c30178f7` (`build 20260810`) |
| License | Creative Commons Attribution 4.0 |
| Approved for production | **No** |

The country display name is joined from GeoNames `countryInfo.txt`; the Town/Country rows
come from the GeoNames-derived `world_cities.csv` source.

The supplied file is intended to make Phase 1 deterministic and reproducible enough for
development, unit tests, ambiguity analysis, and HITL design. For enterprise deployment,
replace it with your organization's approved reference-managed Town/Country source and
bump the configured reference version so stale cache entries are invalidated.

## Why the file preserves duplicate town names

A Town can legitimately occur in multiple countries. The file intentionally keeps those
Town/Country mappings. The provider should index by normalized Town and return the full
candidate country set, rather than selecting one arbitrarily.

## Optional rebuild from official GeoNames dump

`build_geonames_town_country_reference.py` is also supplied. It can build a larger
development reference directly from GeoNames `cities500.zip` and `countryInfo.txt`.

Example:

```bash
python build_geonames_town_country_reference.py   --output data/reference/town_country_reference.csv   --include-ascii-alias
```

GeoNames Gazetteer data is licensed under CC BY 4.0; preserve attribution.

## SWIFTRef / BIC

No SWIFTRef data is included. BIC/SWIFTRef integration should remain a separate entitled
reference provider in a later phase.
