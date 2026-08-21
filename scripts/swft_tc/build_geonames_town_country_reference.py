#!/usr/bin/env python3
"""Build a development Town↔Country reference CSV from official GeoNames dumps.

Source:
  https://download.geonames.org/export/dump/cities500.zip
  https://download.geonames.org/export/dump/countryInfo.txt

GeoNames Gazetteer data is CC BY 4.0. Preserve attribution in your project.
This utility is intended for development/reference validation. An enterprise
deployment should replace the output with an approved reference-managed source.
"""

from __future__ import annotations

import argparse
import csv
import io
import unicodedata
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

CITIES_URL = "https://download.geonames.org/export/dump/cities500.zip"
COUNTRY_URL = "https://download.geonames.org/export/dump/countryInfo.txt"

GEONAMES_FIELDS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
    "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]

def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    return " ".join(text.upper().split())

def download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "swift-tc-reference-builder/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()

def load_country_names(raw: bytes) -> dict[str, str]:
    mapping: dict[str, str] = {}
    text = raw.decode("utf-8")
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 5:
            mapping[parts[0].strip().upper()] = parts[4].strip()
    return mapping

#: The model package that owns the reference file, derived from this script's
#: location so the default output lands in the right place from any directory.
MODEL_ROOT = Path(__file__).resolve().parents[2] / "models" / "swft_tc"

DEFAULT_OUTPUT = MODEL_ROOT / "data" / "reference" / "town_country_reference.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=(
            "Output CSV path. Relative paths resolve against the working "
            f"directory; the default is {DEFAULT_OUTPUT.relative_to(MODEL_ROOT.parents[1])}."
        ),
    )
    parser.add_argument(
        "--include-ascii-alias",
        action="store_true",
        help="Emit an additional row for a distinct ASCII town name",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading GeoNames countryInfo.txt...")
    countries = load_country_names(download_bytes(COUNTRY_URL))

    print("Downloading GeoNames cities500.zip...")
    zip_bytes = download_bytes(CITIES_URL)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        member = next(name for name in zf.namelist() if name.endswith(".txt"))
        source_text = io.TextIOWrapper(zf.open(member), encoding="utf-8")

        run_date = datetime.now(timezone.utc).date().isoformat()
        fieldnames = [
            "town_name",
            "town_name_normalized",
            "country_code",
            "country_name",
            "admin1_code",
            "feature_code",
            "population",
            "geoname_id",
            "source_dataset",
            "source_version",
            "source_url",
            "approved_for_production",
        ]

        count = 0
        with output.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()

            for raw_line in source_text:
                parts = raw_line.rstrip("\n").split("\t")
                if len(parts) < len(GEONAMES_FIELDS):
                    continue
                row = dict(zip(GEONAMES_FIELDS, parts))
                code = row["country_code"].strip().upper()
                if not code:
                    continue

                def emit(name: str) -> None:
                    nonlocal count
                    name = (name or "").strip()
                    if not name:
                        return
                    writer.writerow({
                        "town_name": name,
                        "town_name_normalized": normalize_name(name),
                        "country_code": code,
                        "country_name": countries.get(code, ""),
                        "admin1_code": row["admin1_code"].strip(),
                        "feature_code": row["feature_code"].strip(),
                        "population": row["population"].strip(),
                        "geoname_id": row["geonameid"].strip(),
                        "source_dataset": "GeoNames cities500",
                        "source_version": run_date,
                        "source_url": CITIES_URL,
                        "approved_for_production": "false",
                    })
                    count += 1

                emit(row["name"])
                if args.include_ascii_alias:
                    ascii_name = row["asciiname"].strip()
                    if ascii_name and normalize_name(ascii_name) != normalize_name(row["name"]):
                        emit(ascii_name)

    print(f"Wrote {count:,} reference rows to {output}")
    print("Source: GeoNames cities500 + countryInfo")
    print("License: CC BY 4.0")
    print("Production approval: false — replace with enterprise-approved source.")

if __name__ == "__main__":
    main()
