"""Reference-data providers.

Phase 1 ships a no-op provider and an ISO 3166-1 provider backed by a
configurable local file. :class:`SwiftRefProvider` is a deliberate interface
stub: SWIFTRef is licensed data, so nothing here scrapes it, mirrors it, or
pretends to have consulted it. The provider raises unless an approved API
endpoint or licensed local directory file has actually been configured.

Whatever a provider returns is passed to Gemini as ``reference_context``. When
that context is empty, the prompt's rule 3 ("never claim you consulted a source
that is not present in reference_context") is enforceable rather than aspirational.
"""

from __future__ import annotations

import csv
import logging
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from .cleaning import contains_token_phrase, tokens_casefolded

__all__ = [
    "CompositeReferenceDataProvider",
    "CountryRecord",
    "Iso3166Provider",
    "NullReferenceDataProvider",
    "ReferenceContext",
    "ReferenceDataProvider",
    "SwiftRefNotConfiguredError",
    "SwiftRefProvider",
    "TownCountryProvenance",
    "TownCountryProvider",
    "TownCountryReferenceError",
    "build_provider",
    "build_town_country_provider",
    "find_iso_provider",
    "resolve_town_country_file",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CountryRecord:
    """One ISO 3166-1 entry: the alpha-2 code, its name, and known aliases."""

    alpha2: str
    name: str
    aliases: tuple[str, ...] = ()

    @property
    def presence_names(self) -> tuple[str, ...]:
        """Every name form accepted as explicit textual evidence of this country."""
        return (self.name, *self.aliases)


@dataclass(frozen=True)
class ReferenceContext:
    """Program-supplied evidence handed to the model.

    ``sources`` names exactly what the program contributed. It is what the
    model is permitted to cite in ``reference_basis``.
    """

    sources: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)
    version: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.payload

    def to_prompt_dict(self) -> dict[str, Any]:
        if self.is_empty:
            # Say so explicitly rather than sending {}: an unstated absence is
            # easier for a model to talk past than a stated one.
            return {
                "sources": [],
                "note": (
                    "No enterprise reference data was supplied for this address. "
                    "Rely on the address text alone and do not claim any external "
                    "lookup."
                ),
            }
        return {"sources": list(self.sources), **dict(self.payload)}

    def merge(self, other: "ReferenceContext") -> "ReferenceContext":
        merged_payload = {**dict(self.payload), **dict(other.payload)}
        merged_sources = tuple(dict.fromkeys((*self.sources, *other.sources)))
        version = "+".join(part for part in (self.version, other.version) if part)
        return ReferenceContext(
            sources=merged_sources, payload=merged_payload, version=version
        )


@runtime_checkable
class ReferenceDataProvider(Protocol):
    """Supplies approved reference evidence for one address."""

    name: str

    def get_context(self, address: str) -> ReferenceContext:
        """Return the reference context for ``address`` (never raises for a miss)."""
        ...

    @property
    def context_version(self) -> str:
        """Version token for this provider's corpus; participates in cache keys."""
        ...


class NullReferenceDataProvider:
    """Supplies nothing. The honest Phase 1 default when no data is provisioned."""

    name = "null"

    def __init__(self, version: str = "null-v1") -> None:
        self._version = version

    def get_context(self, address: str) -> ReferenceContext:  # noqa: ARG002
        return ReferenceContext(sources=(), payload={}, version=self._version)

    @property
    def context_version(self) -> str:
        return self._version

    @property
    def provenance(self) -> dict[str, Any]:
        return {"provider": self.name, "approved_for_production": True, "records": 0}


class Iso3166Provider:
    """Validates and resolves ISO 3166-1 alpha-2 codes from a local dataset.

    The dataset path is configuration (``reference_data.iso3166_path``) so a
    reference-managed extract can replace the development file without a code
    change. :attr:`provenance` reports which one is actually loaded, and that
    lands in ``run_metrics.json`` — a run cannot silently claim approved data.
    """

    name = "iso3166"

    def __init__(
        self,
        records: Iterable[CountryRecord],
        *,
        version: str = "iso3166",
        source_path: Path | None = None,
        approved_for_production: bool = False,
        ambiguous_alpha2_tokens: Iterable[str] = (),
        trailing_country_token_window: int = 3,
    ) -> None:
        self._by_code: dict[str, CountryRecord] = {}
        for record in records:
            self._by_code[record.alpha2.upper()] = record
        if not self._by_code:
            raise ValueError("Iso3166Provider requires at least one country record")
        self._version = version
        self._source_path = source_path
        self._approved = approved_for_production
        self._ambiguous = {token.upper() for token in ambiguous_alpha2_tokens}
        self._trailing_window = max(1, int(trailing_country_token_window))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        version: str = "iso3166",
        approved_for_production: bool = False,
        ambiguous_alpha2_tokens: Iterable[str] = (),
        trailing_country_token_window: int = 3,
    ) -> "Iso3166Provider":
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"ISO 3166 reference dataset not found: {csv_path}. Configure "
                "reference_data.iso3166_path, or set reference_data.provider to "
                "'null' to run without deterministic country verification."
            )
        records: list[CountryRecord] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"alpha2", "name"}
            if not required.issubset({(f or "").strip() for f in reader.fieldnames or []}):
                raise ValueError(
                    f"{csv_path} must define at least the columns: {sorted(required)}"
                )
            for row in reader:
                alpha2 = (row.get("alpha2") or "").strip().upper()
                if not alpha2:
                    continue
                aliases = tuple(
                    alias.strip()
                    for alias in (row.get("aliases") or "").split("|")
                    if alias.strip()
                )
                records.append(
                    CountryRecord(
                        alpha2=alpha2,
                        name=(row.get("name") or "").strip(),
                        aliases=aliases,
                    )
                )
        return cls(
            records,
            version=version,
            source_path=csv_path,
            approved_for_production=approved_for_production,
            ambiguous_alpha2_tokens=ambiguous_alpha2_tokens,
            trailing_country_token_window=trailing_country_token_window,
        )

    # -- provider protocol -------------------------------------------------

    def get_context(self, address: str) -> ReferenceContext:
        """Supply the ISO codes whose name or code actually occurs in the address.

        This is evidence the program found, not a search the model performed.
        Nothing is invented: an address mentioning no country yields no codes.
        """
        matches = sorted(
            code
            for code in self._by_code
            if self.country_is_present(address, code)
        )
        if not matches:
            return ReferenceContext(sources=(), payload={}, version=self._version)
        return ReferenceContext(
            sources=("iso3166",),
            payload={
                "iso3166_codes_present_in_address": matches,
                "iso3166_dataset_version": self._version,
            },
            version=self._version,
        )

    @property
    def context_version(self) -> str:
        return self._version

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "source_path": str(self._source_path) if self._source_path else None,
            "dataset_version": self._version,
            "records": len(self._by_code),
            "approved_for_production": self._approved,
        }

    # -- validation / verification ----------------------------------------

    def is_valid_alpha2(self, code: str) -> bool:
        return code.strip().upper() in self._by_code

    def record(self, code: str) -> CountryRecord | None:
        return self._by_code.get(code.strip().upper())

    def country_name(self, code: str) -> str:
        """Expand one alpha-2 code to its reference country name.

        Deterministic and reference-derived — the model is never asked for a
        country name. An unknown code returns the code itself rather than an
        empty string, so a name column never silently loses information.
        """
        record = self.record(code)
        if record is None:
            return code.strip().upper()
        return record.name or record.alpha2

    def country_names(self, codes: Iterable[str]) -> tuple[str, ...]:
        """Expand a candidate list, preserving order and length exactly.

        Element-for-element alignment with the code list is the contract:
        ``("CA", "US")`` becomes ``("Canada", "United States of America")``.
        """
        return tuple(self.country_name(code) for code in codes)

    def invalid_codes(self, codes: Iterable[str]) -> tuple[str, ...]:
        return tuple(code for code in codes if not self.is_valid_alpha2(code))

    def country_is_present(self, address: str, code: str) -> bool:
        """Deterministically verify explicit support for one country in the text.

        Accepts either a country-name alias on token boundaries, or the bare
        alpha-2 code as a standalone token. Codes that collide with ordinary
        address vocabulary (``IN``, ``IT``, ``NO``, ``ME``, …) are only accepted
        in trailing country position, so "SUITE 5 IN TOWER" cannot be read as
        evidence of India.
        """
        return bool(self.matched_presence_forms(address, code))

    def is_ambiguous_alpha2(self, code: str) -> bool:
        """Whether this code collides with ordinary address/English vocabulary."""
        return code.strip().upper() in self._ambiguous

    @property
    def trailing_country_token_window(self) -> int:
        return self._trailing_window

    def matched_presence_forms(self, address: str, code: str) -> tuple[str, ...]:
        """Which textual forms of ``code`` actually occur in ``address``.

        Same rules as :meth:`country_is_present`, but reporting *what* matched
        rather than merely whether something did. The retraction layer needs
        this: it may only remove the exact forms deterministic verification
        found, never a guess at how the country might have been written.
        """
        record = self.record(code)
        if record is None:
            return ()

        forms: list[str] = []
        for name in record.presence_names:
            if name and contains_token_phrase(address, name):
                forms.append(name)

        tokens = tokens_casefolded(address)
        alpha2 = record.alpha2
        if alpha2 in tokens and (
            alpha2 not in self._ambiguous
            # Ambiguous code: require trailing country position.
            or alpha2 in tokens[-self._trailing_window :]
        ):
            forms.append(alpha2)

        return tuple(dict.fromkeys(forms))


class TownCountryReferenceError(RuntimeError):
    """The configured Town/Country reference could not be loaded."""


@dataclass(frozen=True)
class TownCountryProvenance:
    """What was loaded, from where, and whether it may be trusted in production."""

    path: str
    rows: int
    unique_towns: int
    multi_country_towns: int
    country_codes: int
    source_dataset: str
    source_version: str
    source_url: str
    approved_for_production: bool
    configured_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rows": self.rows,
            "unique_towns": self.unique_towns,
            "multi_country_towns": self.multi_country_towns,
            "country_codes": self.country_codes,
            "source_dataset": self.source_dataset,
            "source_version": self.source_version,
            "source_url": self.source_url,
            "approved_for_production": self.approved_for_production,
            "configured_source": self.configured_source,
        }


class TownCountryProvider:
    """In-memory Town -> country-code index over a local reference file.

    The reference file is an **external runtime dependency**: large
    (tens of MB), environment-specific, git-ignored, and never bundled with this
    package. It is read once per run, streamed row by row with the stdlib CSV
    reader, and collapsed into a dictionary. Per-address lookups are then plain
    dict hits — the file is never re-scanned::

        provider.lookup_country_codes("AUCKLAND")  -> ("NZ",)
        provider.lookup_country_codes("HAMILTON")  -> ("BM", "CA", "NZ")

    Each town is indexed under two keys: the file's own normalized form, and a
    punctuation-folded form, so ``SAINT-DENIS`` in the reference still matches a
    predicted ``SAINT DENIS``.

    This is a *validation* signal. It never silently rewrites a model result;
    :mod:`swift_address.scoring` decides what a lookup means.
    """

    name = "town_country"

    #: Expected logical columns. A file with different headers is mapped via
    #: `reference_data.town_country_column_map` rather than being rewritten.
    EXPECTED_COLUMNS: tuple[str, ...] = (
        "town_name",
        "town_name_normalized",
        "country_code",
        "country_name",
        "source_dataset",
        "source_version",
        "source_url",
        "approved_for_production",
    )

    def __init__(
        self,
        index: Mapping[str, tuple[str, ...]],
        *,
        provenance: TownCountryProvenance,
        version: str,
    ) -> None:
        self._index = dict(index)
        self._provenance = provenance
        self._version = version

    # -- construction ------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        version: str,
        configured_source: str = "development_local_reference",
        approved_for_production: bool = False,
        column_map: Mapping[str, str] | None = None,
    ) -> "TownCountryProvider":
        """Load and index the reference file. Fails fast when it is absent."""
        resolved = resolve_town_country_file(path)

        mapping = {str(k): str(v) for k, v in (column_map or {}).items()}
        # The map is given physical -> logical; invert for lookup by logical.
        logical_to_physical = {logical: physical for physical, logical in mapping.items()}

        index: dict[str, set[str]] = defaultdict(set)
        codes_seen: set[str] = set()
        rows = 0
        source_dataset = source_version = source_url = ""
        file_approved: bool | None = None

        delimiter = _sniff_delimiter(resolved)
        with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            headers = [(name or "").strip() for name in (reader.fieldnames or [])]

            def column_for(logical: str) -> str | None:
                physical = logical_to_physical.get(logical, logical)
                return physical if physical in headers else None

            town_column = column_for("town_name_normalized") or column_for("town_name")
            code_column = column_for("country_code")
            raw_town_column = column_for("town_name")
            if town_column is None or code_column is None:
                raise TownCountryReferenceError(
                    f"{resolved} must provide a town column and a country_code column. "
                    f"Found headers: {', '.join(headers[:12])}. Map differing headers "
                    "with reference_data.town_country_column_map "
                    "(physical_name: logical_name)."
                )

            dataset_column = column_for("source_dataset")
            version_column = column_for("source_version")
            url_column = column_for("source_url")
            approved_column = column_for("approved_for_production")

            for row in reader:
                code = (row.get(code_column) or "").strip().upper()
                if not code:
                    continue
                town = (row.get(town_column) or "").strip()
                if not town and raw_town_column:
                    town = (row.get(raw_town_column) or "").strip()
                if not town:
                    continue

                rows += 1
                codes_seen.add(code)
                for key in _town_keys(town):
                    index[key].add(code)

                if rows == 1:
                    source_dataset = (row.get(dataset_column) or "").strip() if dataset_column else ""
                    source_version = (row.get(version_column) or "").strip() if version_column else ""
                    source_url = (row.get(url_column) or "").strip() if url_column else ""
                    if approved_column:
                        file_approved = str(row.get(approved_column) or "").strip().lower() in {
                            "1", "true", "yes", "y",
                        }

        if not rows:
            raise TownCountryReferenceError(
                f"{resolved} contained no usable Town/Country rows"
            )

        frozen = {town: tuple(sorted(codes)) for town, codes in index.items()}
        provenance = TownCountryProvenance(
            path=str(resolved),
            rows=rows,
            unique_towns=len(frozen),
            multi_country_towns=sum(1 for codes in frozen.values() if len(codes) > 1),
            country_codes=len(codes_seen),
            source_dataset=source_dataset,
            source_version=source_version,
            source_url=source_url,
            # A file claiming production approval cannot override the operator's
            # configuration: both must agree before this reads as approved.
            approved_for_production=bool(approved_for_production and file_approved),
            configured_source=configured_source,
        )
        logger.info(
            "loaded Town/Country reference: %d row(s), %d unique town(s), "
            "%d multi-country town(s) from %s",
            rows, provenance.unique_towns, provenance.multi_country_towns, resolved,
        )
        return cls(frozen, provenance=provenance, version=version)

    # -- provider protocol -------------------------------------------------

    def get_context(self, address: str) -> ReferenceContext:  # noqa: ARG002
        """No address-level context.

        This provider answers a question about a *predicted town*, which does
        not exist until after extraction. Feeding a whole-address guess into the
        prompt would invite the model to cite reference data the program never
        supplied. Its findings are applied post-extraction instead.
        """
        return ReferenceContext(sources=(), payload={}, version=self._version)

    @property
    def context_version(self) -> str:
        return self._version

    @property
    def provenance(self) -> dict[str, Any]:
        return {"provider": self.name, **self._provenance.to_dict()}

    # -- lookup ------------------------------------------------------------

    def lookup_country_codes(self, town: str) -> tuple[str, ...]:
        """Distinct, sorted ISO alpha-2 codes for a town name. Empty when unknown."""
        if not town:
            return ()
        for key in _town_keys(town):
            found = self._index.get(key)
            if found:
                return found
        return ()

    def knows(self, town: str) -> bool:
        return bool(self.lookup_country_codes(town))

    def __len__(self) -> int:
        return len(self._index)


def resolve_town_country_file(path: str | Path) -> Path:
    """Resolve the configured reference location to an actual file.

    Accepts the exact file, a common data extension appended to an
    extension-less path, or a directory containing exactly one data file — the
    configured value in the enhancement brief is extension-less while the
    generated file carries `.csv`. Nothing is renamed, copied, or rewritten.
    """
    candidate = Path(path)

    if candidate.is_file():
        return candidate

    if candidate.is_dir():
        members = sorted(
            child
            for child in candidate.iterdir()
            if child.is_file() and child.suffix.lower() in {".csv", ".tsv", ".txt"}
        )
        if len(members) == 1:
            return members[0]
        raise TownCountryReferenceError(
            f"{candidate} is a directory containing {len(members)} candidate data "
            "file(s); set reference_data.town_country_path to the exact file."
        )

    for suffix in (".csv", ".tsv", ".txt"):
        with_suffix = candidate.with_name(candidate.name + suffix)
        if with_suffix.is_file():
            return with_suffix

    raise TownCountryReferenceError(
        f"Town/Country reference file not found: {candidate}. "
        "reference_data.town_country_enabled is true, so this file is required. "
        "Build a development copy with "
        "`python scripts/build_geonames_town_country_reference.py --output "
        f"{candidate if candidate.suffix else str(candidate) + '.csv'}`, point "
        "reference_data.town_country_path at an approved managed source, or set "
        "reference_data.town_country_enabled to false. The pipeline will not "
        "substitute web search or model geographic knowledge for a reference "
        "file it was told to use."
    )


def _sniff_delimiter(path: Path) -> str:
    """Detect comma vs tab from the header line. Defaults to comma."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def _town_keys(town: str) -> tuple[str, ...]:
    """Index/lookup keys for a town name.

    Two forms: NFKC-uppercase-collapsed (the reference file's own normalization)
    and a punctuation-folded token form, so hyphenation and apostrophes do not
    cause misses. Duplicates collapse to one key.
    """
    normalized = " ".join(unicodedata.normalize("NFKC", town).upper().split())
    folded = " ".join(tokens_casefolded(town))
    if folded and folded != normalized:
        return (normalized, folded)
    return (normalized,) if normalized else ()


class SwiftRefNotConfiguredError(RuntimeError):
    """SWIFTRef access was requested without approved, entitled configuration."""


class SwiftRefProvider:
    """Interface point for approved SWIFTRef access. Intentionally not implemented.

    SWIFTRef (including the BIC Directory) is licensed data. Wiring this up
    means supplying an entitled API client or a licensed local directory file
    through ``client``/``directory_path``; until then every call raises rather
    than degrading to a guess. Phase 2 replaces the body of
    :meth:`get_context`, not the pipeline around it.
    """

    name = "swiftref"

    def __init__(
        self,
        *,
        client: Any = None,
        directory_path: str | Path | None = None,
        version: str = "swiftref-unconfigured",
    ) -> None:
        self._client = client
        self._directory_path = Path(directory_path) if directory_path else None
        self._version = version

    @property
    def is_configured(self) -> bool:
        return self._client is not None or (
            self._directory_path is not None and self._directory_path.exists()
        )

    def get_context(self, address: str) -> ReferenceContext:  # noqa: ARG002
        raise SwiftRefNotConfiguredError(
            "SwiftRefProvider is an interface stub. SWIFTRef is licensed data and "
            "must be reached through an entitled API client or an approved local "
            "directory file. This repository does not scrape, mirror, or simulate "
            "SWIFTRef content."
        )

    @property
    def context_version(self) -> str:
        return self._version

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "configured": self.is_configured,
            "approved_for_production": False,
        }


class CompositeReferenceDataProvider:
    """Merges several approved providers into one context."""

    name = "composite"

    def __init__(self, providers: Iterable[ReferenceDataProvider]) -> None:
        self._providers = tuple(providers)
        if not self._providers:
            raise ValueError("CompositeReferenceDataProvider requires >= 1 provider")

    def get_context(self, address: str) -> ReferenceContext:
        context = ReferenceContext()
        for provider in self._providers:
            try:
                context = context.merge(provider.get_context(address))
            except SwiftRefNotConfiguredError:
                # An unconfigured licensed provider contributes nothing; it must
                # not take the run down or fabricate a substitute.
                logger.warning(
                    "reference provider %s is not configured; skipping", provider.name
                )
        return context

    @property
    def context_version(self) -> str:
        return "+".join(provider.context_version for provider in self._providers)

    @property
    def providers(self) -> tuple[ReferenceDataProvider, ...]:
        return self._providers

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "members": [
                getattr(p, "provenance", {"provider": p.name}) for p in self._providers
            ],
        }


def build_provider(config: Any, base_dir: Path | None = None) -> ReferenceDataProvider:
    """Construct the configured provider from ``reference_data`` settings."""
    root = base_dir or Path.cwd()
    kind = config.provider

    if kind == "null":
        return NullReferenceDataProvider(version=config.reference_context_version)

    iso_path = Path(config.iso3166_path)
    iso_path = iso_path if iso_path.is_absolute() else root / iso_path
    iso_provider = Iso3166Provider.from_csv(
        iso_path,
        version=config.reference_context_version,
        approved_for_production=False,
        ambiguous_alpha2_tokens=config.ambiguous_alpha2_tokens,
        trailing_country_token_window=config.trailing_country_token_window,
    )

    if kind == "iso3166":
        return iso_provider

    providers: list[ReferenceDataProvider] = [iso_provider]
    if config.swiftref_enabled:
        providers.append(SwiftRefProvider())
    return CompositeReferenceDataProvider(providers)


def build_town_country_provider(
    config: Any, base_dir: Path | None = None
) -> TownCountryProvider | None:
    """Build the Town/Country provider when enabled, else ``None``.

    Kept separate from :func:`build_provider` because this provider is applied
    *after* extraction rather than being fed into the prompt.
    """
    if not getattr(config, "town_country_enabled", False):
        return None

    root = base_dir or Path.cwd()
    path = Path(config.town_country_path)
    path = path if path.is_absolute() else root / path

    return TownCountryProvider.from_file(
        path,
        version=config.reference_context_version,
        configured_source=config.town_country_source,
        approved_for_production=config.town_country_approved_for_production,
        column_map=config.town_country_column_map,
    )


def find_iso_provider(provider: ReferenceDataProvider) -> Iso3166Provider | None:
    """Locate the ISO provider inside a possibly-composite provider."""
    if isinstance(provider, Iso3166Provider):
        return provider
    if isinstance(provider, CompositeReferenceDataProvider):
        for member in provider.providers:
            found = find_iso_provider(member)
            if found is not None:
                return found
    return None
