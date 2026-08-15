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
    "build_provider",
    "find_iso_provider",
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
        record = self.record(code)
        if record is None:
            return False

        for name in record.presence_names:
            if name and contains_token_phrase(address, name):
                return True

        tokens = tokens_casefolded(address)
        alpha2 = record.alpha2
        if alpha2 not in tokens:
            return False
        if alpha2 not in self._ambiguous:
            return True
        # Ambiguous code: require trailing country position.
        window = tokens[-self._trailing_window :]
        return alpha2 in window


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
