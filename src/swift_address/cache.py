"""Stable cache keys and a JSONL result cache.

The cache is what makes the pipeline economical and restartable. Its key binds
a result to everything that could change that result::

    sha256(prompt_version | model | normalized_address | reference_context_version)

Because the reference-context version is part of the key, swapping the ISO
dataset (or wiring up SWIFTRef) invalidates stale extractions instead of
silently reusing conclusions drawn without that evidence.

Entries are appended as JSON lines and checkpointed periodically, so a killed
run resumes without repeating completed calls. The raw address is stored in the
cache because the cache *is* the audit record; operational logs get the hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from .cleaning import normalize_whitespace

__all__ = ["AddressCache", "CacheEntry", "address_hash", "make_cache_key"]

logger = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def address_hash(address: str) -> str:
    """Stable hash used to reference an address in logs and error records.

    Operational logs must not carry raw customer/payment addresses; they carry
    this instead, which still joins back to the cache and the error sidecar.
    """
    return _sha256(normalize_whitespace(address).upper())


def make_cache_key(
    *,
    prompt_version: str,
    model: str,
    address: str,
    reference_context_version: str,
) -> str:
    """Build the stable cache key for one unique address."""
    normalized = normalize_whitespace(address).upper()
    return _sha256(
        "|".join((prompt_version, model, normalized, reference_context_version))
    )


@dataclass(frozen=True)
class CacheEntry:
    """One cached extraction, including the audit metadata behind it."""

    key: str
    address_hash: str
    address: str
    prompt_version: str
    model: str
    reference_context_version: str
    response: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "key": self.key,
                "address_hash": self.address_hash,
                "address": self.address,
                "prompt_version": self.prompt_version,
                "model": self.model,
                "reference_context_version": self.reference_context_version,
                "response": dict(self.response),
                "metadata": dict(self.metadata),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, line: str) -> "CacheEntry":
        data = json.loads(line)
        return cls(
            key=data["key"],
            address_hash=data.get("address_hash", ""),
            address=data.get("address", ""),
            prompt_version=data.get("prompt_version", ""),
            model=data.get("model", ""),
            reference_context_version=data.get("reference_context_version", ""),
            response=data.get("response", {}),
            metadata=data.get("metadata", {}),
        )


class AddressCache:
    """Thread-safe in-memory cache with optional JSONL persistence.

    ``enabled=False`` yields a working, non-persisting cache so a run can be
    forced to re-extract without deleting files.
    """

    def __init__(self, path: str | Path | None = None, *, enabled: bool = True) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self._path = Path(path) if path else None
        self._enabled = enabled
        self._lock = threading.Lock()
        self._pending: list[CacheEntry] = []
        self._hits = 0
        self._misses = 0

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> int:
        """Load persisted entries. Corrupt lines are skipped, not fatal."""
        if not self._enabled or self._path is None or not self._path.exists():
            return 0
        loaded = 0
        skipped = 0
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = CacheEntry.from_json(line)
                except (json.JSONDecodeError, KeyError):
                    skipped += 1
                    continue
                self._entries[entry.key] = entry
                loaded += 1
        if skipped:
            logger.warning("skipped %d unreadable cache line(s) in %s", skipped, self._path)
        logger.info("loaded %d cached extraction(s)", loaded)
        return loaded

    def flush(self) -> int:
        """Append pending entries to the JSONL file. Returns the count written."""
        if not self._enabled or self._path is None:
            with self._lock:
                self._pending.clear()
            return 0
        with self._lock:
            pending, self._pending = self._pending, []
        if not pending:
            return 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            for entry in pending:
                handle.write(entry.to_json() + "\n")
        return len(pending)

    # -- access ------------------------------------------------------------

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
            else:
                self._hits += 1
            return entry

    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            self._entries[entry.key] = entry
            if self._enabled and self._path is not None:
                self._pending.append(entry)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __iter__(self) -> Iterator[CacheEntry]:
        with self._lock:
            return iter(list(self._entries.values()))

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
            }
