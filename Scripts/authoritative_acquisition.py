#!/usr/bin/env python3
"""Shared fail-closed helpers for first-party historical acquisition."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import pathlib
import urllib.parse
from collections.abc import Callable
from typing import BinaryIO


DEFAULT_MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
ACQUISITION_COLUMNS = (
    "requested_url",
    "final_url",
    "occurrence_identity",
    "acquisition_status",
    "artifact_sha256",
    "diagnostic",
)


@dataclasses.dataclass(frozen=True)
class Download:
    requested_url: str
    final_url: str
    data: bytes


@dataclasses.dataclass(frozen=True)
class AcquisitionRecord:
    requested_url: str
    final_url: str
    occurrence_identity: str
    acquisition_status: str
    artifact_sha256: str = ""
    diagnostic: str = ""


class ResourceRedirectError(RuntimeError):
    def __init__(self, message: str, requested_url: str, final_url: str) -> None:
        super().__init__(message)
        self.requested_url = requested_url
        self.final_url = final_url


def canonical_https_url(
    value: str,
    *,
    allowed_hosts: set[str],
    host_aliases: dict[str, str] | None = None,
) -> str:
    """Normalize only explicitly allowed HTTPS host aliases."""
    parsed = urllib.parse.urlsplit(value)
    host = (parsed.hostname or "").lower()
    aliases = host_aliases or {}
    host = aliases.get(host, host)
    if parsed.scheme.lower() != "https" or host not in allowed_hosts:
        raise ValueError(f"not an allowed first-party HTTPS URL: {value}")
    port = parsed.port
    if port not in {None, 443}:
        raise ValueError(f"non-default HTTPS port is not allowed: {value}")
    return urllib.parse.urlunsplit(
        ("https", host, parsed.path or "/", parsed.query, parsed.fragment)
    )


def validate_final_resource(
    requested_url: str,
    final_url: str,
    *,
    canonicalize: Callable[[str], str],
) -> str:
    """Return the final canonical URL only when it is the requested resource."""
    requested = canonicalize(requested_url)
    try:
        final = canonicalize(final_url)
    except (TypeError, ValueError) as error:
        raise ResourceRedirectError(
            f"redirect final resource is not canonical: requested={requested_url} "
            f"final={final_url}",
            requested_url,
            final_url,
        ) from error
    if final != requested:
        raise ResourceRedirectError(
            f"redirect changed requested resource: requested={requested} final={final}",
            requested_url,
            final_url,
        )
    return final


def read_bounded(
    response: BinaryIO,
    *,
    requested_url: str,
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> bytes:
    if max_download_bytes <= 0:
        raise ValueError("max_download_bytes must be positive")
    data = response.read(max_download_bytes + 1)
    if len(data) > max_download_bytes:
        raise RuntimeError(
            f"download exceeds {max_download_bytes} byte safety limit: "
            f"{requested_url}"
        )
    if not data:
        raise RuntimeError(f"empty download: {requested_url}")
    return data


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_acquisition_manifest(
    path: pathlib.Path,
    records: list[AcquisitionRecord],
) -> None:
    """Write deterministic occurrence-specific acquisition evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=ACQUISITION_COLUMNS, delimiter="\t")
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda item: (item.requested_url, item.occurrence_identity),
        ):
            writer.writerow(dataclasses.asdict(record))
