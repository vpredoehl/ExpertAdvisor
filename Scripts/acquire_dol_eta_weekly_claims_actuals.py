#!/usr/bin/env python3
"""Resumable acquisition of release-specific DOL/ETA Weekly Claims evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from typing import Callable, Iterable, Mapping

from authoritative_acquisition import Download
from fetch_dol_eta_weekly_claims import (
    archive_occurrences,
    canonical_press_url,
    extractor_identity,
    fetch,
)


PARSER_VERSION = "dol_eta_weekly_claims_actual_v1"
UTC = dt.timezone.utc
KNOWN_FILENAME_YEAR_EXCEPTION = (
    "https://oui.doleta.gov/press/2019/010318.pdf"
)


def canonical_instant(value: dt.datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("retrieval_timestamp_missing_timezone")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _release_date(url: str) -> str:
    canonical = canonical_press_url(url)
    parsed = urllib.parse.urlsplit(canonical)
    year = parsed.path.split("/")[2]
    compact = pathlib.PurePosixPath(parsed.path).stem
    # The official 2019-01-03 occurrence is retained at 2019/010318.pdf.
    # Treat the directory plus month/day as the acquisition identity; the
    # parser independently requires the embedded release date to agree.
    if compact[4:] != year[2:] and canonical != KNOWN_FILENAME_YEAR_EXCEPTION:
        raise ValueError("dol_eta_artifact_identity_year_mismatch")
    return f"{year}-{compact[:2]}-{compact[2:4]}"


def _identity(url: str) -> str:
    parsed = urllib.parse.urlsplit(canonical_press_url(url))
    relative = parsed.path.removeprefix("/press/").lower()
    return "dol_eta:press:" + relative.replace("/", ":")


def _paths(output: pathlib.Path, url: str) -> tuple[pathlib.Path, pathlib.Path]:
    parsed = urllib.parse.urlsplit(canonical_press_url(url))
    relative = pathlib.PurePosixPath(parsed.path.removeprefix("/press/").lower())
    source = output.joinpath(*relative.parts)
    parser = source if source.suffix == ".asp" else source.with_suffix(".txt")
    return source, parser


def _read_manifest(path: pathlib.Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"dol_eta_acquisition_manifest_invalid:{line_number}"
            ) from error
        if row.get("manifest_version") != 1 or row.get("parser_version") != PARSER_VERSION:
            raise ValueError("dol_eta_acquisition_manifest_contract_mismatch")
        url = canonical_press_url(str(row.get("source_url", "")))
        if url in rows:
            raise ValueError("dol_eta_acquisition_manifest_duplicate_url")
        rows[url] = row
    return rows


def _write_manifest(
    path: pathlib.Path,
    rows: Mapping[str, Mapping[str, object]],
) -> None:
    payload = "".join(
        json.dumps(dict(rows[url]), sort_keys=True, separators=(",", ":")) + "\n"
        for url in sorted(rows)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_retained(
    repo_root: pathlib.Path,
    output: pathlib.Path,
    url: str,
    row: Mapping[str, object],
) -> None:
    source, parser = _paths(output, url)
    if row.get("release_date") != _release_date(url):
        raise ValueError("dol_eta_retained_release_date_conflict")
    if row.get("requested_url") != url or row.get("final_url") != url:
        raise ValueError("dol_eta_retained_resource_identity_conflict")
    if row.get("source_artifact_identity") != _identity(url):
        raise ValueError("dol_eta_retained_artifact_identity_conflict")
    expected_source = repo_root / pathlib.Path(str(row["local_artifact_path"]))
    expected_parser = repo_root / pathlib.Path(str(row["parser_artifact_path"]))
    if source.resolve() != expected_source.resolve() or parser.resolve() != expected_parser.resolve():
        raise ValueError("dol_eta_retained_path_conflict")
    if not source.is_file() or not parser.is_file():
        raise ValueError("dol_eta_retained_artifact_missing")
    if _sha256(source.read_bytes()) != row.get("source_artifact_sha256"):
        raise ValueError("dol_eta_retained_source_hash_conflict")
    if _sha256(parser.read_bytes()) != row.get("parser_artifact_sha256"):
        raise ValueError("dol_eta_retained_parser_hash_conflict")
    extractor = str(row.get("extractor", ""))
    if source.suffix == ".asp":
        if parser != source or extractor != "none":
            raise ValueError("dol_eta_retained_html_provenance_conflict")
    elif parser != source.with_suffix(".txt") or not extractor.startswith("pdftotext-"):
        raise ValueError("dol_eta_retained_pdf_provenance_conflict")


def _extract_pdf(
    source: pathlib.Path,
    parser_path: pathlib.Path,
    pdftotext: str,
) -> bytes:
    with tempfile.NamedTemporaryFile(
        dir=parser_path.parent, delete=False
    ) as temporary_file:
        temporary = pathlib.Path(temporary_file.name)
    try:
        result = subprocess.run(
            [pdftotext, "-layout", "-enc", "UTF-8", str(source), str(temporary)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "pdftotext_failed:" + " ".join(result.stderr.split())
            )
        data = temporary.read_bytes()
        if not data.strip():
            raise RuntimeError("pdftotext_empty")
        return data
    finally:
        temporary.unlink(missing_ok=True)


def acquire(
    urls: Iterable[str],
    output: pathlib.Path,
    *,
    repo_root: pathlib.Path,
    pdftotext: str | None,
    fetcher: Callable[[str], Download] = lambda url: fetch(
        url, canonical_press_url
    ),
    observed_at: str | None = None,
) -> tuple[int, int, pathlib.Path]:
    """Acquire every URL, preserving successes and recording every failure."""
    repo_root = repo_root.resolve()
    output = output.resolve()
    try:
        output.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("dol_eta_output_outside_repository") from error
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "actual_acquisition_manifest.jsonl"
    rows = _read_manifest(manifest_path)
    extractor = extractor_identity(pdftotext) if pdftotext else None
    succeeded = 0
    failed = 0

    for requested in sorted({canonical_press_url(url) for url in urls}):
        existing = rows.get(requested)
        if existing and existing.get("http_acquisition_result") == "succeeded":
            _verify_retained(repo_root, output, requested, existing)
            succeeded += 1
            continue

        source_path, parser_path = _paths(output, requested)
        if source_path.exists() or (
            parser_path != source_path and parser_path.exists()
        ):
            raise ValueError("dol_eta_unmanifested_existing_artifact")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        retrieval = observed_at or canonical_instant(dt.datetime.now(tz=UTC))
        base: dict[str, object] = {
            "manifest_version": 1,
            "parser_version": PARSER_VERSION,
            "release_date": _release_date(requested),
            "source_url": requested,
            "requested_url": requested,
            "final_url": None,
            "source_artifact_identity": _identity(requested),
            "http_acquisition_result": "failed",
            "local_artifact_path": source_path.relative_to(repo_root).as_posix(),
            "source_artifact_sha256": None,
            "parser_artifact_path": parser_path.relative_to(repo_root).as_posix(),
            "parser_artifact_sha256": None,
            "extractor": "none" if source_path.suffix == ".asp" else extractor,
            "retrieved_at": retrieval,
            "source_publication_timestamp_evidence": (
                "embedded_embargo_header_pending_parser_validation"
            ),
            "mapped_economic_event_id": None,
            "mapped_source_event_id": None,
            "extraction_disposition": "extraction_failed",
            "import_eligibility_disposition": "extraction_failed",
            "diagnostic": None,
        }
        try:
            download = fetcher(requested)
            final_url = canonical_press_url(download.final_url)
            if final_url != requested:
                raise RuntimeError("dol_eta_final_resource_identity_mismatch")
            source_path.write_bytes(download.data)
            if source_path.suffix == ".asp":
                parser_bytes = download.data
            else:
                if not pdftotext:
                    raise RuntimeError("pdftotext_required")
                parser_bytes = _extract_pdf(source_path, parser_path, pdftotext)
                parser_path.write_bytes(parser_bytes)
            base.update({
                "final_url": final_url,
                "http_acquisition_result": "succeeded",
                "source_artifact_sha256": _sha256(download.data),
                "parser_artifact_sha256": _sha256(parser_bytes),
                "extraction_disposition": "pending_parser_validation",
                "import_eligibility_disposition": "pending_catalog_mapping",
            })
            succeeded += 1
        except Exception as error:
            source_path.unlink(missing_ok=True)
            if parser_path != source_path:
                parser_path.unlink(missing_ok=True)
            base["diagnostic"] = str(error).replace("\n", " | ")
            failed += 1
        rows[requested] = base
        _write_manifest(manifest_path, rows)

    _write_manifest(manifest_path, rows)
    return succeeded, failed, manifest_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--repo-root", type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--archive-year", action="append", type=int, default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pdftotext", default=shutil.which("pdftotext"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    urls = [canonical_press_url(url) for url in args.url]
    for year in args.archive_year:
        urls.extend(url for url, found_year in archive_occurrences(year) if found_year == year)
    urls = sorted(set(urls))
    if args.limit is not None:
        urls = urls[:args.limit]
    if not urls:
        parser.error("at least one --url or --archive-year is required")
    succeeded, failed, manifest = acquire(
        urls, args.output_dir, repo_root=args.repo_root,
        pdftotext=args.pdftotext
    )
    print(
        "DOL_ETA_WEEKLY_CLAIMS_ACTUAL_ACQUISITION"
        f",discovered={len(urls)},succeeded={succeeded},failed={failed}"
        f",manifest={manifest}"
    )
    return 2 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"DOL_ETA_WEEKLY_CLAIMS_ACTUAL_ACQUISITION_FAILED:{error}", file=sys.stderr)
        raise SystemExit(1)
