#!/usr/bin/env python3
"""Fetch bounded first-party DOL/ETA Weekly Claims artifacts and a manifest.

This acquisition-only program never connects to PostgreSQL.  PDF extraction is
explicitly delegated to a locally installed pdftotext executable; both the
original PDF and extracted parser input are hashed in the manifest.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import html.parser
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

from authoritative_acquisition import (
    AcquisitionRecord,
    DEFAULT_MAX_DOWNLOAD_BYTES,
    Download,
    ResourceRedirectError,
    canonical_https_url,
    digest,
    read_bounded,
    validate_final_resource,
    write_acquisition_manifest,
)


ARCHIVE_URL = "https://oui.doleta.gov/unemploy/archive.asp"
ALLOWED_HOST = "oui.doleta.gov"
PRESS_PATH = re.compile(r"^/press/(?P<year>[0-9]{4})/(?P<name>[0-9]{6}\.(?:asp|pdf))$", re.I)


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.links.append(value)


def canonical_press_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme.lower() != "https" or parsed.hostname != ALLOWED_HOST:
        raise ValueError(f"not an HTTPS first-party DOL/ETA URL: {value}")
    if parsed.query or parsed.fragment or not PRESS_PATH.fullmatch(parsed.path):
        raise ValueError(f"not a canonical immutable DOL/ETA occurrence URL: {value}")
    return urllib.parse.urlunsplit(("https", ALLOWED_HOST, parsed.path, "", ""))


def canonical_dol_eta_resource_url(value: str) -> str:
    return canonical_https_url(value, allowed_hosts={ALLOWED_HOST})


def fetch(
    url: str,
    canonicalize=canonical_dol_eta_resource_url,
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> Download:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ExpertAdvisor-authoritative-calendar-acquisition/1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        final_url = validate_final_resource(
            url, response.geturl(), canonicalize=canonicalize
        )
        data = read_bounded(
            response,
            requested_url=url,
            max_download_bytes=max_download_bytes,
        )
    return Download(url, final_url, data)


@functools.lru_cache(maxsize=None)
def archive_occurrences(year: int) -> tuple[tuple[str, int], ...]:
    selection = urllib.parse.urlencode(
        {"report": "press", "year": str(year)}
    ).encode("ascii")
    request = urllib.request.Request(
        ARCHIVE_URL,
        data=selection,
        headers={
            "User-Agent": "ExpertAdvisor-authoritative-calendar-acquisition/1",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        validate_final_resource(
            ARCHIVE_URL,
            response.geturl(),
            canonicalize=canonical_dol_eta_resource_url,
        )
        archive = read_bounded(response, requested_url=ARCHIVE_URL)

    parser = LinkParser()
    parser.feed(archive.decode("utf-8", errors="strict"))
    urls: set[tuple[str, int]] = set()
    for link in parser.links:
        absolute = urllib.parse.urljoin(ARCHIVE_URL, link)
        try:
            canonical = canonical_press_url(absolute)
        except ValueError:
            continue
        match = PRESS_PATH.fullmatch(urllib.parse.urlsplit(canonical).path)
        if match and int(match.group("year")) == year:
            urls.add((canonical, year))
    return tuple(sorted(urls))


def enumerate_archive(year: int) -> list[str]:
    urls = {
        canonical
        for canonical, found_year in archive_occurrences(year)
        if found_year == year
    }
    if not urls:
        raise RuntimeError(f"archive exposed no immutable occurrence links for {year}")
    return sorted(urls)


def sha256(data: bytes) -> str:
    return digest(data)


def write_new(path: pathlib.Path, data: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {path}")
    path.write_bytes(data)


def extractor_identity(executable: str) -> str:
    result = subprocess.run(
        [executable, "-v"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    first_line = result.stdout.splitlines()[0] if result.stdout.splitlines() else "unknown"
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", first_line).strip("-").lower()
    return "pdftotext-" + (normalized or "unknown")


def acquire(urls: list[str], output: pathlib.Path, pdftotext: str | None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output}")

    entries: list[list[str]] = []
    acquisition_records: list[AcquisitionRecord] = []
    extractor = extractor_identity(pdftotext) if pdftotext else None

    for url in sorted(set(urls)):
        canonical = canonical_press_url(url)
        match = PRESS_PATH.fullmatch(urllib.parse.urlsplit(canonical).path)
        assert match is not None
        relative_source = pathlib.Path(match.group("year")) / match.group("name").lower()
        source_path = output / relative_source
        source_path.parent.mkdir(parents=True, exist_ok=True)
        occurrence_identity = f"dol-eta-url:{match.group('year')}/{match.group('name').lower()}"
        try:
            download = fetch(canonical, canonical_press_url)
        except Exception as error:
            acquisition_records.append(AcquisitionRecord(
                canonical,
                error.final_url if isinstance(error, ResourceRedirectError) else "",
                occurrence_identity,
                "failed",
                diagnostic=str(error).replace("\t", " ").replace("\n", " | "),
            ))
            write_acquisition_manifest(output / "acquisition.tsv", acquisition_records)
            raise
        source_bytes = download.data
        write_new(source_path, source_bytes)
        source_digest = sha256(source_bytes)
        acquisition_records.append(AcquisitionRecord(
            canonical,
            download.final_url,
            occurrence_identity,
            "succeeded",
            source_digest,
        ))

        if source_path.suffix.lower() == ".asp":
            entries.append([
                relative_source.as_posix(), source_digest, "html", canonical,
                relative_source.as_posix(), source_digest, "none",
            ])
            continue

        if not pdftotext:
            raise RuntimeError(
                "PDF occurrence requires --pdftotext or a pdftotext executable on PATH"
            )
        relative_text = relative_source.with_suffix(".txt")
        text_path = output / relative_text
        with tempfile.NamedTemporaryFile(dir=text_path.parent, delete=False) as temporary:
            temporary_path = pathlib.Path(temporary.name)
        try:
            subprocess.run(
                [pdftotext, "-layout", "-enc", "UTF-8", str(source_path), str(temporary_path)],
                check=True,
            )
            text_bytes = temporary_path.read_bytes()
            if not text_bytes.strip():
                raise RuntimeError(f"pdftotext produced empty output: {canonical}")
            write_new(text_path, text_bytes)
        finally:
            temporary_path.unlink(missing_ok=True)
        entries.append([
            relative_text.as_posix(), sha256(text_bytes), "pdf_text", canonical,
            relative_source.as_posix(), source_digest, extractor or "",
        ])

    manifest_lines = [
        "manifest_version\t1",
        "parser_version\tdol_eta_weekly_claims_v3",
        "artifact_path\tartifact_sha256\tartifact_type\tsource_url\t"
        "source_artifact_path\tsource_artifact_sha256\textractor",
    ]
    manifest_lines.extend("\t".join(entry) for entry in entries)
    write_new(output / "manifest.tsv", ("\n".join(manifest_lines) + "\n").encode("utf-8"))
    write_acquisition_manifest(output / "acquisition.tsv", acquisition_records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--archive-year", action="append", type=int, default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pdftotext", default=shutil.which("pdftotext"))
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    urls = [canonical_press_url(value) for value in args.url]
    for year in args.archive_year:
        urls.extend(enumerate_archive(year))
    urls = sorted(set(urls))
    if args.limit is not None:
        urls = urls[: args.limit]
    if not urls:
        parser.error("at least one --url or --archive-year is required")

    acquire(urls, args.output_dir.resolve(), args.pdftotext)
    print(f"DOL_ETA_ACQUISITION_COMPLETE artifacts={len(urls)} manifest={args.output_dir.resolve() / 'manifest.tsv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"DOL_ETA_ACQUISITION_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
