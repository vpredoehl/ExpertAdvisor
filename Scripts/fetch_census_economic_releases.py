#!/usr/bin/env python3
"""Fetch bounded first-party Census economic-release PDFs and a manifest.

This acquisition-only program accepts explicit immutable occurrence URLs or
enumerates one of six official Census historical-release indexes. It never
connects to PostgreSQL. PDF extraction is delegated to a locally installed
pdftotext executable; both the original PDF and extracted parser input are
hashed in the shared ingestion manifest.
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
import time
import urllib.error
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


USER_AGENT = "ExpertAdvisor-authoritative-calendar-acquisition/1"
MAX_ARTIFACTS = 2000
RETRY_ATTEMPTS = 6
MAX_RETRY_DELAY_SECONDS = 60
ARCHIVES = {
    "retail-sales-advance": "https://www.census.gov/retail/marts/historic_releases.html",
    "new-residential-construction": "https://www.census.gov/construction/nrc/data/releases.html",
    "new-residential-sales": "https://www.census.gov/construction/nrs/data/releases.html",
    "manufacturers-orders": "https://www.census.gov/manufacturing/m3/historical_data/index.html",
    "durable-goods-advance": "https://www.census.gov/manufacturing/m3/adv/historical_data/index.html",
    "construction-spending": "https://www.census.gov/construction/c30/prpdf.html",
}

RETAIL = re.compile(
    r"^/retail/releases/historical/marts/adv(?P<yy>[0-9]{2})(?P<month>[0-9]{2})\.pdf$"
)
NRC = re.compile(
    r"^/construction/nrc/pdf/newresconst_(?P<year>[0-9]{4})(?P<month>[0-9]{2})\.pdf$"
)
NRS = re.compile(
    r"^/construction/nrs/pdf/newressales_(?P<year>[0-9]{4})(?P<month>[0-9]{2})\.pdf$"
)
M3 = re.compile(
    r"^/manufacturing/m3/historical_data/pressreleases/(?P<kind>adv|prel)/"
    r"(?P<year>[0-9]{4})/(?P<month>[a-z]{3})(?P<yy>[0-9]{2})(?P=kind)\.pdf$"
)
C30 = re.compile(
    r"^/construction/c30/pdf/pr(?P<year>[0-9]{4})(?P<month>[0-9]{2})\.pdf$"
)
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


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


def occurrence(value: str) -> tuple[str, str, int, int, str]:
    """Return canonical URL, family, reference year/month, and filename."""
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme.lower() != "https" or parsed.query or parsed.fragment:
        raise ValueError(f"not a canonical HTTPS Census occurrence URL: {value}")

    match = RETAIL.fullmatch(parsed.path)
    if parsed.hostname == "www2.census.gov" and match:
        year = 2000 + int(match.group("yy"))
        month = int(match.group("month"))
        family = "retail-sales-advance"
        host = "www2.census.gov"
    else:
        if parsed.hostname != "www.census.gov":
            raise ValueError(f"not a first-party Census occurrence URL: {value}")
        match = NRC.fullmatch(parsed.path)
        if match:
            family = "new-residential-construction"
            year = int(match.group("year"))
            month = int(match.group("month"))
        else:
            match = NRS.fullmatch(parsed.path)
            if match:
                family = "new-residential-sales"
                year = int(match.group("year"))
                month = int(match.group("month"))
            else:
                match = M3.fullmatch(parsed.path)
                if match:
                    year = int(match.group("year"))
                    if int(match.group("yy")) != year % 100:
                        raise ValueError(f"contradictory M3 occurrence path: {value}")
                    month = MONTHS[match.group("month")]
                    family = (
                        "durable-goods-advance"
                        if match.group("kind") == "adv"
                        else "manufacturers-orders"
                    )
                else:
                    match = C30.fullmatch(parsed.path)
                    if not match:
                        raise ValueError(f"unsupported Census occurrence URL: {value}")
                    family = "construction-spending"
                    year = int(match.group("year"))
                    month = int(match.group("month"))
        host = "www.census.gov"

    if year < 2000 or year > 2099 or month < 1 or month > 12:
        raise ValueError(f"unsupported Census reference period in URL: {value}")
    canonical = urllib.parse.urlunsplit(("https", host, parsed.path, "", ""))
    return canonical, family, year, month, pathlib.PurePosixPath(parsed.path).name


def canonical_census_resource_url(value: str) -> str:
    return canonical_https_url(
        value,
        allowed_hosts={"www.census.gov", "www2.census.gov"},
    )


def fetch(
    url: str,
    canonicalize=canonical_census_resource_url,
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> Download:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRY_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                final_url = validate_final_resource(
                    url, response.geturl(), canonicalize=canonicalize
                )
                data = read_bounded(
                    response,
                    requested_url=url,
                    max_download_bytes=max_download_bytes,
                )
            break
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            if attempt + 1 == RETRY_ATTEMPTS:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() \
                else 2 ** attempt
            time.sleep(min(delay, MAX_RETRY_DELAY_SECONDS))
    return Download(url, final_url, data)


@functools.lru_cache(maxsize=None)
def archive_occurrences(family: str) -> tuple[tuple[str, int], ...]:
    archive_url = ARCHIVES[family]
    parser = LinkParser()
    parser.feed(fetch(archive_url).data.decode("utf-8", errors="strict"))
    matches: set[tuple[str, int]] = set()
    for link in parser.links:
        absolute = urllib.parse.urljoin(archive_url, link)
        try:
            canonical, found_family, found_year, _, _ = occurrence(absolute)
        except ValueError:
            continue
        if found_family == family:
            matches.add((canonical, found_year))
    return tuple(sorted(matches))


def enumerate_archive(family: str, year: int) -> list[str]:
    matches = {
        canonical
        for canonical, found_year in archive_occurrences(family)
        if found_year == year
    }
    if not matches:
        raise RuntimeError(f"Census archive exposed no {family} releases for {year}")
    return sorted(matches)


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


def acquire(
    urls: list[str],
    output: pathlib.Path,
    pdftotext: str,
    resume: bool = False,
    request_delay: float = 0.0,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not resume:
        raise RuntimeError(f"output directory must be empty: {output}")

    extractor = extractor_identity(pdftotext)
    entries: list[list[str]] = []
    acquisition_records: list[AcquisitionRecord] = []
    for url in sorted(set(urls)):
        canonical, family, year, _, filename = occurrence(url)
        occurrence_identity = f"census-url:{urllib.parse.urlsplit(canonical).hostname}{urllib.parse.urlsplit(canonical).path}"
        relative_source = pathlib.Path(family) / str(year) / filename
        source_path = output / relative_source
        source_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.exists():
            source_bytes = source_path.read_bytes()
            if not source_bytes:
                raise RuntimeError(f"existing source artifact is empty: {source_path}")
            acquisition_records.append(AcquisitionRecord(
                canonical,
                canonical,
                occurrence_identity,
                "reused_verified_later_by_manifest_hash",
                sha256(source_bytes),
            ))
        else:
            relative_text = relative_source.with_suffix(".txt")
            if (output / relative_text).exists():
                raise RuntimeError(
                    f"extracted artifact exists without source artifact: {relative_text}"
                )
            try:
                download = fetch(canonical, lambda value: occurrence(value)[0])
            except Exception as error:
                acquisition_records.append(AcquisitionRecord(
                    canonical,
                    error.final_url if isinstance(error, ResourceRedirectError) else "",
                    occurrence_identity,
                    "failed",
                    diagnostic=str(error).replace("\t", " ").replace("\n", " | "),
                ))
                write_acquisition_manifest(
                    output / "acquisition.tsv", acquisition_records
                )
                raise
            source_bytes = download.data
            write_new(source_path, source_bytes)
            acquisition_records.append(AcquisitionRecord(
                canonical,
                download.final_url,
                occurrence_identity,
                "succeeded",
                sha256(source_bytes),
            ))
            if request_delay:
                time.sleep(request_delay)
        source_digest = sha256(source_bytes)

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
            if text_path.exists():
                if text_path.read_bytes() != text_bytes:
                    raise RuntimeError(
                        f"existing extracted artifact is not reproducible: {text_path}"
                    )
            else:
                write_new(text_path, text_bytes)
        finally:
            temporary_path.unlink(missing_ok=True)

        entries.append([
            relative_text.as_posix(), sha256(text_bytes), "pdf_text", canonical,
            relative_source.as_posix(), source_digest, extractor,
        ])

    manifest_lines = [
        "manifest_version\t1",
        "parser_version\tcensus_economic_release_v2",
        "artifact_path\tartifact_sha256\tartifact_type\tsource_url\t"
        "source_artifact_path\tsource_artifact_sha256\textractor",
    ]
    manifest_lines.extend("\t".join(entry) for entry in entries)
    manifest_bytes = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    manifest_path = output / "manifest.tsv"
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_bytes:
            raise RuntimeError(
                f"existing manifest differs from reproducible result: {manifest_path}"
            )
    else:
        write_new(manifest_path, manifest_bytes)
    acquisition_path = output / "acquisition.tsv"
    if acquisition_path.exists():
        acquisition_path.unlink()
    write_acquisition_manifest(acquisition_path, acquisition_records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--archive-year", action="append", type=int, default=[])
    parser.add_argument("--family", action="append", choices=sorted(ARCHIVES), default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pdftotext", default=shutil.which("pdftotext"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-delay", type=float, default=0.0)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.request_delay < 0:
        parser.error("--request-delay must be non-negative")
    if args.archive_year and not args.family:
        parser.error("--archive-year requires at least one --family")
    if args.family and not args.archive_year:
        parser.error("--family requires at least one --archive-year")
    if not args.pdftotext:
        parser.error("--pdftotext or a pdftotext executable on PATH is required")

    urls = [occurrence(value)[0] for value in args.url]
    for year in sorted(set(args.archive_year)):
        if year < 2000 or year > 2099:
            parser.error("--archive-year must be between 2000 and 2099")
        for family in sorted(set(args.family)):
            urls.extend(enumerate_archive(family, year))
    urls = sorted(set(urls))
    if args.limit is not None:
        urls = urls[: args.limit]
    if not urls:
        parser.error("at least one --url or --archive-year/--family selection is required")
    if len(urls) > MAX_ARTIFACTS:
        parser.error(f"selection exceeds the safety cap of {MAX_ARTIFACTS} artifacts")

    acquire(
        urls,
        args.output_dir.resolve(),
        args.pdftotext,
        resume=args.resume,
        request_delay=args.request_delay,
    )
    print(
        f"CENSUS_ACQUISITION_COMPLETE artifacts={len(urls)} "
        f"manifest={args.output_dir.resolve() / 'manifest.tsv'}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"CENSUS_ACQUISITION_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
