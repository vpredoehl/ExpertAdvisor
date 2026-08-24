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


USER_AGENT = "ExpertAdvisor-authoritative-calendar-acquisition/1"
MAX_ARTIFACTS = 100
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


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        final = urllib.parse.urlsplit(response.geturl())
        if final.scheme.lower() != "https" or final.hostname not in {
            "www.census.gov", "www2.census.gov"
        }:
            raise RuntimeError(
                f"download redirected outside first-party Census: {response.geturl()}"
            )
        data = response.read()
    if not data:
        raise RuntimeError(f"empty download: {url}")
    return data


def enumerate_archive(family: str, year: int) -> list[str]:
    archive_url = ARCHIVES[family]
    parser = LinkParser()
    parser.feed(fetch(archive_url).decode("utf-8", errors="strict"))
    matches: set[str] = set()
    for link in parser.links:
        absolute = urllib.parse.urljoin(archive_url, link)
        try:
            canonical, found_family, found_year, _, _ = occurrence(absolute)
        except ValueError:
            continue
        if found_family == family and found_year == year:
            matches.add(canonical)
    if not matches:
        raise RuntimeError(f"Census archive exposed no {family} releases for {year}")
    return sorted(matches)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def acquire(urls: list[str], output: pathlib.Path, pdftotext: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output}")

    extractor = extractor_identity(pdftotext)
    entries: list[list[str]] = []
    for url in sorted(set(urls)):
        canonical, family, year, _, filename = occurrence(url)
        relative_source = pathlib.Path(family) / str(year) / filename
        source_path = output / relative_source
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_bytes = fetch(canonical)
        write_new(source_path, source_bytes)
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
            write_new(text_path, text_bytes)
        finally:
            temporary_path.unlink(missing_ok=True)

        entries.append([
            relative_text.as_posix(), sha256(text_bytes), "pdf_text", canonical,
            relative_source.as_posix(), source_digest, extractor,
        ])

    manifest_lines = [
        "manifest_version\t1",
        "parser_version\tcensus_economic_release_v1",
        "artifact_path\tartifact_sha256\tartifact_type\tsource_url\t"
        "source_artifact_path\tsource_artifact_sha256\textractor",
    ]
    manifest_lines.extend("\t".join(entry) for entry in entries)
    write_new(output / "manifest.tsv", ("\n".join(manifest_lines) + "\n").encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--archive-year", action="append", type=int, default=[])
    parser.add_argument("--family", action="append", choices=sorted(ARCHIVES), default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pdftotext", default=shutil.which("pdftotext"))
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
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

    acquire(urls, args.output_dir.resolve(), args.pdftotext)
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
