#!/usr/bin/env python3
"""Fetch bounded first-party BEA occurrence pages and a Phase 2 manifest.

This acquisition-only program enumerates BEA's news archive or accepts explicit
occurrence URLs. It never connects to PostgreSQL and does not transform the
downloaded HTML before hashing and manifest handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import html.parser
import pathlib
import re
import sys
import urllib.parse
import urllib.request


ARCHIVE_URL = "https://www.bea.gov/news/archive"
ALLOWED_HOST = "www.bea.gov"
PRODUCT_IDS = {
    "gdp": "451",
    "personal-income-outlays": "476",
}
OCCURRENCE_PATH = re.compile(
    r"^/news/(?P<year>[0-9]{4})/(?P<slug>[a-z0-9][a-z0-9-]*)$"
)


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


def canonical_occurrence_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme.lower() != "https" or parsed.hostname not in {"bea.gov", ALLOWED_HOST}:
        raise ValueError(f"not an HTTPS first-party BEA URL: {value}")
    if parsed.query or parsed.fragment or not OCCURRENCE_PATH.fullmatch(parsed.path):
        raise ValueError(f"not a canonical immutable BEA occurrence URL: {value}")
    return urllib.parse.urlunsplit(("https", ALLOWED_HOST, parsed.path, "", ""))


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ExpertAdvisor-authoritative-calendar-acquisition/1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        final = urllib.parse.urlsplit(response.geturl())
        if final.scheme.lower() != "https" or final.hostname not in {"bea.gov", ALLOWED_HOST}:
            raise RuntimeError(f"download redirected outside first-party BEA: {response.geturl()}")
        data = response.read()
    if not data:
        raise RuntimeError(f"empty download: {url}")
    return data


def archive_page(product_id: str, page: int) -> tuple[list[str], bool]:
    query = urllib.parse.urlencode(
        {
            "created_1": "All",
            "field_related_product_target_id": product_id,
            "title": "",
            "page": str(page),
        }
    )
    page_url = ARCHIVE_URL + "?" + query
    parser = LinkParser()
    parser.feed(fetch(page_url).decode("utf-8", errors="strict"))
    occurrences: list[str] = []
    next_page = False
    for link in parser.links:
        absolute = urllib.parse.urljoin(ARCHIVE_URL, link)
        parsed = urllib.parse.urlsplit(absolute)
        if OCCURRENCE_PATH.fullmatch(parsed.path):
            occurrences.append(canonical_occurrence_url(
                urllib.parse.urlunsplit(("https", ALLOWED_HOST, parsed.path, "", ""))
            ))
        query_values = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/news/archive" and query_values.get("page") == [str(page + 1)]:
            next_page = True
    return sorted(set(occurrences)), next_page


def enumerate_archive(family: str, year: int) -> list[str]:
    product_id = PRODUCT_IDS[family]
    matches: set[str] = set()
    for page in range(100):
        urls, has_next = archive_page(product_id, page)
        for url in urls:
            path_match = OCCURRENCE_PATH.fullmatch(urllib.parse.urlsplit(url).path)
            assert path_match is not None
            if int(path_match.group("year")) == year:
                matches.add(url)
        if not has_next:
            break
    else:
        raise RuntimeError("BEA archive pagination exceeded safety limit")
    if not matches:
        raise RuntimeError(f"BEA archive exposed no {family} occurrence links for {year}")
    return sorted(matches)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new(path: pathlib.Path, data: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {path}")
    path.write_bytes(data)


def acquire(urls: list[str], output: pathlib.Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output}")

    entries: list[list[str]] = []
    for url in sorted(set(urls)):
        canonical = canonical_occurrence_url(url)
        path_match = OCCURRENCE_PATH.fullmatch(urllib.parse.urlsplit(canonical).path)
        assert path_match is not None
        relative = pathlib.Path(path_match.group("year")) / (path_match.group("slug") + ".html")
        artifact_path = output / relative
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = fetch(canonical)
        write_new(artifact_path, artifact)
        digest = sha256(artifact)
        entries.append([
            relative.as_posix(), digest, "html", canonical,
            relative.as_posix(), digest, "none",
        ])

    manifest_lines = [
        "manifest_version\t1",
        "parser_version\tbea_economic_release_v1",
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
    parser.add_argument("--family", action="append", choices=sorted(PRODUCT_IDS), default=[])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.archive_year and not args.family:
        parser.error("--archive-year requires at least one --family")
    urls = [canonical_occurrence_url(value) for value in args.url]
    for year in args.archive_year:
        for family in sorted(set(args.family)):
            urls.extend(enumerate_archive(family, year))
    urls = sorted(set(urls))
    if args.limit is not None:
        urls = urls[:args.limit]
    if not urls:
        parser.error("at least one --url or --archive-year/--family selection is required")

    acquire(urls, args.output_dir.resolve())
    print(
        f"BEA_ACQUISITION_COMPLETE artifacts={len(urls)} "
        f"manifest={args.output_dir.resolve() / 'manifest.tsv'}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"BEA_ACQUISITION_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
