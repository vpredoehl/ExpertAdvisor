#!/usr/bin/env python3
"""Fetch bounded first-party Federal Reserve publication artifacts.

The caller can select explicit occurrence URLs or enumerate authoritative
annual press-release and Beige Book indexes.  This acquisition-only program
never opens PostgreSQL and writes the shared version-1 manifest consumed by
the common economic-event import path.
"""

from __future__ import annotations

import argparse
import datetime
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
    Download,
    ResourceRedirectError,
    canonical_https_url,
    digest,
    read_bounded,
    validate_final_resource,
    write_acquisition_manifest,
)


USER_AGENT = "ExpertAdvisor-authoritative-calendar-acquisition/1"
ALLOWED_HOST = "www.federalreserve.gov"
MAX_ARTIFACTS = 500
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
PRESS_ARCHIVE_INDEX = "https://www.federalreserve.gov/newsevents/pressreleases.htm"
BEIGE_BOOK_ARCHIVE_INDEX = (
    "https://www.federalreserve.gov/monetarypolicy/beige-book-archive.htm"
)
BEIGE_BOOK_CURRENT_INDEX = (
    "https://www.federalreserve.gov/monetarypolicy/publications/"
    "beige-book-default.htm"
)
FAMILIES = ("beige_book", "fomc_minutes", "fomc_statement")
PRESS_RELEASE = re.compile(
    r"^/newsevents/pressreleases/monetary(?P<date>[0-9]{8})(?P<suffix>[a-z])\.htm$"
)
PRESS_ARCHIVE_PATH = re.compile(
    r"^/newsevents/pressreleases/(?P<year>[0-9]{4})"
    r"(?P<kind>all|-press|-press-fomc)\.htm$"
)
BEIGE_BOOK_YEAR_PATH = re.compile(
    r"^/monetarypolicy/beigebook(?P<year>[0-9]{4})\.htm$"
)
BEIGE_BOOK_PATTERNS = (
    re.compile(
        r"^/fomc/beigebook/(?P<year>[0-9]{4})/(?P<date>[0-9]{8})/"
        r"fullreport(?P=date)\.pdf$"
    ),
    re.compile(
        r"^/monetarypolicy/beigebook/files/"
        r"(?:fullreport|Beige[Bb]ook_)(?P<date>[0-9]{8})\.pdf$"
    ),
    re.compile(r"^/monetarypolicy/files/Beige[Bb]ook_(?P<date>[0-9]{8})\.pdf$"),
    re.compile(r"^/publications/files/BeigeBook_(?P<date>[0-9]{8})\.pdf$"),
)


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        self._href = next(
            (value for name, value in attrs if name.lower() == "href" and value),
            None,
        )
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.links.append((self._href, " ".join(" ".join(self._text).split())))
            self._href = None
            self._text = []


def parse_links(url: str) -> list[tuple[str, str]]:
    parser = LinkParser()
    parser.feed(fetch(url).data.decode("utf-8-sig", errors="strict"))
    return [
        (urllib.parse.urljoin(url, href), text)
        for href, text in parser.links
    ]


def annual_press_archive_urls(year: int) -> list[str]:
    matches: dict[str, str] = {}
    for absolute, _ in parse_links(PRESS_ARCHIVE_INDEX):
        parsed = urllib.parse.urlsplit(absolute)
        match = PRESS_ARCHIVE_PATH.fullmatch(parsed.path)
        if (
            parsed.scheme.lower() == "https"
            and parsed.hostname == ALLOWED_HOST
            and match
            and int(match.group("year")) == year
            and not parsed.query
            and not parsed.fragment
        ):
            matches[match.group("kind")] = absolute
    if not matches:
        raise RuntimeError(
            f"Federal Reserve press archive exposed no annual index for {year}"
        )
    if "all" in matches:
        return [matches["all"]]
    # Newer archives split FOMC statements from the general annual index that
    # carries FOMC minutes.  Both indexes are required for the supported raw
    # families and their union is de-duplicated below.
    return [matches[kind] for kind in ("-press", "-press-fomc") if kind in matches]


def classify_press_release(title: str) -> str | None:
    if re.match(
        r"^(?:Federal Reserve (?:Board )?issues )?FOMC statement\b",
        title,
        re.I,
    ):
        return "fomc_statement"
    if re.match(
        r"^Minutes of (?:the )?Federal Open Market Committee\b",
        title,
        re.I,
    ):
        return "fomc_minutes"
    return None


def enumerate_press_archive(year: int, families: set[str]) -> list[tuple[str, str]]:
    selections: set[tuple[str, str]] = set()
    for archive_url in annual_press_archive_urls(year):
        for absolute, title in parse_links(archive_url):
            parsed = urllib.parse.urlsplit(absolute)
            match = PRESS_RELEASE.fullmatch(parsed.path)
            if (
                parsed.scheme.lower() != "https"
                or parsed.hostname != ALLOWED_HOST
                or not match
                or int(match.group("date")[:4]) != year
                or parsed.query
                or parsed.fragment
            ):
                continue
            family = classify_press_release(title)
            if family in families:
                canonical, _ = canonical_occurrence_url(absolute, family)
                selections.add((family, canonical))
    for family in sorted(families & {"fomc_statement", "fomc_minutes"}):
        if not any(found == family for found, _ in selections):
            raise RuntimeError(
                f"Federal Reserve annual index exposed no {family} releases for {year}"
            )
    return sorted(selections)


def beige_book_year_index_url(year: int) -> str:
    for absolute, _ in parse_links(BEIGE_BOOK_ARCHIVE_INDEX):
        parsed = urllib.parse.urlsplit(absolute)
        match = BEIGE_BOOK_YEAR_PATH.fullmatch(parsed.path)
        if (
            parsed.scheme.lower() == "https"
            and parsed.hostname == ALLOWED_HOST
            and match
            and int(match.group("year")) == year
            and not parsed.query
            and not parsed.fragment
        ):
            return absolute
    return BEIGE_BOOK_CURRENT_INDEX


def enumerate_beige_book_archive(year: int) -> list[tuple[str, str]]:
    index_url = beige_book_year_index_url(year)
    selections: set[tuple[str, str]] = set()
    for absolute, _ in parse_links(index_url):
        try:
            canonical, identity = canonical_occurrence_url(absolute, "beige_book")
        except ValueError:
            continue
        if identity[10:14] == str(year):
            selections.add(("beige_book", canonical))
    if not selections:
        raise RuntimeError(
            f"Federal Reserve Beige Book index exposed no releases for {year}"
        )
    return sorted(selections)


def enumerate_archive(year: int, families: set[str]) -> list[tuple[str, str]]:
    if year < 1996 or year > datetime.date.today().year:
        raise ValueError("Federal Reserve archive year is outside supported indexes")
    selections = enumerate_press_archive(year, families)
    if "beige_book" in families:
        selections.extend(enumerate_beige_book_archive(year))
    return sorted(set(selections))


def canonical_occurrence_url(value: str, family: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname != ALLOWED_HOST
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"not a canonical HTTPS Federal Reserve URL: {value}")

    if family in {"fomc_statement", "fomc_minutes"}:
        match = PRESS_RELEASE.fullmatch(parsed.path)
        if not match:
            raise ValueError(f"unsupported Federal Reserve press-release URL: {value}")
        identity = "monetary" + match.group("date") + match.group("suffix")
        compact_date = match.group("date")
    elif family == "beige_book":
        match = next(
            (pattern.fullmatch(parsed.path) for pattern in BEIGE_BOOK_PATTERNS
             if pattern.fullmatch(parsed.path)),
            None,
        )
        if not match:
            raise ValueError(f"unsupported Federal Reserve Beige Book URL: {value}")
        if match.groupdict().get("year") and match.group("year") != match.group("date")[:4]:
            raise ValueError(f"contradictory Beige Book occurrence URL: {value}")
        identity = "beigebook-" + match.group("date")
        compact_date = match.group("date")
    else:
        raise ValueError(f"unsupported Federal Reserve artifact family: {family}")

    try:
        datetime.datetime.strptime(compact_date, "%Y%m%d").date()
    except ValueError as error:
        raise ValueError(
            f"invalid Federal Reserve occurrence date: {value}"
        ) from error

    canonical = urllib.parse.urlunsplit(("https", ALLOWED_HOST, parsed.path, "", ""))
    return canonical, identity


def canonical_federal_resource_url(value: str) -> str:
    return canonical_https_url(value, allowed_hosts={ALLOWED_HOST})


def fetch(
    url: str,
    canonicalize=canonical_federal_resource_url,
    max_download_bytes: int = MAX_DOWNLOAD_BYTES,
) -> Download:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
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
    first = result.stdout.splitlines()[0] if result.stdout.splitlines() else "unknown"
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", first).strip("-").lower()
    return "pdftotext-" + (normalized or "unknown")


def acquire(
    selections: list[tuple[str, str]],
    output: pathlib.Path,
    pdftotext: str | None,
) -> None:
    if not selections:
        raise ValueError("at least one Federal Reserve occurrence is required")
    canonical_selections = sorted(
        {(family, *canonical_occurrence_url(url, family)) for family, url in selections}
    )
    if len(canonical_selections) > MAX_ARTIFACTS:
        raise ValueError(f"selection exceeds the safety cap of {MAX_ARTIFACTS} artifacts")
    if any(family == "beige_book" for family, _, _ in canonical_selections) and not pdftotext:
        raise ValueError("pdftotext is required for Beige Book PDF acquisition")

    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output}")

    extractor = extractor_identity(pdftotext) if pdftotext and any(
        family == "beige_book" for family, _, _ in canonical_selections
    ) else "none"
    entries: list[list[str]] = []
    acquisition_records: list[AcquisitionRecord] = []
    for family, url, identity in canonical_selections:
        occurrence_identity = f"federal-reserve:{family}:{identity}"
        try:
            download = fetch(
                url,
                lambda value, selected_family=family: canonical_occurrence_url(
                    value, selected_family
                )[0],
            )
        except Exception as error:
            acquisition_records.append(AcquisitionRecord(
                url,
                error.final_url if isinstance(error, ResourceRedirectError) else "",
                occurrence_identity,
                "failed",
                diagnostic=str(error).replace("\t", " ").replace("\n", " | "),
            ))
            write_acquisition_manifest(output / "acquisition.tsv", acquisition_records)
            raise
        source_bytes = download.data
        acquisition_records.append(AcquisitionRecord(
            url,
            download.final_url,
            occurrence_identity,
            "succeeded",
            sha256(source_bytes),
        ))
        if family == "beige_book":
            if not source_bytes.startswith(b"%PDF-"):
                raise RuntimeError(f"Beige Book occurrence is not a PDF: {url}")
            source_relative = pathlib.Path("beige-book") / f"{identity}.pdf"
            source_path = output / source_relative
            source_path.parent.mkdir(parents=True, exist_ok=True)
            write_new(source_path, source_bytes)

            text_relative = source_relative.with_suffix(".txt")
            text_path = output / text_relative
            with tempfile.NamedTemporaryFile(dir=text_path.parent, delete=False) as temporary:
                temporary_path = pathlib.Path(temporary.name)
            try:
                subprocess.run(
                    [pdftotext, "-layout", "-enc", "UTF-8", str(source_path),
                     str(temporary_path)],
                    check=True,
                )
                text_bytes = temporary_path.read_bytes()
                if not text_bytes.strip():
                    raise RuntimeError(f"pdftotext produced empty output: {url}")
                write_new(text_path, text_bytes)
            finally:
                temporary_path.unlink(missing_ok=True)
            entries.append([
                text_relative.as_posix(), sha256(text_bytes), "beige_book_pdf_text", url,
                source_relative.as_posix(), sha256(source_bytes), extractor,
            ])
        else:
            if b"<html" not in source_bytes[:4096].lower():
                raise RuntimeError(f"Federal Reserve press release is not HTML: {url}")
            directory = "fomc-statement" if family == "fomc_statement" else "fomc-minutes"
            relative = pathlib.Path(directory) / f"{identity}.html"
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            write_new(path, source_bytes)
            digest = sha256(source_bytes)
            entries.append([
                relative.as_posix(), digest, family + "_html", url,
                relative.as_posix(), digest, "none",
            ])

    manifest_lines = [
        "manifest_version\t1",
        "parser_version\tfederal_reserve_economic_release_v2",
        "artifact_path\tartifact_sha256\tartifact_type\tsource_url\t"
        "source_artifact_path\tsource_artifact_sha256\textractor",
    ]
    manifest_lines.extend("\t".join(entry) for entry in sorted(entries))
    write_new(output / "manifest.tsv", ("\n".join(manifest_lines) + "\n").encode("utf-8"))
    write_acquisition_manifest(output / "acquisition.tsv", acquisition_records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--statement-url", action="append", default=[])
    parser.add_argument("--minutes-url", action="append", default=[])
    parser.add_argument("--beige-book-url", action="append", default=[])
    parser.add_argument("--archive-year", action="append", type=int, default=[])
    parser.add_argument("--family", action="append", choices=FAMILIES, default=[])
    parser.add_argument("--pdftotext", default=shutil.which("pdftotext"))
    args = parser.parse_args()

    if args.archive_year and not args.family:
        parser.error("--archive-year requires at least one --family")
    if args.family and not args.archive_year:
        parser.error("--family requires at least one --archive-year")

    selections = (
        [("fomc_statement", url) for url in args.statement_url]
        + [("fomc_minutes", url) for url in args.minutes_url]
        + [("beige_book", url) for url in args.beige_book_url]
    )
    for year in sorted(set(args.archive_year)):
        try:
            selections.extend(enumerate_archive(year, set(args.family)))
        except ValueError as error:
            parser.error(str(error))
    acquire(selections, args.output_dir.resolve(), args.pdftotext)
    print(
        f"FEDERAL_RESERVE_ACQUISITION_COMPLETE artifacts={len(set(selections))} "
        f"manifest={args.output_dir.resolve() / 'manifest.tsv'}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FEDERAL_RESERVE_ACQUISITION_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
