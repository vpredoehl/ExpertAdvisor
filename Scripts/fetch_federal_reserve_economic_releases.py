#!/usr/bin/env python3
"""Fetch a bounded set of first-party Federal Reserve publication artifacts.

The caller explicitly selects statement, minutes, and Beige Book occurrence
URLs.  This acquisition-only program never opens PostgreSQL and writes the
shared version-1 manifest consumed by the common economic-event import path.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request


USER_AGENT = "ExpertAdvisor-authoritative-calendar-acquisition/1"
ALLOWED_HOST = "www.federalreserve.gov"
MAX_ARTIFACTS = 100
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
PRESS_RELEASE = re.compile(
    r"^/newsevents/pressreleases/monetary(?P<date>[0-9]{8})(?P<suffix>[a-z])\.htm$"
)
BEIGE_BOOK_PATTERNS = (
    re.compile(
        r"^/fomc/beigebook/(?P<year>[0-9]{4})/(?P<date>[0-9]{8})/"
        r"fullreport(?P=date)\.pdf$"
    ),
    re.compile(
        r"^/monetarypolicy/beigebook/files/BeigeBook_(?P<date>[0-9]{8})\.pdf$"
    ),
    re.compile(r"^/monetarypolicy/files/BeigeBook_(?P<date>[0-9]{8})\.pdf$"),
)


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


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        final = urllib.parse.urlsplit(response.geturl())
        final_url = urllib.parse.urlunsplit(
            (final.scheme.lower(), final.hostname or "", final.path, final.query, final.fragment)
        )
        if final_url != url:
            raise RuntimeError(f"unsupported Federal Reserve redirect: {response.geturl()}")
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
    if not data:
        raise RuntimeError(f"empty download: {url}")
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise RuntimeError(f"download exceeds safety limit: {url}")
    return data


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
    for family, url, identity in canonical_selections:
        source_bytes = fetch(url)
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
        "parser_version\tfederal_reserve_economic_release_v1",
        "artifact_path\tartifact_sha256\tartifact_type\tsource_url\t"
        "source_artifact_path\tsource_artifact_sha256\textractor",
    ]
    manifest_lines.extend("\t".join(entry) for entry in sorted(entries))
    write_new(output / "manifest.tsv", ("\n".join(manifest_lines) + "\n").encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--statement-url", action="append", default=[])
    parser.add_argument("--minutes-url", action="append", default=[])
    parser.add_argument("--beige-book-url", action="append", default=[])
    parser.add_argument("--pdftotext", default=shutil.which("pdftotext"))
    args = parser.parse_args()

    selections = (
        [("fomc_statement", url) for url in args.statement_url]
        + [("fomc_minutes", url) for url in args.minutes_url]
        + [("beige_book", url) for url in args.beige_book_url]
    )
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
