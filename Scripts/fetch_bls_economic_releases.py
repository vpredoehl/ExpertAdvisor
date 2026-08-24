#!/usr/bin/env python3
"""Acquire first-party BLS annual schedules and enumerate supported releases."""

from __future__ import annotations

import argparse
import datetime
import html.parser
import pathlib
import re
import sys
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


USER_AGENT = (
    "Mozilla/5.0 (compatible; ExpertAdvisor economic-calendar audit; "
    "+https://www.bls.gov/)"
)
ALLOWED_HOST = "www.bls.gov"
SCHEDULE_URL = "https://www.bls.gov/schedule/{year}/home.htm"
SUPPORTED_TITLES = {
    "Consumer Price Index": "CPI",
    "Employment Situation": "EMPLOYMENT",
    "Employment Situation of Veterans": "EMPLOYMENT_ANNUAL",
    "Job Openings and Labor Turnover": "JOLTS",
    "Job Openings and Labor Turnover Survey": "JOLTS",
    "Producer Price Index": "PPI",
}


class ScheduleParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, str]] = []
        self._row: dict[str, str] | None = None
        self._cell: str | None = None
        self._text: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag.lower() == "tr":
            self._row = {}
        elif tag.lower() == "td" and self._row is not None:
            css_class = attributes.get("class", "") or ""
            if css_class in {"date-cell", "time-cell", "desc-cell"}:
                self._cell = css_class
                self._text = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "td" and self._cell is not None and self._row is not None:
            self._row[self._cell] = " ".join(" ".join(self._text).split())
            self._cell = None
            self._text = []
        elif tag.lower() == "tr" and self._row is not None:
            if {"date-cell", "time-cell", "desc-cell"} <= self._row.keys():
                self.rows.append(self._row)
            self._row = None


def canonical_schedule_url(value: str) -> str:
    canonical = canonical_https_url(value, allowed_hosts={ALLOWED_HOST})
    parsed = urllib.parse.urlsplit(canonical)
    if parsed.query or parsed.fragment or not re.fullmatch(
        r"/schedule/[0-9]{4}/home\.htm", parsed.path
    ):
        raise ValueError(f"not a canonical BLS annual schedule URL: {value}")
    return canonical


def fetch(
    url: str,
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> Download:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        final_url = validate_final_resource(
            url, response.geturl(), canonicalize=canonical_schedule_url
        )
        data = read_bounded(
            response,
            requested_url=url,
            max_download_bytes=max_download_bytes,
        )
    return Download(url, final_url, data)


def parse_occurrences(
    data: bytes,
    schedule_url: str,
    year: int,
    as_of_date: datetime.date | None = None,
) -> list[dict[str, str]]:
    parser = ScheduleParser()
    parser.feed(data.decode("utf-8-sig", errors="strict"))
    occurrences: list[dict[str, str]] = []
    for row in parser.rows:
        description = row["desc-cell"]
        matched = next(
            (
                (title, family)
                for title, family in SUPPORTED_TITLES.items()
                if description.startswith(title + " for ")
            ),
            None,
        )
        if matched is None:
            continue
        title, family = matched
        release_date = datetime.datetime.strptime(
            row["date-cell"], "%A, %B %d, %Y"
        ).date()
        if release_date.year != year:
            raise RuntimeError(
                f"BLS supported occurrence falls outside schedule year: {description}"
            )
        if as_of_date is not None and release_date > as_of_date:
            continue
        release_time = datetime.datetime.strptime(
            row["time-cell"].upper(), "%I:%M %p"
        ).time().strftime("%H:%M:00")
        reference_period = description[len(title + " for "):]
        if not reference_period:
            raise RuntimeError(f"BLS reference period missing: {description}")
        occurrences.append({
            "family": family,
            "release_date": release_date.isoformat(),
            "release_time": release_time,
            "title": title,
            "reference_period": reference_period,
            "schedule_url": schedule_url,
        })
    identities = {
        (row["family"], row["reference_period"]) for row in occurrences
    }
    if len(identities) != len(occurrences):
        raise RuntimeError(f"BLS schedule contains duplicate supported identities: {year}")
    return sorted(
        occurrences,
        key=lambda row: (row["release_date"], row["release_time"], row["family"]),
    )


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not normalized:
        raise ValueError("BLS identity component is empty")
    return normalized


def acquire(
    years: list[int],
    output: pathlib.Path,
    as_of_date: datetime.date | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output}")
    records: list[AcquisitionRecord] = []
    entries: list[list[str]] = []
    for year in sorted(set(years)):
        requested = canonical_schedule_url(SCHEDULE_URL.format(year=year))
        resource_identity = f"bls-schedule-year:{year}"
        try:
            download = fetch(requested)
        except Exception as error:
            records.append(AcquisitionRecord(
                requested,
                error.final_url if isinstance(error, ResourceRedirectError) else "",
                resource_identity,
                "failed",
                diagnostic=str(error).replace("\t", " ").replace("\n", " | "),
            ))
            write_acquisition_manifest(output / "acquisition.tsv", records)
            raise
        source_relative = pathlib.Path("schedules") / f"{year}.html"
        source_path = output / source_relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(download.data)
        source_hash = digest(download.data)
        records.append(AcquisitionRecord(
            requested,
            download.final_url,
            resource_identity,
            "succeeded",
            source_hash,
        ))
        for occurrence in parse_occurrences(
            download.data, requested, year, as_of_date
        ):
            family_identity = occurrence["family"].lower().replace("_", "-")
            identity = f"bls:{family_identity}-{slug(occurrence['reference_period'])}"
            row_relative = pathlib.Path("occurrences") / f"{identity[4:]}.txt"
            row_data = (
                "bls_schedule_row_version\t1\n"
                + "\n".join(
                    f"{key}\t{occurrence[key]}"
                    for key in (
                        "family",
                        "release_date",
                        "release_time",
                        "title",
                        "reference_period",
                        "schedule_url",
                    )
                )
                + f"\nsource_event_id\t{identity}\n"
            ).encode("utf-8")
            row_path = output / row_relative
            row_path.parent.mkdir(parents=True, exist_ok=True)
            row_path.write_bytes(row_data)
            entries.append([
                row_relative.as_posix(),
                digest(row_data),
                "bls_schedule_row",
                requested,
                source_relative.as_posix(),
                source_hash,
                "bls_schedule_row_v1",
            ])
    manifest = [
        "manifest_version\t1",
        "parser_version\tbls_schedule_release_v1",
        "artifact_path\tartifact_sha256\tartifact_type\tsource_url\t"
        "source_artifact_path\tsource_artifact_sha256\textractor",
        *("\t".join(entry) for entry in sorted(entries)),
    ]
    (output / "manifest.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    write_acquisition_manifest(output / "acquisition.tsv", records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=datetime.date.today().year)
    parser.add_argument(
        "--as-of-date",
        type=datetime.date.fromisoformat,
        default=datetime.date.today(),
    )
    args = parser.parse_args()
    if args.start_year < 2000 or args.end_year > datetime.date.today().year:
        parser.error("BLS years are outside the supported archive range")
    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")
    acquire(
        list(range(args.start_year, args.end_year + 1)),
        args.output_dir.resolve(),
        args.as_of_date,
    )
    print(
        "BLS_ACQUISITION_COMPLETE"
        f" years={args.end_year - args.start_year + 1}"
        f" manifest={args.output_dir.resolve() / 'manifest.tsv'}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"BLS_ACQUISITION_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
