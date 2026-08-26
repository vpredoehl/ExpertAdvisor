#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("EconomicCalendar/raw/oanda")
DAILY = ROOT / "daily"

OUTPUT = ROOT / "oanda_report_id_discovery.csv"
EVENT_NAMES = ROOT / "oanda_event_name_inventory.csv"


#
# We are deliberately broad here.
#
# This is a discovery probe, not the final canonical mapper.
# False positives are acceptable in this output; silent exclusion is not.
#
FAMILY_PATTERNS = {
    "DURABLE_GOODS": [
        r"\bdurable\b",
        r"\bdurable goods\b",
        r"\bdurable goods orders\b",
        r"\bcore durable\b",
        r"\bnondefense capital goods\b",
        r"\bcapital goods orders\b",
    ],

    "EMPLOYMENT": [
        r"\bnonfarm\b",
        r"\bnon-farm\b",
        r"\bpayroll",
        r"\bemployment change\b",
        r"\bunemployment rate\b",
        r"\baverage hourly earnings\b",
        r"\blabor force participation\b",
        r"\bparticipation rate\b",
    ],

    "FOMC": [
        r"\bfomc\b",
        r"\bfederal funds\b",
        r"\bfed funds\b",
        r"\binterest rate decision\b",
        r"\brate decision\b",
    ],

    "GDP": [
        r"\bgdp\b",
        r"\bgross domestic product\b",
    ],

    "JOLTS": [
        r"\bjolts\b",
        r"\bjob openings\b",
    ],

    "PCE": [
        r"\bpce\b",
        r"\bpersonal consumption expenditures\b",
        r"\bpersonal spending\b",
        r"\bpersonal income\b",
        r"\bcore pce\b",
    ],
}


def normalize(value):
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def load_json(path: Path):
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except Exception as exc:
        print(
            f"WARNING: could not read "
            f"{path}: {exc}"
        )
        return None


def iter_cached_rows():
    files = sorted(
        DAILY.glob("*.json")
    )

    if not files:
        raise SystemExit(
            f"No cached OANDA daily JSON "
            f"files found under {DAILY}"
        )

    for path in files:
        data = load_json(
            path
        )

        if data is None:
            continue

        if isinstance(data, dict):
            rows = (
                data.get("data")
                or data.get("rows")
                or []
            )
        elif isinstance(data, list):
            rows = data
        else:
            continue

        if not isinstance(
            rows,
            list,
        ):
            continue

        for row in rows:
            if not isinstance(
                row,
                dict,
            ):
                continue

            country = normalize(
                row.get("Country")
            ).upper()

            if country not in {
                "US",
                "USA",
            }:
                continue

            yield path, row


def classify_event(
    event_name: str,
):
    value = event_name.lower()

    families = []

    for family, patterns in (
        FAMILY_PATTERNS.items()
    ):
        if any(
            re.search(
                pattern,
                value,
                re.I,
            )
            for pattern in patterns
        ):
            families.append(
                family
            )

    return families


def report_id_from_row(row):
    value = (
        row.get("IDReport")
        or row.get("ReportID")
        or row.get("reportID")
    )

    if value in (
        None,
        "",
    ):
        return ""

    return str(value)


def event_id_from_row(row):
    value = (
        row.get("ID")
        or row.get("Id")
        or row.get("id")
    )

    if value in (
        None,
        "",
    ):
        return ""

    return str(value)


def main():
    print(
        "OANDA report-ID discovery probe"
    )
    print()
    print(
        f"Cached daily source : {DAILY}"
    )
    print(
        "Network access       : none"
    )
    print(
        "Database access      : none"
    )
    print()

    candidate_stats = defaultdict(
        lambda: {
            "rows": 0,
            "dates": set(),
            "forecast_rows": 0,
            "previous_rows": 0,
            "actual_rows": 0,
            "event_ids": set(),
            "sample_dates": [],
        }
    )

    name_stats = Counter()
    total_us_rows = 0

    for path, row in (
        iter_cached_rows()
    ):
        total_us_rows += 1

        event_name = normalize(
            row.get("Event")
            or row.get("Name")
        )

        if not event_name:
            continue

        name_stats[
            event_name
        ] += 1

        families = classify_event(
            event_name
        )

        if not families:
            continue

        report_id = (
            report_id_from_row(
                row
            )
        )

        event_id = (
            event_id_from_row(
                row
            )
        )

        date_value = normalize(
            row.get("Date")
        )

        forecast = normalize(
            row.get("Forecast")
        )
        previous = normalize(
            row.get("Previous")
        )
        actual = normalize(
            row.get("Actual")
        )

        for family in families:
            key = (
                family,
                report_id,
                event_name,
            )

            stats = (
                candidate_stats[key]
            )

            stats[
                "rows"
            ] += 1

            if date_value:
                stats[
                    "dates"
                ].add(
                    date_value[:10]
                )

                if (
                    len(
                        stats[
                            "sample_dates"
                        ]
                    )
                    < 5
                ):
                    stats[
                        "sample_dates"
                    ].append(
                        date_value
                    )

            if forecast:
                stats[
                    "forecast_rows"
                ] += 1

            if previous:
                stats[
                    "previous_rows"
                ] += 1

            if actual:
                stats[
                    "actual_rows"
                ] += 1

            if event_id:
                stats[
                    "event_ids"
                ].add(
                    event_id
                )

    output_rows = []

    for (
        family,
        report_id,
        event_name,
    ), stats in sorted(
        candidate_stats.items(),
        key=lambda item: (
            item[0][0],
            -item[1]["rows"],
            item[0][2],
            item[0][1],
        ),
    ):
        dates = sorted(
            stats["dates"]
        )

        output_rows.append({
            "event_family":
                family,
            "oanda_report_id":
                report_id,
            "oanda_event":
                event_name,
            "observed_rows":
                stats["rows"],
            "distinct_dates":
                len(dates),
            "first_date":
                dates[0]
                if dates
                else "",
            "last_date":
                dates[-1]
                if dates
                else "",
            "forecast_rows":
                stats[
                    "forecast_rows"
                ],
            "previous_rows":
                stats[
                    "previous_rows"
                ],
            "actual_rows":
                stats[
                    "actual_rows"
                ],
            "distinct_event_ids":
                len(
                    stats[
                        "event_ids"
                    ]
                ),
            "sample_dates":
                " | ".join(
                    stats[
                        "sample_dates"
                    ]
                ),
        })

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fields = [
            "event_family",
            "oanda_report_id",
            "oanda_event",
            "observed_rows",
            "distinct_dates",
            "first_date",
            "last_date",
            "forecast_rows",
            "previous_rows",
            "actual_rows",
            "distinct_event_ids",
            "sample_dates",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(
            output_rows
        )

    with EVENT_NAMES.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fields = [
            "oanda_event",
            "observed_rows",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for name, count in sorted(
            name_stats.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):
            writer.writerow({
                "oanda_event":
                    name,
                "observed_rows":
                    count,
            })

    print(
        f"US cached rows       : "
        f"{total_us_rows}"
    )
    print(
        f"Candidate tuples     : "
        f"{len(output_rows)}"
    )
    print()

    for family in (
        "DURABLE_GOODS",
        "EMPLOYMENT",
        "FOMC",
        "GDP",
        "JOLTS",
        "PCE",
    ):
        rows = [
            row
            for row in output_rows
            if (
                row[
                    "event_family"
                ]
                == family
            )
        ]

        print(
            "=" * 100
        )
        print(
            family
        )
        print(
            "=" * 100
        )

        if not rows:
            print(
                "No candidates found."
            )
            print()
            continue

        for row in rows:
            print(
                f"report={row['oanda_report_id']:<6} "
                f"rows={row['observed_rows']:>4} "
                f"forecast={row['forecast_rows']:>4} "
                f"actual={row['actual_rows']:>4} "
                f"{row['first_date']}.."
                f"{row['last_date']} "
                f"| {row['oanda_event']}"
            )

        print()

    print(
        "Discovery output :",
        OUTPUT,
    )
    print(
        "Event inventory  :",
        EVENT_NAMES,
    )
    print()
    print(
        "No HTTP requests were performed."
    )
    print(
        "No database access was performed."
    )
    print(
        "No files other than the two reports "
        "above were modified."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
