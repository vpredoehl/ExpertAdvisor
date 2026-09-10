#!/usr/bin/env python3
"""Aggregate predeclared Phase 19 artifacts without selecting thresholds."""

from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict


PRIMARY_MODELS = {1745, 1743, 1729, 1735, 1662, 1692}
DIAGNOSTIC_MODEL = 1805
WINDOWS = (
    "discovery_history_2025",
    "temporal_validation_history_2026",
)


def parse_record(line: str) -> tuple[str, dict[str, str]]:
    parts = line.rstrip("\n").split(",")
    values: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            values[key] = value
    return parts[0], values


def sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def read_artifacts(directory: pathlib.Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.csv")):
        header: dict[str, str] | None = None
        strata: dict[str, dict[str, str]] = {}
        deltas: dict[str, dict[str, str]] = {}
        populations: dict[str, dict[str, str]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            record, values = parse_record(line)
            if record == "PHASE19_ANALYSIS":
                header = values
            elif record == "PHASE19_STATE_STRATUM":
                strata[values["scope"]] = values
            elif record == "PHASE19_STATE_STRATUM_DELTA":
                deltas[values["scope"]] = values
            elif record == "PHASE19_POPULATION_DELTA":
                populations[values["scope"]] = values
        if header is None:
            continue
        model_id = int(header["model_id"])
        if model_id not in PRIMARY_MODELS | {DIAGNOSTIC_MODEL}:
            raise ValueError(f"unexpected Phase 19 model {model_id}: {path}")
        window = header["window_label"]
        if window not in WINDOWS:
            raise ValueError(f"unexpected Phase 19 window {window}: {path}")
        missing = set(strata) ^ set(deltas)
        if missing:
            raise ValueError(f"stratum/delta mismatch {sorted(missing)}: {path}")
        results.append(
            {
                "path": path,
                "header": header,
                "model_id": model_id,
                "window": window,
                "role": header["cohort_role"],
                "accepted": header["model_acceptance_qualified"] == "true",
                "strata": strata,
                "deltas": deltas,
                "populations": populations,
            }
        )
    expected = {(model, window) for model in PRIMARY_MODELS | {DIAGNOSTIC_MODEL}
                for window in WINDOWS}
    actual = {(int(item["model_id"]), str(item["window"])) for item in results}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"artifact cohort mismatch; missing={missing}; extra={extra}")
    return results


def aggregate(results: list[dict[str, object]]) -> list[dict[str, object]]:
    by_window_scope: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for result in results:
        if result["role"] != "primary":
            continue
        strata = result["strata"]
        deltas = result["deltas"]
        assert isinstance(strata, dict) and isinstance(deltas, dict)
        for scope, state in strata.items():
            delta = deltas[scope]
            by_window_scope[(str(result["window"]), scope)].append(
                {
                    "model_id": result["model_id"],
                    "accepted": result["accepted"],
                    "count": int(state["activated_count"]),
                    "delta": float(delta["aggregate_return_delta"]),
                }
            )

    rows: list[dict[str, object]] = []
    for (window, scope), items in sorted(by_window_scope.items()):
        supported = [item for item in items if int(item["count"]) > 0]
        nontrivial = [item for item in supported if int(item["count"]) >= 30]
        accepted = [item for item in supported if bool(item["accepted"])]
        signs = [sign(float(item["delta"])) for item in supported]
        nontrivial_signs = [sign(float(item["delta"])) for item in nontrivial]
        rows.append(
            {
                "window": window,
                "scope": scope,
                "models_with_support": len(supported),
                "positive_models": signs.count(1),
                "negative_models": signs.count(-1),
                "zero_models": signs.count(0),
                "combined_aggregate_delta": sum(float(x["delta"]) for x in supported),
                "total_activated_count": sum(int(x["count"]) for x in supported),
                "accepted_models_with_support": len(accepted),
                "accepted_positive_models": sum(sign(float(x["delta"])) == 1 for x in accepted),
                "accepted_negative_models": sum(sign(float(x["delta"])) == -1 for x in accepted),
                "accepted_zero_models": sum(sign(float(x["delta"])) == 0 for x in accepted),
                "accepted_combined_delta": sum(float(x["delta"]) for x in accepted),
                "accepted_activated_count": sum(int(x["count"]) for x in accepted),
                "nontrivial_support_models": len(nontrivial),
                "sign_agrees_across_nontrivial_models": (
                    len(nontrivial_signs) >= 2
                    and len(set(nontrivial_signs)) == 1
                    and nontrivial_signs[0] != 0
                ),
            }
        )

    lookup = {(str(row["window"]), str(row["scope"])): row for row in rows}
    for row in rows:
        other_window = WINDOWS[1] if row["window"] == WINDOWS[0] else WINDOWS[0]
        other = lookup.get((other_window, str(row["scope"])))
        current_sign = sign(float(row["combined_aggregate_delta"]))
        other_sign = sign(float(other["combined_aggregate_delta"])) if other else 0
        row["aggregate_sign_agrees_between_windows"] = (
            other is not None and current_sign != 0 and current_sign == other_sign
        )
    return rows


def write_outputs(rows: list[dict[str, object]], prefix: pathlib.Path) -> None:
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Phase 19 cross-model interaction summary\n\n")
        handle.write(
            "Exploratory historical mechanism evidence only; this is not "
            "independent validation and does not promote a trading rule.\n\n"
        )
        for window in WINDOWS:
            handle.write(f"## {window}\n\n")
            for row in (item for item in rows if item["window"] == window):
                handle.write(
                    f"- `{row['scope']}`: n={row['total_activated_count']}, "
                    f"models +/−/0={row['positive_models']}/"
                    f"{row['negative_models']}/{row['zero_models']}, "
                    f"combined delta={float(row['combined_aggregate_delta']):.12g}, "
                    f"acceptance-qualified delta="
                    f"{float(row['accepted_combined_delta']):.12g}.\n"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-prefix", type=pathlib.Path, required=True)
    args = parser.parse_args()
    write_outputs(aggregate(read_artifacts(args.artifact_dir)), args.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
