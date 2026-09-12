#!/usr/bin/env python3
"""Frozen Phase 19C causal-predictability analysis (no strategy search)."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA = "phase19c_causal_path_predictability_analysis_v1"
PLAN = "phase19c_frozen_analysis_plan_v1"
PRIMARY_MODELS = (1745, 1743, 1729, 1735, 1662, 1692)
DIAGNOSTIC_MODEL = 1805
FROZEN_COHORT = {
    1745: ("AUDCHF", 4, "primary"),
    1743: ("EURCHF", 12, "primary"),
    1729: ("USDCAD", 6, "primary"),
    1735: ("GBPUSD", 6, "primary"),
    1662: ("GBPJPY", 8, "primary"),
    1692: ("AUDCAD", 14, "primary"),
    1805: ("EURUSD", 4, "diagnostic"),
}
WINDOWS = (("2025-01-01", "2026-01-01"),
           ("2026-01-01", "2026-09-01"))
EXPECTED = {(model, start, end) for model in PRIMARY_MODELS + (DIAGNOSTIC_MODEL,)
            for start, end in WINDOWS}
FORBIDDEN_PREDICTOR_TOKENS = (
    "maximum_adverse_excursion", "maximum_favorable_excursion", "stop_hit",
    "stop_time", "recovery", "adverse_excursion_beyond_fixed_stop",
    "required_extra_room", "terminal", "future", "path_point")
L2_STRENGTH = 1.0
LOGISTIC_MAX_ITERATIONS = 100
LOGISTIC_CONVERGENCE = 1.0e-10
DATASET_SCHEMA = "phase19c_causal_path_predictability_dataset_v1"
DECISION_PREDICTORS = {
    "predictor__predicted_class": "categorical",
    "predictor__direction": "categorical",
    "predictor__directional_probability": "numeric",
    "predictor__normalized_directional_confidence": "numeric",
}
ALLOWED_MODEL_INPUT_PROVENANCE = {
    ("authoritative_model_input_tensor_decision_row",
     "completed_decision_row_at_or_before_entry"),
    ("authoritative_model_input_return_suffix",
     "computed_from_completed_closes_ending_at_decision_row"),
}


def fnv1a64(content: bytes) -> str:
    value = 14695981039346656037
    for byte in content:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{value:016x}"


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "undefined"
    return format(value, ".17g")


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]], bytes]:
    content = path.read_bytes()
    text = content.decode("utf-8")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"phase19c_empty_tsv:{path}")
    parsed = list(csv.reader(lines, delimiter="\t"))
    width = len(parsed[0])
    if width == 0 or any(len(row) != width for row in parsed):
        raise ValueError(f"phase19c_tsv_field_count_mismatch:{path}")
    if len(set(parsed[0])) != width:
        raise ValueError(f"phase19c_duplicate_column:{path}")
    return parsed[0], [dict(zip(parsed[0], row)) for row in parsed[1:]], content


def read_metadata(path: Path) -> dict[str, str]:
    _, rows, _ = read_tsv(path)
    result: dict[str, str] = {}
    for row in rows:
        if row["schema_identity"] != DATASET_SCHEMA or row["schema_version"] != "1":
            raise ValueError("phase19c_dataset_metadata_schema_mismatch")
        if row["key"] in result:
            raise ValueError("phase19c_duplicate_dataset_metadata_key")
        result[row["key"]] = row["value"]
    return result


@dataclass(frozen=True)
class RunData:
    model_id: int
    start: str
    end: str
    role: str
    rows: tuple[dict[str, str], ...]
    predictors: tuple[str, ...]
    categorical: frozenset[str]
    metadata: dict[str, str]
    path: Path


def load_runs(input_dir: Path, allow_incomplete: bool = False) -> list[RunData]:
    paths = sorted(input_dir.glob("*.phase19c.dataset.tsv"))
    if not paths:
        raise ValueError("phase19c_no_causal_datasets_found")
    runs: list[RunData] = []
    seen_scope: set[tuple[int, str, str]] = set()
    seen_observations: set[tuple[int, str, str, str]] = set()
    for path in paths:
        header, rows, content = read_tsv(path)
        if not header or header[0:2] != ["schema_identity", "schema_version"]:
            raise ValueError(f"phase19c_dataset_header_invalid:{path}")
        if any(not name.startswith(("predictor__", "outcome__")) and
               name not in {"schema_identity", "schema_version", "model_id",
                            "experiment_id", "symbol", "cohort_role",
                            "prediction_horizon", "window_start", "window_end",
                            "observation_id", "observation_ordinal",
                            "entry_source_row", "entry_timestamp_unix_seconds",
                            "feature_layout_identity", "feature_layout_hash",
                            "source_phase19b_observations_hash",
                            "source_phase19b_result_hash"}
               for name in header):
            raise ValueError("phase19c_unclassified_dataset_column")
        predictor_columns = tuple(name for name in header
                                  if name.startswith("predictor__"))
        if not predictor_columns:
            raise ValueError("phase19c_predictor_manifest_mismatch")
        for name in predictor_columns:
            lowered = name.lower()
            if any(token in lowered for token in FORBIDDEN_PREDICTOR_TOKENS):
                raise ValueError(f"phase19c_forbidden_predictor_leakage:{name}")
        required_outcomes = {"outcome__outcome_class",
                             "outcome__extension_minus_fixed_return"}
        if not required_outcomes.issubset(header):
            raise ValueError("phase19c_required_outcome_missing")
        meta_path = path.with_name(path.name.replace(
            ".phase19c.dataset.tsv", ".phase19c.metadata.tsv"))
        predictor_path = path.with_name(path.name.replace(
            ".phase19c.dataset.tsv", ".phase19c.predictors.tsv"))
        if not meta_path.is_file() or not predictor_path.is_file():
            raise ValueError(f"phase19c_companion_artifact_missing:{path}")
        metadata = read_metadata(meta_path)
        required_metadata = {
            "analysis_plan_identity", "model_id", "experiment_id", "symbol",
            "cohort_role", "prediction_horizon", "window_start", "window_end",
            "activated_count", "saved_count", "harmed_count", "unchanged_count",
            "joined_count", "join_failure_count", "usable_predictor_count",
            "model_input_width", "semantic_layout_version",
            "feature_layout_identity", "feature_layout_hash",
            "source_phase19b_observations_hash", "source_phase19b_result_hash",
            "dataset_hash", "result_hash", "target_comparison_semantics",
            "primary_target_population", "unchanged_retained_not_classified",
            "predictor_state_semantics", "post_entry_predictors_included",
            "leakage_audit_passed", "requires_read_only_transaction",
            "production_rows_modified",
        }
        if not required_metadata.issubset(metadata):
            raise ValueError("phase19c_dataset_metadata_incomplete")
        if metadata.get("dataset_hash") != fnv1a64(content):
            raise ValueError(f"phase19c_dataset_hash_mismatch:{path}")
        result_canonical = (
            "phase19c_causal_dataset_result_v1;dataset_hash=" +
            metadata["dataset_hash"] + ";feature_layout_hash=" +
            metadata["feature_layout_hash"] + ";source_phase19b_result_hash=" +
            metadata["source_phase19b_result_hash"] + ";")
        if metadata.get("result_hash") != fnv1a64(result_canonical.encode()):
            raise ValueError(f"phase19c_result_hash_mismatch:{path}")
        if metadata.get("leakage_audit_passed") != "true" or \
                metadata.get("post_entry_predictors_included") != "false" or \
                metadata.get("requires_read_only_transaction") != "true" or \
                metadata.get("production_rows_modified") != "false" or \
                metadata.get("analysis_plan_identity") != PLAN or \
                metadata.get("target_comparison_semantics") != \
                "exact_binary64_delta_sign_v1" or \
                metadata.get("primary_target_population") != \
                "SAVED_vs_HARMED_only" or \
                metadata.get("unchanged_retained_not_classified") != "true":
            raise ValueError("phase19c_leakage_audit_not_passed")
        _, manifest_rows, _ = read_tsv(predictor_path)
        manifest: dict[str, dict[str, str]] = {}
        manifest_order = []
        next_model_column = 0
        for item in manifest_rows:
            if item["schema_identity"] != DATASET_SCHEMA or \
                    item["schema_version"] != "1" or \
                    item["included_in_full_model"] != "true" or \
                    item["value_type"] not in {"numeric", "categorical"}:
                raise ValueError("phase19c_predictor_manifest_invalid")
            name = "predictor__" + item["predictor_name"]
            if name in manifest:
                raise ValueError("phase19c_predictor_manifest_duplicate")
            if name in DECISION_PREDICTORS:
                if item["source_category"] != "frozen_decision_state" or \
                        item["causal_timing"] != "at_entry_decision" or \
                        item["model_input_column"] or \
                        item["value_type"] != DECISION_PREDICTORS[name]:
                    raise ValueError("phase19c_predictor_manifest_invalid")
            else:
                provenance = (item["source_category"], item["causal_timing"])
                if provenance not in ALLOWED_MODEL_INPUT_PROVENANCE or \
                        item["model_input_column"] != str(next_model_column):
                    raise ValueError("phase19c_predictor_manifest_invalid")
                next_model_column += 1
            manifest[name] = item
            manifest_order.append(name)
        if tuple(manifest_order) != predictor_columns:
            raise ValueError("phase19c_predictor_manifest_mismatch")
        categorical = frozenset(name for name, item in manifest.items()
                                if item["value_type"] == "categorical")
        model_id = int(metadata["model_id"])
        start, end, role = (metadata["window_start"], metadata["window_end"],
                            metadata["cohort_role"])
        scope = (model_id, start, end)
        if scope in seen_scope:
            raise ValueError("phase19c_duplicate_model_window")
        seen_scope.add(scope)
        if model_id not in FROZEN_COHORT:
            raise ValueError("phase19c_frozen_cohort_role_mismatch")
        expected_symbol, expected_horizon, expected_role = FROZEN_COHORT[model_id]
        if role != expected_role or metadata["symbol"] != expected_symbol or \
                int(metadata["prediction_horizon"]) != expected_horizon:
            raise ValueError("phase19c_frozen_cohort_role_mismatch")
        if (start, end) not in WINDOWS:
            raise ValueError("phase19c_nonfrozen_window")
        expected_layout_hash = metadata["feature_layout_hash"]
        expected_source_hash = metadata["source_phase19b_observations_hash"]
        expected_source_result_hash = metadata["source_phase19b_result_hash"]
        for row in rows:
            if row["schema_identity"] != DATASET_SCHEMA or \
                    row["schema_version"] != "1" or \
                    (int(row["model_id"]), row["symbol"],
                     int(row["prediction_horizon"]), row["window_start"],
                     row["window_end"], row["cohort_role"]) != \
                    (model_id, expected_symbol, expected_horizon, start, end, role) or \
                    row["feature_layout_identity"] != metadata["feature_layout_identity"] or \
                    row["feature_layout_hash"] != expected_layout_hash or \
                    row["source_phase19b_observations_hash"] != expected_source_hash or \
                    row["source_phase19b_result_hash"] != expected_source_result_hash:
                raise ValueError("phase19c_inconsistent_dataset_scope")
            identity = (model_id, start, end, row["observation_id"])
            if identity in seen_observations:
                raise ValueError("phase19c_duplicate_observation_identity")
            seen_observations.add(identity)
            if row["outcome__outcome_class"] not in {"SAVED", "HARMED", "UNCHANGED"}:
                raise ValueError("phase19c_invalid_target_class")
            for name in predictor_columns:
                if name == "predictor__direction":
                    continue
                try:
                    value = float(row[name])
                except ValueError as error:
                    raise ValueError(
                        f"phase19c_invalid_predictor_value:{name}") from error
                if not math.isfinite(value):
                    raise ValueError(f"phase19c_invalid_predictor_value:{name}")
            predicted_class = row["predictor__predicted_class"]
            direction = row["predictor__direction"]
            if (predicted_class, direction) not in {("0", "short"), ("2", "long")}:
                raise ValueError("phase19c_invalid_entry_decision_state")
            for name in ("predictor__directional_probability",
                         "predictor__normalized_directional_confidence"):
                if not 0.0 <= float(row[name]) <= 1.0:
                    raise ValueError("phase19c_invalid_entry_decision_state")
        outcome_counts = {name: sum(row["outcome__outcome_class"] == name
                                    for row in rows)
                          for name in ("SAVED", "HARMED", "UNCHANGED")}
        if int(metadata["activated_count"]) != len(rows) or \
                int(metadata["joined_count"]) != len(rows) or \
                int(metadata["join_failure_count"]) != 0 or \
                int(metadata["saved_count"]) != outcome_counts["SAVED"] or \
                int(metadata["harmed_count"]) != outcome_counts["HARMED"] or \
                int(metadata["unchanged_count"]) != outcome_counts["UNCHANGED"] or \
                int(metadata["usable_predictor_count"]) != len(predictor_columns) or \
                int(metadata["model_input_width"]) != next_model_column:
            raise ValueError("phase19c_dataset_join_accounting_mismatch")
        runs.append(RunData(model_id, start, end, role, tuple(rows),
                            predictor_columns, categorical, metadata, path))
    missing = EXPECTED - seen_scope
    extra = seen_scope - EXPECTED
    if extra or (missing and not allow_incomplete):
        raise ValueError("phase19c_frozen_cohort_incomplete:missing=" +
                         ",".join(f"{m}:{s}" for m, s, _ in sorted(missing)))
    return sorted(runs, key=lambda run: (run.start, run.model_id))


def binary_rows(runs: Iterable[RunData]) -> list[dict[str, str]]:
    return [row for run in runs for row in run.rows
            if row["outcome__outcome_class"] in {"SAVED", "HARMED"}]


def labels(rows: Sequence[dict[str, str]]) -> list[int]:
    return [1 if row["outcome__outcome_class"] == "SAVED" else 0 for row in rows]


def auc(values: Sequence[float], y: Sequence[int]) -> float | None:
    positives = [value for value, label in zip(values, y) if label == 1]
    negatives = [value for value, label in zip(values, y) if label == 0]
    if not positives or not negatives:
        return None
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            favorable += 1.0 if positive > negative else (0.5 if positive == negative else 0.0)
    return favorable / (len(positives) * len(negatives))


def average_precision(values: Sequence[float], y: Sequence[int]) -> float | None:
    positive_count = sum(y)
    if positive_count == 0 or positive_count == len(y):
        return None
    groups: dict[float, list[int]] = {}
    for value, label in zip(values, y):
        groups.setdefault(value, []).append(label)
    result = 0.0
    previous_recall = 0.0
    # Deliberately group ties before updating precision.
    true_positive = false_positive = 0
    for value in sorted(groups, reverse=True):
        group = groups[value]
        true_positive += sum(group)
        false_positive += len(group) - sum(group)
        recall = true_positive / positive_count
        precision = true_positive / (true_positive + false_positive)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return result


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def smd(saved: Sequence[float], harmed: Sequence[float]) -> float | None:
    if not saved or not harmed:
        return None
    mean_saved, mean_harmed = statistics.fmean(saved), statistics.fmean(harmed)
    degrees = len(saved) + len(harmed) - 2
    if degrees <= 0:
        return None
    var_saved = statistics.variance(saved) if len(saved) > 1 else 0.0
    var_harmed = statistics.variance(harmed) if len(harmed) > 1 else 0.0
    pooled = math.sqrt(((len(saved) - 1) * var_saved +
                        (len(harmed) - 1) * var_harmed) / degrees)
    if pooled == 0.0:
        return 0.0 if mean_saved == mean_harmed else None
    return (mean_saved - mean_harmed) / pooled


def numeric_metric(rows: Sequence[dict[str, str]], predictor: str) -> dict[str, object]:
    y = labels(rows)
    values = [float(row[predictor]) for row in rows]
    saved = [value for value, label in zip(values, y) if label]
    harmed = [value for value, label in zip(values, y) if not label]
    score_auc = auc(values, y)
    return {"saved_n": len(saved), "harmed_n": len(harmed),
            "saved_mean": statistics.fmean(saved) if saved else None,
            "saved_median": median(saved),
            "harmed_mean": statistics.fmean(harmed) if harmed else None,
            "harmed_median": median(harmed), "effect": smd(saved, harmed),
            "auc": score_auc, "ap": average_precision(values, y),
            "orientation": ("higher_predicts_SAVED" if score_auc is not None
                            else "undefined")}


def categorical_metrics(rows: Sequence[dict[str, str]], predictor: str) -> list[dict[str, object]]:
    categories = sorted({row[predictor] for row in rows})
    result = []
    for category in categories:
        y = labels(rows)
        values = [1.0 if row[predictor] == category else 0.0 for row in rows]
        saved = [value for value, label in zip(values, y) if label]
        harmed = [value for value, label in zip(values, y) if not label]
        score_auc = auc(values, y)
        result.append({"predictor": predictor + "=" + category,
                       "saved_n": len(saved), "harmed_n": len(harmed),
                       "saved_mean": statistics.fmean(saved) if saved else None,
                       "saved_median": median(saved),
                       "harmed_mean": statistics.fmean(harmed) if harmed else None,
                       "harmed_median": median(harmed),
                       "effect": ((statistics.fmean(saved) - statistics.fmean(harmed))
                                  if saved and harmed else None),
                       "auc": score_auc, "ap": average_precision(values, y),
                       "orientation": "category_presence_predicts_SAVED"
                       if score_auc is not None else "undefined"})
    return result


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-14:
            raise ValueError("phase19c_logistic_singular_system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        for item in range(column, n + 1):
            augmented[column][item] /= scale
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            for item in range(column, n + 1):
                augmented[row][item] -= factor * augmented[column][item]
    return [augmented[row][n] for row in range(n)]


@dataclass(frozen=True)
class Standardizer:
    means: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def fit(cls, matrix: Sequence[Sequence[float]]) -> "Standardizer":
        columns = list(zip(*matrix))
        means = tuple(statistics.fmean(column) for column in columns)
        scales = tuple(math.sqrt(sum((value - mean) ** 2 for value in column) /
                                 len(column)) or 1.0
                       for column, mean in zip(columns, means))
        return cls(means, scales)

    def transform(self, matrix: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[(value - mean) / scale for value, mean, scale in
                 zip(row, self.means, self.scales)] for row in matrix]


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    standardizer: Standardizer
    coefficients: tuple[float, ...]  # intercept first
    iterations: int

    def probabilities(self, matrix: Sequence[Sequence[float]]) -> list[float]:
        transformed = self.standardizer.transform(matrix)
        result = []
        for row in transformed:
            score = self.coefficients[0] + sum(
                coefficient * value for coefficient, value in
                zip(self.coefficients[1:], row))
            result.append(1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, score)))))
        return result


def fit_logistic(matrix: Sequence[Sequence[float]], y: Sequence[int],
                 feature_names: Sequence[str],
                 l2_strength: float = L2_STRENGTH) -> LogisticModel:
    if not matrix or len(matrix) != len(y) or len(set(y)) != 2:
        raise ValueError("phase19c_logistic_training_classes_not_estimable")
    standardizer = Standardizer.fit(matrix)
    x = [[1.0] + row for row in standardizer.transform(matrix)]
    coefficients = [0.0] * len(x[0])
    iterations = 0
    for iteration in range(LOGISTIC_MAX_ITERATIONS):
        gradient = [0.0] * len(coefficients)
        hessian = [[0.0] * len(coefficients) for _ in coefficients]
        for row, label in zip(x, y):
            score = sum(value * coefficient for value, coefficient in
                        zip(row, coefficients))
            probability = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, score))))
            weight = max(probability * (1.0 - probability), 1.0e-12)
            for j in range(len(coefficients)):
                gradient[j] += row[j] * (label - probability)
                for k in range(len(coefficients)):
                    hessian[j][k] += row[j] * row[k] * weight
        for j in range(1, len(coefficients)):
            gradient[j] -= l2_strength * coefficients[j]
            hessian[j][j] += l2_strength
        step = solve_linear(hessian, gradient)
        coefficients = [value + delta for value, delta in zip(coefficients, step)]
        iterations = iteration + 1
        if max(abs(delta) for delta in step) < LOGISTIC_CONVERGENCE:
            break
    return LogisticModel(tuple(feature_names), standardizer,
                         tuple(coefficients), iterations)


def model_matrix(rows: Sequence[dict[str, str]], feature_names: Sequence[str]) -> list[list[float]]:
    matrix = []
    for row in rows:
        values = []
        for name in feature_names:
            if name == "derived__direction_long":
                values.append(1.0 if row["predictor__direction"] == "long" else 0.0)
            else:
                values.append(float(row[name]))
        matrix.append(values)
    return matrix


def calibration(probabilities: Sequence[float], y: Sequence[int]) -> tuple[float | None, float | None]:
    if not probabilities or len(set(y)) != 2:
        return None, None
    logits = [[math.log(max(1e-15, min(1 - 1e-15, value)) /
                        (1 - max(1e-15, min(1 - 1e-15, value))))]
              for value in probabilities]
    try:
        model = fit_logistic(logits, y, ["prediction_logit"], 0.0)
    except ValueError:
        return None, None
    # Convert standardized calibration coefficient back to raw-logit scale.
    slope = model.coefficients[1] / model.standardizer.scales[0]
    intercept = model.coefficients[0] - slope * model.standardizer.means[0]
    return intercept, slope


def evaluation(probabilities: Sequence[float], y: Sequence[int]) -> dict[str, object]:
    score_auc = auc(probabilities, y)
    score_ap = average_precision(probabilities, y)
    brier = (sum((value - label) ** 2 for value, label in zip(probabilities, y)) /
             len(y)) if y else None
    calibration_intercept, calibration_slope = calibration(probabilities, y)
    tp = sum(value >= 0.5 and label == 1 for value, label in zip(probabilities, y))
    fp = sum(value >= 0.5 and label == 0 for value, label in zip(probabilities, y))
    tn = sum(value < 0.5 and label == 0 for value, label in zip(probabilities, y))
    fn = sum(value < 0.5 and label == 1 for value, label in zip(probabilities, y))
    return {"n": len(y), "saved": sum(y), "harmed": len(y) - sum(y),
            "auc": score_auc, "ap": score_ap, "brier": brier,
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def write_table(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def analyze(runs: list[RunData], output_dir: Path, smoke: bool = False) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = [run for run in runs if run.role == "primary"]
    diagnostic = [run for run in runs if run.role == "diagnostic"]
    counts_rows = []
    for run in runs:
        counts = {name: sum(row["outcome__outcome_class"] == name for row in run.rows)
                  for name in ("SAVED", "HARMED", "UNCHANGED")}
        counts_rows.append((run.model_id, run.role, run.start, run.end, len(run.rows),
                            counts["SAVED"], counts["HARMED"], counts["UNCHANGED"],
                            run.metadata["joined_count"], run.metadata["join_failure_count"],
                            run.metadata["feature_layout_identity"], len(run.predictors)))
    write_table(output_dir / "counts.tsv",
                ("model_id", "cohort_role", "window_start", "window_end",
                 "activated_count", "saved_count", "harmed_count", "unchanged_count",
                 "joined_count", "join_failure_count", "feature_layout_identity",
                 "usable_predictor_columns"), counts_rows)

    scopes: list[tuple[str, list[RunData]]] = []
    for run in runs:
        scopes.append((f"model_{run.model_id}_{run.start[:4]}", [run]))
    for start, end in WINDOWS:
        scopes.append((f"primary_pooled_{start[:4]}",
                       [run for run in primary if (run.start, run.end) == (start, end)]))
    scopes.append(("primary_pooled_both_windows", primary))
    scopes.extend((f"diagnostic_1805_{start[:4]}",
                   [run for run in diagnostic if (run.start, run.end) == (start, end)])
                  for start, end in WINDOWS)
    univariate_rows = []
    baseline = ("predictor__directional_probability",
                "predictor__normalized_directional_confidence",
                "predictor__predicted_class", "predictor__direction")
    for scope_name, scope_runs in scopes:
        rows = binary_rows(scope_runs)
        if not scope_runs:
            continue
        common = set.intersection(*(set(run.predictors) for run in scope_runs))
        categorical = set.union(*(set(run.categorical) for run in scope_runs))
        ordered = list(baseline) + sorted(common - set(baseline))
        for predictor in ordered:
            if predictor in categorical:
                metrics = categorical_metrics(rows, predictor)
            else:
                metrics = [{"predictor": predictor, **numeric_metric(rows, predictor)}]
            for metric in metrics:
                univariate_rows.append((scope_name, metric["predictor"],
                    "categorical_prevalence_difference" if predictor in categorical
                    else "numeric_standardized_mean_difference",
                    metric["saved_n"], metric["harmed_n"], fmt(metric["saved_mean"]),
                    fmt(metric["saved_median"]), fmt(metric["harmed_mean"]),
                    fmt(metric["harmed_median"]), fmt(metric["effect"]),
                    fmt(metric["auc"]), metric["orientation"], fmt(metric["ap"])))
    write_table(output_dir / "univariate.tsv",
                ("scope", "predictor", "effect_type", "saved_n", "harmed_n",
                 "saved_mean_or_prevalence", "saved_median", "harmed_mean_or_prevalence",
                 "harmed_median", "effect", "roc_auc", "auc_orientation",
                 "average_precision"), univariate_rows)

    temporal_rows = []
    all_predictors = sorted(set.intersection(*(set(run.predictors) for run in primary)))
    temporal_scopes = [("primary_pooled", "ALL", primary)] + [
        (f"model_{model}", str(model),
         [run for run in primary if run.model_id == model])
        for model in PRIMARY_MODELS]
    for scope_name, model_id, scope_runs in temporal_scopes:
        if not scope_runs:
            continue
        scope_predictors = sorted(set.intersection(
            *(set(run.predictors) for run in scope_runs)))
        ordered_predictors = baseline + tuple(
            name for name in scope_predictors if name not in baseline)
        for predictor in ordered_predictors:
            by_year = []
            for start, end in WINDOWS:
                selected = [run for run in scope_runs
                            if (run.start, run.end) == (start, end)]
                rows = binary_rows(selected)
                categorical = set().union(*(set(run.categorical)
                                            for run in selected))
                metric = (numeric_metric(rows, predictor)
                          if predictor not in categorical else None)
                by_year.append(metric)
            if by_year[0] is None or by_year[1] is None:
                continue
            sign_2025 = (None if by_year[0]["effect"] is None else
                         math.copysign(1, by_year[0]["effect"]))
            sign_2026 = (None if by_year[1]["effect"] is None else
                         math.copysign(1, by_year[1]["effect"]))
            model_ids = {run.model_id for run in scope_runs}
            support = sum(1 for model in model_ids if all(
                any(run.model_id == model and run.start == start and
                    {row["outcome__outcome_class"] for row in run.rows}.
                    issuperset({"SAVED", "HARMED"})
                    for run in scope_runs) for start, _ in WINDOWS))
            temporal_rows.append((scope_name, model_id, predictor,
                                  fmt(by_year[0]["effect"]),
                                  fmt(by_year[1]["effect"]),
                                  "same" if sign_2025 is not None and
                                  sign_2025 == sign_2026 else
                                  "different_or_undefined",
                                  fmt(by_year[0]["auc"]),
                                  fmt(by_year[1]["auc"]),
                                  fmt(by_year[0]["ap"]),
                                  fmt(by_year[1]["ap"]), support))
    write_table(output_dir / "temporal_stability.tsv",
                ("scope", "model_id", "predictor", "effect_2025",
                 "effect_2026", "effect_sign_stability",
                 "roc_auc_2025", "roc_auc_2026", "average_precision_2025",
                 "average_precision_2026", "models_estimable_both_windows"), temporal_rows)

    train_runs = [run for run in primary if run.start == WINDOWS[0][0]]
    oot_runs = [run for run in primary if run.start == WINDOWS[1][0]]
    train_rows, oot_rows = binary_rows(train_runs), binary_rows(oot_runs)
    common_features = sorted(set.intersection(*(set(run.predictors) for run in primary)))
    model_features = ["predictor__directional_probability",
                      "predictor__normalized_directional_confidence",
                      "derived__direction_long"] + [name for name in common_features
                        if name not in {"predictor__predicted_class",
                                        "predictor__direction",
                                        "predictor__directional_probability",
                                        "predictor__normalized_directional_confidence"}]
    comparator_features = ["predictor__normalized_directional_confidence"]
    model_rows = []
    coefficient_rows = []
    evaluations: dict[str, dict[str, object]] = {}
    for model_name, features in (("full_causal_common_vector", model_features),
                                 ("confidence_only", comparator_features)):
        model = fit_logistic(model_matrix(train_rows, features), labels(train_rows), features)
        for period, rows in (("train_2025", train_rows), ("oot_2026", oot_rows)):
            probabilities = model.probabilities(model_matrix(rows, features))
            score = evaluation(probabilities, labels(rows))
            evaluations[f"{model_name}_{period}"] = score
            model_rows.append((model_name, period, score["n"], score["saved"],
                               score["harmed"], fmt(score["auc"]), fmt(score["ap"]),
                               fmt(score["brier"]), fmt(score["calibration_intercept"]),
                               fmt(score["calibration_slope"]), score["tn"], score["fp"],
                               score["fn"], score["tp"], "0.50", L2_STRENGTH,
                               "unweighted", model.iterations))
        coefficient_rows.append((model_name, "(intercept)", model.coefficients[0]))
        coefficient_rows.extend((model_name, feature, coefficient)
                                for feature, coefficient in
                                zip(features, model.coefficients[1:]))
    write_table(output_dir / "multivariate_metrics.tsv",
                ("model", "period", "n", "saved", "harmed", "roc_auc",
                 "average_precision", "brier_score", "calibration_intercept",
                 "calibration_slope", "tn", "fp", "fn", "tp", "threshold",
                 "l2_strength", "class_weighting", "iterations"), model_rows)
    write_table(output_dir / "coefficients.tsv", ("model", "feature", "coefficient"),
                ((name, feature, fmt(value)) for name, feature, value in coefficient_rows))

    full = evaluations["full_causal_common_vector_oot_2026"]
    confidence = evaluations["confidence_only_oot_2026"]
    estimable_models = sum(1 for model in PRIMARY_MODELS if all(
        any(run.model_id == model and run.start == start and
            {row["outcome__outcome_class"] for row in run.rows}.issuperset({"SAVED", "HARMED"})
            for run in primary) for start, _ in WINDOWS))
    robust = (full["auc"] is not None and confidence["auc"] is not None and
              full["auc"] >= 0.60 and full["auc"] - confidence["auc"] >= 0.03 and
              estimable_models >= 3 and full["saved"] >= 20)
    weak = (full["auc"] is not None and (abs(full["auc"] - 0.5) >= 0.05 or
            (confidence["auc"] is not None and abs(confidence["auc"] - 0.5) >= 0.05)))
    verdict = ("ROBUST CAUSAL PREDICTIVE SIGNAL IDENTIFIED" if robust else
               "WEAK / UNSTABLE CAUSAL PREDICTIVE SIGNAL" if weak else
               "NO USEFUL CAUSAL PREDICTIVE SIGNAL IDENTIFIED")

    counts_by_window = {}
    for start, end in WINDOWS:
        rows = [row for run in primary if (run.start, run.end) == (start, end)
                for row in run.rows]
        counts_by_window[start[:4]] = {name: sum(row["outcome__outcome_class"] == name
                                                  for row in rows)
                                         for name in ("SAVED", "HARMED", "UNCHANGED")}
    report = ["# LSTM Phase 19C — Causal Pre-Entry Predictability",
              "", f"Schema: `{SCHEMA}`  ", f"Analysis plan: `{PLAN}`", "",
              "This is a causal-predictability diagnostic, not a trading strategy. No threshold, stop distance, activation rule, or multiplier was optimized.", "",
              "## Leakage audit", "",
              "PASS. Predictors are limited to frozen model outputs at the decision timestamp, the exact completed Tensor decision-row prefix consumed by each persisted model, and its four causal lookback-return suffix values. Every source artifact reports zero join failures. All Phase 19B post-entry fields remain `outcome__*` columns and are excluded from every predictor matrix.", "",
              "## Frozen primary counts", ""]
    for year in ("2025", "2026"):
        count = counts_by_window[year]
        report.append(f"- {year}: SAVED {count['SAVED']}, HARMED {count['HARMED']}, UNCHANGED {count['UNCHANGED']}.")
    report += ["", "## Highlighted model stability", ""]
    for model_id in (1745, 1743, 1692):
        symbol = FROZEN_COHORT[model_id][0]
        year_parts = []
        for start, end in WINDOWS:
            selected = [run for run in primary if run.model_id == model_id and
                        (run.start, run.end) == (start, end)]
            rows = binary_rows(selected)
            metric = numeric_metric(
                rows, "predictor__normalized_directional_confidence")
            year_parts.append(
                f"{start[:4]} n={len(rows)} (SAVED {sum(labels(rows))}, "
                f"HARMED {len(rows) - sum(labels(rows))}), confidence AUC "
                f"{fmt(metric['auc'])}, AP {fmt(metric['ap'])}")
        report.append(f"- {symbol} / model {model_id}: " + "; ".join(year_parts) + ".")
    report += ["", "These highlighted results are descriptive model × window diagnostics; every causal feature's effect size, AUC, and AP is retained in `univariate.tsv`, with cross-window comparisons in `temporal_stability.tsv`.",
               "", "## Fixed 2025 → 2026 diagnostics", "",
               f"- Full common causal vector: 2026 ROC AUC {fmt(full['auc'])}, average precision {fmt(full['ap'])}, Brier {fmt(full['brier'])}; n={full['n']} (SAVED {full['saved']}, HARMED {full['harmed']}).",
               f"- Confidence-only comparator: 2026 ROC AUC {fmt(confidence['auc'])}, average precision {fmt(confidence['ap'])}, Brier {fmt(confidence['brier'])}.",
               f"- The full model uses {len(model_features)} frozen, untuned columns common to every primary model. Logistic regression is unweighted with fixed L2={L2_STRENGTH}; numeric standardization uses 2025 training data only.", "",
               "Undefined metrics always mean that one target class was absent; no smoothing or synthetic observations were used. Inferential p-values were not reported, so no multiple-comparison correction was applicable. See `counts.tsv`, `univariate.tsv`, and `temporal_stability.tsv` for model × window evidence, including models 1745, 1743, and 1692.", "",
               "## Scientific verdict", "", verdict, "",
               "This verdict does not authorize a new strategy. Any trading rule requires a separately frozen specification and independent validation.", ""]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    artifacts = [output_dir / name for name in ("counts.tsv", "univariate.tsv",
                 "temporal_stability.tsv", "multivariate_metrics.tsv",
                 "coefficients.tsv", "report.md")]
    metadata_rows = [("schema_identity", SCHEMA), ("analysis_plan_identity", PLAN),
                     ("frozen_cohort_complete", str(not smoke).lower()),
                     ("diagnostic_model_excluded_from_primary_fit", "true"),
                     ("train_window", "2025-01-01/2026-01-01"),
                     ("oot_window", "2026-01-01/2026-09-01"),
                     ("logistic_l2_strength", fmt(L2_STRENGTH)),
                     ("hyperparameter_tuning", "false"),
                     ("class_weighting", "unweighted"),
                     ("profitability_threshold_search", "false"),
                     ("scientific_verdict", verdict)]
    metadata_rows.extend((f"artifact_hash:{path.name}", fnv1a64(path.read_bytes()))
                         for path in artifacts)
    canonical = "".join(f"{key}={value};" for key, value in metadata_rows)
    metadata_rows.append(("analysis_result_hash", fnv1a64(canonical.encode())))
    write_table(output_dir / "analysis.metadata.tsv", ("key", "value"), metadata_rows)
    return {"verdict": verdict, "full_oot": full, "confidence_oot": confidence,
            "train_n": len(train_rows), "oot_n": len(oot_rows),
            "common_feature_count": len(model_features)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete-smoke", action="store_true")
    args = parser.parse_args()
    runs = load_runs(args.input_dir, args.allow_incomplete_smoke)
    result = analyze(runs, args.output_dir, args.allow_incomplete_smoke)
    print(f"PHASE19C_ANALYSIS,schema_identity={SCHEMA},verdict={result['verdict']},"
          f"train_n={result['train_n']},oot_n={result['oot_n']},"
          f"full_oot_auc={fmt(result['full_oot']['auc'])},"
          f"confidence_oot_auc={fmt(result['confidence_oot']['auc'])},"
          f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
