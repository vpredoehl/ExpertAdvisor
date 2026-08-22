#!/usr/bin/env python3
"""Static guardrails for Profitability Phase 2A's diagnostic-only boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def function_body(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    marker = f"{name}("
    start = text.find(marker)
    if start < 0:
        raise AssertionError(f"missing function {name} in {path}")
    brace = text.find("{", start)
    if brace < 0:
        raise AssertionError(f"missing body for {name} in {path}")
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace : index + 1]
    raise AssertionError(f"unterminated body for {name} in {path}")


def assert_diagnostic_only(path: Path, names: list[str]) -> None:
    for name in names:
        body = function_body(path, name).lower()
        if "profitability" in body:
            raise AssertionError(
                f"{name} must not reference profitability evidence"
            )


scheduler = ROOT / "Sources" / "ExperimentScheduler.cpp"
policy = ROOT / "Sources" / "ContinuationPolicy.cpp"
scoring = ROOT / "Sources" / "ExperimentRecommendationScoring.cpp"

assert_diagnostic_only(
    scheduler,
    [
        "BetterContinuationRankCandidate",
        "RankContinuationSource",
        "EvaluateContinuationPolicy",
        "BetterContinuationAutoQueueCandidate",
        "DecideCheckpointPolicy",
    ],
)
assert_diagnostic_only(
    policy,
    [
        "ContinuationPolicySemanticCanonicalText",
        "BetterBestContinuationSource",
        "SelectContinuationSourceEvidence",
        "PreferContinuationEvidenceAtSameEpoch",
        "DeduplicateContinuationEvidence",
        "ContinuationEvidenceWatermark",
        "EvaluateContinuationTrend",
    ],
)
assert_diagnostic_only(
    scoring,
    [
        "ScoreExperimentRecommendation",
        "RankRecommendationScores",
    ],
)

status_body = function_body(scheduler, "RunContinuationStatusCommand")
if "SetTransactionReadOnly(w);" not in status_body:
    raise AssertionError("continuation status must use a read-only transaction")
if "SetTransactionReadWrite(w);" in status_body:
    raise AssertionError("continuation status must not use a read-write transaction")
control_body = function_body(scheduler, "RunContinuationPolicyControlCommand")
if "SetTransactionReadWrite(w);" not in control_body:
    raise AssertionError("continuation policy controls must remain write-capable")

loader_body = function_body(scheduler, "LoadContinuationEvidence")
if "ResolveExactFinalInferenceResult" not in loader_body:
    raise AssertionError("final continuation diagnostics must use exact provenance")
if "ORDER BY ier.completed_at DESC" in loader_body:
    raise AssertionError("final continuation diagnostics must not use a newest-row fallback")

print("ContinuationProfitabilityPolicyIsolationTests passed")
